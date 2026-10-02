# =====================================================================
# PARCHE — mejora del entrenamiento (BETO / DistilBETO / RoBERTuito)
# Aplicar igual en los tres notebooks. Cada bloque indica la celda.
# =====================================================================


# ---------------------------------------------------------------------
# [CELDA 6 — Configuración]  Reemplazar RUN_TAG y SEARCH_SPACE; añadir 2 constantes
# ---------------------------------------------------------------------
RUN_TAG = "b5x3t20_v4"   # carpeta NUEVA: no reutilizar estados/curvas de la v3

EVALS_POR_EPOCA = 4      # evaluación/checkpoint cada 25 % de época
PACIENCIA_EVALS = 3      # early stopping: 3 evaluaciones (~0.75 épocas) sin mejora

SEARCH_SPACE = {
    "preprocessing": ["PYSENTIMIENTO"],
    "learning_rate": [1e-5, 2e-5, 3e-5],     # antes 2e-5..5e-5
    "batch_size": [16, 32],
    "num_train_epochs": [3, 4, 5],
    "weight_decay": [0.01, 0.1],              # antes 4 valores con efecto pequeño
    "loss_mode": ["standard", "balanced"],
    "classifier_dropout": [0.1, 0.2, 0.3],
    "label_smoothing": [0.0, 0.1],            # NUEVO
    "lr_decay": [1.0, 0.9, 0.8],              # NUEVO (1.0 = sin decay por capas)
}
SEARCH_SPACE_REQUERIDO = set(SEARCH_SPACE)   # incluye las dos claves nuevas
# (Actualiza también _ruta_ckpt_ejemplo con un id más largo, p. ej.
#  "P_l3_b32_e5_w10_b_d3_s1_r8", para la comprobación de longitud en Windows.)


# ---------------------------------------------------------------------
# [CELDA 16]  Reemplazar la clase ClasificadorTrainer completa
# ---------------------------------------------------------------------
class ClasificadorTrainer(Trainer):
    """Pérdida ponderada opcional + label smoothing (solo train) + LR decay por capas."""

    def __init__(self, *args, class_weights=None, label_smoothing=0.0, lr_decay=1.0, **kwargs):
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)
        self.lr_decay = float(lr_decay)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        weights = (
            self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        )
        # Smoothing SOLO al entrenar: la pérdida de validación sigue siendo CE estándar.
        ls = self.label_smoothing if model.training else 0.0
        loss = F.cross_entropy(outputs.logits, labels, weight=weights, label_smoothing=ls)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if self.lr_decay >= 1.0:
            return super().create_optimizer()

        patron = re.compile(r"\.layer\.(\d+)\.")  # vale para BERT, RoBERTa y DistilBERT
        n_capas = 1 + max(
            int(m.group(1))
            for n, _ in self.model.named_parameters()
            if (m := patron.search(n))
        )
        base_lr, wd = self.args.learning_rate, self.args.weight_decay
        grupos = {}
        for nombre, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            m = patron.search(nombre)
            if m:
                profundidad = n_capas - int(m.group(1))   # última capa = 1
            elif "embeddings" in nombre:
                profundidad = n_capas + 1                 # embeddings = LR más bajo
            else:
                profundidad = 0                           # cabeza/pooler = LR base
            sin_wd = nombre.endswith("bias") or "LayerNorm" in nombre or "layer_norm" in nombre
            clave = (base_lr * self.lr_decay**profundidad, 0.0 if sin_wd else wd)
            grupos.setdefault(clave, []).append(p)

        self.optimizer = torch.optim.AdamW(
            [{"params": ps, "lr": lr, "weight_decay": w} for (lr, w), ps in grupos.items()]
        )
        return self.optimizer


# ---------------------------------------------------------------------
# [CELDA 18]  Reemplazar argumentos_entrenamiento (nuevo parámetro n_train)
# ---------------------------------------------------------------------
def argumentos_entrenamiento(carpeta, cfg, seed, con_validacion=True, epochs=None, n_train=None):
    params = {
        "output_dir": str(carpeta),
        "learning_rate": cfg["learning_rate"],
        "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": cfg["batch_size"],
        "num_train_epochs": float(epochs if epochs is not None else cfg["num_train_epochs"]),
        "weight_decay": cfg["weight_decay"],
        "warmup_ratio": 0.10,
        "optim": "adamw_torch",
        "load_best_model_at_end": bool(con_validacion),
        "save_total_limit": 2,
        "save_only_model": False,
        "save_safetensors": True,
        "report_to": "none",
        "disable_tqdm": True,
        "restore_callback_states_from_checkpoint": True,
        "dataloader_num_workers": 0,
        "seed": seed,
        "data_seed": seed,
        "bf16": USE_BF16,
        "fp16": USE_FP16,
    }
    sig = inspect.signature(TrainingArguments.__init__).parameters
    clave_eval = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"

    if con_validacion:
        if n_train is None:
            raise ValueError("Con validación hace falta n_train para calcular eval_steps.")
        pasos_epoca = int(np.ceil(n_train / cfg["batch_size"]))
        eval_steps = max(1, pasos_epoca // EVALS_POR_EPOCA)
        params.update(
            {
                "logging_strategy": "steps",
                "logging_steps": eval_steps,
                "save_strategy": "steps",
                "save_steps": eval_steps,
                "eval_steps": eval_steps,
                clave_eval: "steps",
                "metric_for_best_model": "f1_macro",
                "greater_is_better": True,
            }
        )
    else:  # refit: sin validación, guardado por época como antes
        params.update({"logging_strategy": "epoch", "save_strategy": "epoch", clave_eval: "no"})

    return TrainingArguments(**{k: v for k, v in params.items() if k in sig})


# ---------------------------------------------------------------------
# [CELDA 22]  Cambios puntuales
# ---------------------------------------------------------------------
# (a) config_id debe incluir los parámetros nuevos (si no, dos configs distintas
#     compartirían carpeta y estado). Reemplazar la función:
def config_id(cfg):
    prep = {"PYSENTIMIENTO": "P"}[cfg["preprocessing"]]
    lr = int(round(float(cfg["learning_rate"]) * 1e5))
    bs = int(cfg["batch_size"])
    ep = int(cfg["num_train_epochs"])
    wd = int(round(float(cfg["weight_decay"]) * 100))
    loss = {"standard": "s", "balanced": "b"}[cfg["loss_mode"]]
    drop = int(round(float(cfg["classifier_dropout"]) * 10))
    ls = int(round(float(cfg.get("label_smoothing", 0.0)) * 10))
    lrd = int(round(float(cfg.get("lr_decay", 1.0)) * 10))
    return f"{prep}_l{lr}_b{bs}_e{ep}_w{wd}_{loss}_d{drop}_s{ls}_r{lrd}"


# (b) La mejor "época" ahora puede ser fraccionaria (1.25, 1.5, ...). Reemplazar:
def obtener_mejor_epoca_desde_curva(curva, max_epochs):
    if curva.empty or "val_f1_macro" not in curva or curva["val_f1_macro"].dropna().empty:
        return float(max_epochs)
    epoca = float(curva.loc[curva["val_f1_macro"].idxmax(), "epoch"])
    return float(np.clip(round(epoca * EVALS_POR_EPOCA) / EVALS_POR_EPOCA, 0.25, max_epochs))


# (c) Dentro de ejecutar_inner_fit_reanudable, cambiar estas líneas:
#
#   args = argumentos_entrenamiento(rutas["ckpt"], cfg, seed_modelo,
#                                   con_validacion=True, n_train=len(idx_train))
#   callbacks = [EarlyStoppingCallback(early_stopping_patience=PACIENCIA_EVALS)]
#
#   ClasificadorTrainer(..., class_weights=class_weights,
#                       label_smoothing=cfg["label_smoothing"],
#                       lr_decay=cfg["lr_decay"])
#
#   y en el json_atomico:  "best_epoch": float(best_epoch),
#
# (d) En la rama "REUTILIZADO" del mismo bloque:  float(previo["best_epoch"])
#
# (e) En graficar_mejor_trial y en la celda 40 el eje x ya funciona con épocas
#     fraccionarias; solo quita ax.set_xticks(range(...)) o usa paso 0.5.


# ---------------------------------------------------------------------
# [CELDA 24]  sugerir_config: añadir las dos claves; refit_epochs fraccionario
# ---------------------------------------------------------------------
#   "label_smoothing": trial.suggest_categorical("label_smoothing", SEARCH_SPACE["label_smoothing"]),
#   "lr_decay": trial.suggest_categorical("lr_decay", SEARCH_SPACE["lr_decay"]),
#
# En objetivo():
#   epocas.append(float(best_epoch))
#   trial.set_user_attr("best_epochs_inner", [float(e) for e in epocas])
#   trial.set_user_attr(
#       "refit_epochs",
#       float(np.clip(round(np.median(epocas) * EVALS_POR_EPOCA) / EVALS_POR_EPOCA,
#                     0.25, cfg["num_train_epochs"])),
#   )


# ---------------------------------------------------------------------
# [CELDA 26 — refit_externo]  pasar los parámetros nuevos al Trainer
# ---------------------------------------------------------------------
#   ClasificadorTrainer(..., class_weights=cw,
#                       label_smoothing=cfg["label_smoothing"],
#                       lr_decay=cfg["lr_decay"])
#
# (num_train_epochs acepta float: 1.75 épocas entrena 1 época y 3/4.)


# ---------------------------------------------------------------------
# [CELDA 28 — ejecutar_variante]
#   refit_epochs = float(study.best_trial.user_attrs["refit_epochs"])
# [CELDA 38]  cfg["refit_epochs"] = float(estudio.best_trial.user_attrs["refit_epochs"])
# [CELDA 40]  refit = float(cfg["refit_epochs"])
# [CELDA 43/44 test]  si usa refit_epochs con int(...), cambiar a float(...)
# [Celda de diagnóstico]  pasar label_smoothing y lr_decay a ClasificadorTrainer
# ---------------------------------------------------------------------

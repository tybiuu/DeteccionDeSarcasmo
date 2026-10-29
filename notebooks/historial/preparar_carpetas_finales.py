"""Copia los resultados ya calculados de BETO, DistilBETO y RoBERTuito a las carpetas que usan los
notebooks finales (04a_BETO, 04b_DistilBETO, 04c_RoBERTuito), para que al ejecutarlos se reanuden
desde lo ya hecho en lugar de repetir la Nested CV, Optuna y el test.

Uso (desde la carpeta notebooks/):   py -m uv run python preparar_carpetas_finales.py
- No borra ni modifica las carpetas originales: solo COPIA.
- Si la carpeta final ya existe, no la toca (usa --forzar para reemplazarla).
- Solo procesa los modelos cuya carpeta original exista en esta máquina.
"""
import json
import shutil
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
MODELOS = {"beto": "b5x3t20", "distilbeto": "d5x3t20", "robertuito": "r5x3t20"}
SUFIJO = "_v6"
EXTENSIONES_TEXTO = {".csv", ".json", ".log", ".txt", ".md"}
FIXED_TRAINING = {
    "optim": "adamw_torch",
    "early_stopping": "parada_temprana_f1_sin_checkpoints",
    "refit_epoch_strategy": "median_inner_best_epoch_rounded_same_scheduler",
    "loss": "ce_label_smoothing_train_only",
    "optimizer": "adamw_layerwise_lr_decay",
    "regularizacion": "dropout_interno_y_mascara_fijos",
    "metricas": "nll_y_ece_sin_pesos",
    "curvas": "train_eval_loss_epoch_csv_best_outer_png",
    "checkpoint": "refit_and_test_only",
}


def main(forzar=False):
    hechos = 0
    for pref, tag in MODELOS.items():
        origen = DATA / f"resultados_{pref}_grouped_5x3_tpe20{SUFIJO}"
        destino = DATA / f"resultados_{pref}_grouped_5x3_tpe20"
        if not origen.exists():
            print(f"[{pref}] no hay resultados previos en esta máquina ({origen.name}); se omite.")
            continue
        if destino.exists():
            if not forzar:
                print(f"[{pref}] {destino.name} ya existe; no se toca (usa --forzar para reemplazarla).")
                continue
            shutil.rmtree(destino)
        print(f"[{pref}] copiando {origen.name} -> {destino.name} ...")
        shutil.copytree(origen, destino)
        seg_viejo = destino / "seguimiento" / f"{tag}{SUFIJO}"
        if seg_viejo.exists():
            seg_viejo.rename(destino / "seguimiento" / tag)
        cambios = 0
        for ruta in destino.rglob("*"):
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_TEXTO:
                texto = ruta.read_text(encoding="utf-8", errors="surrogateescape")
                nuevo = texto.replace(f"tpe20{SUFIJO}", "tpe20").replace(f"{tag}{SUFIJO}", tag)
                if nuevo != texto:
                    ruta.write_text(nuevo, encoding="utf-8", errors="surrogateescape")
                    cambios += 1
        for cfg_path in (destino / "seguimiento" / tag).glob("*/config_ejecucion.json"):
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            ft = cfg.get("fixed_training", {})
            conservar = {k: ft[k] for k in ("warmup_fraccion", "early_stopping_patience") if k in ft}
            cfg["fixed_training"] = {**FIXED_TRAINING, **conservar}
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{pref}] listo: {cambios} archivos con rutas actualizadas. El original queda intacto.")
        hechos += 1
    print(f"\nModelos preparados: {hechos}. Ahora ejecuta los notebooks 04 de esta máquina.")


if __name__ == "__main__":
    main(forzar="--forzar" in sys.argv)

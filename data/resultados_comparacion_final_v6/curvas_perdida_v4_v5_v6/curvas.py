import json, sys
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
D = '/mnt/user-data/uploads/data'
MOD = [('beto','BETO'),('distilbeto','DistilBETO'),('robertuito','RoBERTuito')]
VAR = [('mx','MX'),('es','ES'),('cu','CU')]
AZ, NA = '#3B75AF', '#EF8636'

def datos(m, v):
    d = pd.read_csv(f'{D}/resultados_{m}_grouped_5x3_tpe20_{v}/{m}_diagnostico_convergencia.csv')
    c = json.load(open(f'{D}/resultados_{m}_grouped_5x3_tpe20_{v}/{m}_config_final.json'))
    return d, c

def curva(d, var, col):
    s = d[d.variante == var].groupby('epoch_r')[col].mean()
    return s.index.values, s.values

def figura(v, titulo, salida):
    fig, ax = plt.subplots(3, 3, figsize=(15, 9), sharex=True, sharey=True)
    for i, (m, M) in enumerate(MOD):
        d, c = datos(m, v)
        for j, (var, V) in enumerate(VAR):
            a = ax[i, j]
            x, y = curva(d, var, 'train_loss'); a.plot(x, y, color=AZ, lw=1.8)
            x, y = curva(d, var, 'val_loss'); a.plot(x, y, color=NA, lw=1.8)
            e = c[var]['refit_epochs']; fin = c[var]['num_train_epochs']
            a.axvline(e, color='k', ls=':', lw=1.1)
            a.axvspan(e, fin, color='0.85', alpha=.4)
            a.grid(alpha=.3); a.set_xlim(-0.25, 6.25); a.set_ylim(0.06, 0.76)
            if i == 0: a.set_title(V, fontsize=13)
            if j == 0: a.set_ylabel(f'{M}\nPérdida')
            if i == 2: a.set_xlabel('Época')
    fig.legend([Line2D([], [], color=AZ, lw=1.8), Line2D([], [], color=NA, lw=1.8), Line2D([], [], color='k', ls=':')],
               ['Train', 'Validación', 'Épocas del entrenamiento final'], loc='lower center', ncol=3, frameon=False)
    fig.suptitle(titulo, fontsize=14, y=.995)
    fig.tight_layout(rect=(0, .04, 1, .98))
    fig.savefig(salida + '.png', dpi=200); fig.savefig(salida + '.pdf'); plt.close(fig)

T = {'v4': 'v4 — sin dropout interno (base)',
     'v5': 'v5 — regularización dentro de la búsqueda de Optuna (dropout interno 0.1/0.2, congelar 0/25 %)',
     'v6': 'v6 (final) — dropout interno 0.2 fijo; BETO y DistilBETO además enmascaran 15 % de tokens'}
for v in ['v4', 'v5', 'v6']:
    figura(v, T[v], f'curvas_perdida_{v}')

# ---------- figura comparativa v4 vs v6 (y v5) ----------
fig, ax = plt.subplots(3, 3, figsize=(15, 9), sharex=True, sharey=True)
estilo = {'v4': dict(ls=(0, (1.5, 1.5)), lw=1.4, alpha=.55), 'v5': dict(ls='--', lw=1.4, alpha=.8), 'v6': dict(ls='-', lw=2.0, alpha=1)}
filas = []
for i, (m, M) in enumerate(MOD):
    for j, (var, V) in enumerate(VAR):
        a = ax[i, j]
        for v in ['v4', 'v5', 'v6']:
            d, c = datos(m, v)
            x, y = curva(d, var, 'train_loss'); a.plot(x, y, color=AZ, **estilo[v])
            x2, y2 = curva(d, var, 'val_loss'); a.plot(x2, y2, color=NA, **estilo[v])
            e = c[var]['refit_epochs']
            k = abs(x - e).argmin()
            filas.append(dict(modelo=M, variante=V, version=v, epocas_refit=e,
                              train_loss_refit=round(y[k], 3), val_loss_refit=round(y2[k], 3),
                              brecha_loss=round(y2[k] - y[k], 3), val_loss_min=round(y2.min(), 3)))
        a.grid(alpha=.3); a.set_xlim(-0.25, 6.25); a.set_ylim(0.06, 0.76)
        if i == 0: a.set_title(V, fontsize=13)
        if j == 0: a.set_ylabel(f'{M}\nPérdida')
        if i == 2: a.set_xlabel('Época')
h = [Line2D([], [], color=AZ, lw=2), Line2D([], [], color=NA, lw=2)] + [Line2D([], [], color='0.3', **estilo[v]) for v in ['v4', 'v5', 'v6']]
fig.legend(h, ['Train', 'Validación', 'v4', 'v5', 'v6 (final)'], loc='lower center', ncol=5, frameon=False)
fig.suptitle('Evolución de las curvas de pérdida: v4 → v5 → v6', fontsize=14, y=.995)
fig.tight_layout(rect=(0, .04, 1, .98))
fig.savefig('curvas_perdida_comparacion_v4_v5_v6.png', dpi=200); fig.savefig('curvas_perdida_comparacion_v4_v5_v6.pdf'); plt.close(fig)
t = pd.DataFrame(filas); t.to_csv('tabla_brecha_loss_v4_v5_v6.csv', index=False)
print(t.pivot_table(index=['modelo', 'variante'], columns='version', values=['brecha_loss', 'val_loss_refit']).round(3).to_string())

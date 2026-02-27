import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from config import CFG
from data.instances import INSTANCES
from core.distance import build_distance_matrix, gap
from algorithms.ga import ga_solver
from algorithms.aco import aco_solver
from algorithms.cbga import cbga_solver
from experiment.runner import run_all


if __name__ == '__main__':
    # Asegurar directorio de salida
    os.makedirs(CFG['OUT_DIR'], exist_ok=True)

    # Construir matrices vectorizadas
    DIST = {}
    valid_instances = []
    for name, inst in INSTANCES.items():
        DIST[name] = build_distance_matrix(inst['coords'], inst['type'])
        valid_instances.append(name)
        print(f"  {name:10s}  nodos={len(inst['coords']):3d}  optimo={inst['optimal']:6d}")
    
    if not DIST:
        print("Error: No se cargaron instancias. Verifica la ruta de TSPLIB.")
        exit(1)
        
    print('\nMatrices listas.\n')

    # Prueba rapida de los tres algoritmos
    for fn, label in [(ga_solver,'GA'),(aco_solver,'ACO'),(cbga_solver,'CBGA')]:
        b, _ = fn(DIST['berlin52'], 52, seed=0, budget=1) # 1s budget for fast check
        print(f"  Prueba {label} berlin52 (1s, seed=0): {b:.0f}  GAP={gap(b,7542):.2f}%")
    print()

    # Experimento completo paralelizado
    all_results = []
    for fn, name in [(ga_solver,'GA'),(aco_solver,'ACO'),(cbga_solver,'CBGA')]:
        icon = {'GA':'[GA]','ACO':'[ACO]','CBGA':'[CBGA]'}[name]
        print(f"{'='*60}\n{icon} Ejecutando {name}...\n{'='*60}")
        all_results.extend(run_all(fn, name, DIST))
    print('\nExperimento completo.\n')

    # --- Tabla resumen ---
    ALGOS = ['GA', 'ACO', 'CBGA']
    print('TABLA RESUMEN')
    print('='*90)
    hdr = (f"{'Inst':10s} {'Algo':6s} {'Optimo':7s} {'Best':7s} {'Avg':8s} "
           f"{'Std':7s} {'Peor':7s} {'GAP_b%':7s} {'GAP_a%':7s} {'T(s)':6s}")
    print(hdr); print('-'*len(hdr))
    prev = None
    for inst in valid_instances:
        opt = INSTANCES[inst]['optimal']
        for algo in ALGOS:
            sub   = [r for r in all_results if r['instance']==inst and r['algo']==algo]
            if not sub: continue
            bests = [r['best'] for r in sub]
            if inst != prev and prev is not None: print()
            prev = inst
            print(f"{inst:10s} {algo:6s} {opt:7d} {int(min(bests)):7d} "
                  f"{np.mean(bests):8.1f} {np.std(bests):7.1f} {int(max(bests)):7d} "
                  f"{gap(min(bests),opt):7.2f} {np.mean([gap(b,opt) for b in bests]):7.2f} "
                  f"{np.mean([r['time'] for r in sub]):6.2f}")

    # --- Guardar CSV ---
    csv_path = os.path.join(CFG['OUT_DIR'], 'resultados_modular.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['algo','instance','seed','best','gap','time'])
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in ['algo','instance','seed','best','gap','time']})
    print(f'\nCSV guardado: {csv_path}  ({len(all_results)} filas)')

    # --- Visualización (Basada en el código original) ---
    COLORS = CFG['COLORS']
    
    # FIG 1: Boxplots
    fig, axes = plt.subplots(1, len(valid_instances), figsize=(5*len(valid_instances), 6))
    if len(valid_instances) == 1: axes = [axes]
    fig.suptitle('Distribucion de Mejores Distancias (30 seeds)', fontsize=15, fontweight='bold')
    for ax, inst in zip(axes, valid_instances):
        opt = INSTANCES[inst]['optimal']
        data = [[r['best'] for r in all_results if r['instance']==inst and r['algo']==a] for a in ALGOS]
        bp = ax.boxplot(data, tick_labels=ALGOS, patch_artist=True, medianprops=dict(color='black', linewidth=2))
        for patch, a in zip(bp['boxes'], ALGOS):
            patch.set_facecolor(COLORS[a]); patch.set_alpha(0.75)
        ax.axhline(opt, color='red', linestyle='--', linewidth=1.5, label=f'Optimo ({opt})')
        ax.set_title(f'{inst}\n(optimo = {opt})', fontsize=11)
        ax.set_ylabel('Longitud del tour'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['OUT_DIR'], 'fig1_boxplots_modular.png'), dpi=130)
    plt.close()

    # FIG 2: Convergencia
    rows = (len(valid_instances) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 6*rows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    fig.suptitle('Curvas de Convergencia Promedio', fontsize=14, fontweight='bold')
    for i, inst in enumerate(valid_instances):
        ax = axes_flat[i]
        opt = INSTANCES[inst]['optimal']
        for algo in ALGOS:
            sub = [r for r in all_results if r['instance']==inst and r['algo']==algo]
            if not sub: continue
            ml = min(len(r['history']) for r in sub)
            mat = np.array([r['history'][:ml] for r in sub])
            mh, sh = mat.mean(0), mat.std(0)
            ax.plot(mh, color=COLORS[algo], label=algo, linewidth=2)
            ax.fill_between(range(ml), mh-sh, mh+sh, color=COLORS[algo], alpha=0.15)
        ax.axhline(opt, color='red', linestyle='--', linewidth=1.5, label='Optimo')
        ax.set_title(inst, fontsize=12); ax.set_xlabel('Iteracion'); ax.set_ylabel('best-so-far')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['OUT_DIR'], 'fig2_convergence_modular.png'), dpi=130)
    plt.close()

    # FIG 3: GAP bars
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(valid_instances)); width = 0.25
    for i, algo in enumerate(ALGOS):
        gv = []
        for inst in valid_instances:
            avg_gap = np.mean([r['gap'] for r in all_results if r['instance']==inst and r['algo']==algo])
            gv.append(avg_gap)
        bars = ax.bar(x_pos + i*width, gv, width, label=algo, color=COLORS[algo], alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, gv):
            ypos = val + 0.1 if val >= 0 else val - 0.1   # ← FIX 1
            va   = 'bottom'  if val >= 0 else 'top'        # ← FIX 1
            ax.text(bar.get_x()+bar.get_width()/2, ypos, f'{val:.1f}%', ha='center', va=va, fontsize=8, fontweight='bold')
    ax.set_xticks(x_pos + width); ax.set_xticklabels(valid_instances)
    ax.set_ylabel('GAP promedio (%)'); ax.set_title('GAP promedio respecto al optimo')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.9)            # ← FIX 2
    all_gaps = [np.mean([r['gap'] for r in all_results if r['instance']==inst and r['algo']==algo])
            for algo in ALGOS for inst in valid_instances]
    margin = (max(all_gaps) - min(all_gaps)) * 0.15
    ax.set_ylim(min(all_gaps) - margin, max(all_gaps) + margin)  # ← FIX 2
    plt.tight_layout()
    plt.savefig(os.path.join(CFG['OUT_DIR'], 'fig3_gap_bars_modular.png'), dpi=130)
    plt.close()

    print('\nTodo listo. Gráficos guardados en:', CFG['OUT_DIR'])

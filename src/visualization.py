# src/visualization.py
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional


def plot_optimization_flow(steps: List[Dict[str, Any]], save_path: Optional[str] = None):
    """
    График прогресса infidelity.
    """
    if not steps:
        print("No steps to visualize")
        return

    print(f"Total steps: {len(steps)}")

    step_nums = []
    infidelities = []
    methods = []
    improvements = []

    prev_inf = None
    for i, step in enumerate(steps):
        step_nums.append(i + 1)
        metadata = step.get('metadata', {})

        inf = metadata.get('output_infidelity')
        if inf is None:
            print(f"WARNING: No output_infidelity in step {i+1}")
            continue
        infidelities.append(inf)

        method = step['method']
        if 'optimize' in method and 'population' not in method:
            methods.append('Gradient')
        elif 'population' in method:
            methods.append('CMA-ES')
        elif 'multi_start' in method:
            methods.append('Multi-start')
        elif 'sequence' in method:
            methods.append('Sequence')
        else:
            methods.append(method[:15])

        if prev_inf is not None:
            improvements.append(prev_inf - inf)
        else:
            improvements.append(0)
        prev_inf = inf

    if not infidelities:
        print("No infidelity data available")
        return

    color_map = {
        'Gradient': '#1f77b4',
        'CMA-ES': '#2ca02c',
        'Multi-start': '#ff7f0e',
        'Sequence': '#d62728',
    }
    colors = [color_map.get(m, '#888888') for m in methods]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(step_nums[:len(infidelities)], infidelities, 'o-', linewidth=2,
                markersize=10, color='#333333', zorder=1, alpha=0.5)

    for i, (x, y, method, imp) in enumerate(zip(step_nums[:len(infidelities)], infidelities, methods, improvements)):
        ax.plot(x, y, 'o', markersize=12, color=colors[i], zorder=2,
                markeredgecolor='black', markeredgewidth=1.5)
        ax.annotate(method, (x, y), xytext=(0, 12), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors[i])
        ax.annotate(f'{y:.3e}', (x, y), xytext=(8, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        if imp > 0 and i > 0:
            ax.annotate(f'▼ {imp:.3e}', (x, y), xytext=(0, -20), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='#2e7d32')

    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel('Infidelity (log scale)', fontsize=12)
    ax.set_title('Optimization Progress - Infidelity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.set_xlim(0.5, len(step_nums) + 0.5)

    table_data = [[f"{i+1}", methods[i], f"{infidelities[i]:.3e}", f"{improvements[i]:.3e}" if improvements[i] > 0 else '-']
                  for i in range(len(infidelities))]
    table = ax.table(cellText=table_data,
                     colLabels=['Step', 'Method', 'Infidelity', 'Improvement'],
                     loc='bottom', bbox=[0, -0.4, 1, 0.25],
                     cellLoc='center', fontsize=8)
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
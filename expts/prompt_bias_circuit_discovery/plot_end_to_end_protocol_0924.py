"""Draw the proposed complete attention scope; no experimental measurements."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15})
    fig, ax = plt.subplots(figsize=(10, 6.5))
    names = ['Prompt P', 'Reasoning R', 'Readout Q']
    for i in range(3):
        for j in range(3):
            allowed = j <= i
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1,
                                   facecolor='#0072B2' if allowed else '#EEEEEE',
                                   edgecolor='white', linewidth=3))
            ax.text(j, i, f'{names[j][-1]} → {names[i][-1]}' if allowed else 'Future: blocked',
                    ha='center', va='center', color='white' if allowed else '#555555',
                    fontsize=16)
    ax.set(xlim=(-.5, 2.5), ylim=(2.5, -.5), xticks=range(3), yticks=range(3),
           xticklabels=names, yticklabels=names,
           xlabel='Source / key chunk', ylabel='Reader / query chunk')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out=Path('notes/images/end_to_end_protocol_0924')
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'masking_cells.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()

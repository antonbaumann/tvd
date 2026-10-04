"""Plot general-performance gains, with SEM across the three paired seeds."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with args.summary.open() as handle:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(handle)]
    fig, ax = plt.subplots(figsize=(6.5, 4.0), layout='constrained')
    def series(group, **style):
        group = sorted(group, key=lambda r: r['effective_lr'])
        ax.errorbar([r['effective_lr'] / 1e-5 for r in group],
                    [r['general_gain_bpt_mean'] for r in group],
                    yerr=[r['general_gain_bpt_sem'] for r in group], capsize=3, **style)
    series([r for r in rows if r['integration_lambda'] == 1], color='0.5',
           linestyle='--', marker='o', label=r'Full integration ($\lambda=1$)')
    for lr, color in zip((1e-5, 2e-5, 3e-5, 4e-5), ('#0072B2', '#009E73', '#D55E00', '#CC79A7')):
        series([r for r in rows if r['base_lr'] == lr], color=color, marker='o',
               label=rf'Base learning rate ${lr / 1e-5:g}\times10^{{-5}}$')
    ax.set(xlabel=r'Effective learning rate $\lambda\eta$ ($10^{-5}$)',
           ylabel='Mean general-performance gain (BPT)')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()

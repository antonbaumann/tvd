"""List, validate, or run the 72 paper configurations sequentially on four GPUs."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true', help='Run training; requires four visible CUDA GPUs')
    action.add_argument('--dry-run', action='store_true', help='Validate configurations without loading data or models')
    parser.add_argument('--index', type=int, help='Select one zero-based configuration index (0–71)')
    args = parser.parse_args()
    configs = sorted((Path(__file__).resolve().parents[1] / 'config/paper').glob('*.json'))
    if len(configs) != 72:
        parser.error(f'Expected 72 paper configurations; found {len(configs)}')
    if args.index is not None and not 0 <= args.index < len(configs):
        parser.error('--index must be between 0 and 71')
    for index, path in enumerate(configs):
        if args.index is not None and index != args.index:
            continue
        print(f'{index:02d} {path.name}', flush=True)
        if args.dry_run:
            # Validate in one process to avoid repeatedly importing the training stack.
            from experiments.pg19.scripts.train import run
            import json
            run(json.loads(path.read_text()), dry_run=True)
        elif args.run:
            subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                            '--nproc_per_node=4', '-m', 'experiments.pg19.scripts.train',
                            '--config', str(path)], check=True)


if __name__ == '__main__':
    main()

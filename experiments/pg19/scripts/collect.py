"""Reproduce the paper's PG-19 summaries from complete run logs or reference CSV."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def records(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metadata(directory):
    rows = records(directory / 'metadata.jsonl')
    configs = [r['config'] for r in rows if r['event'] == 'resolved_config']
    if len(configs) != 1:
        raise ValueError(f'{directory}: expected exactly one resolved configuration')
    return configs[0], rows


def trajectory(directory, limit=None, visited=None):
    """Follow checkpoint provenance, keeping only its saved prefix before a resume."""
    directory = directory.resolve()
    visited = set() if visited is None else set(visited)
    if directory in visited:
        raise ValueError('Resume provenance contains a cycle')
    visited.add(directory)
    cfg, meta = metadata(directory)
    if cfg.get('_runtime', {}).get('eval_only'):
        raise ValueError('Test-only evaluations do not belong in the training figure')
    if limit is None:
        completed = [r for r in meta if r['event'] == 'run_complete']
        if len(completed) != 1 or completed[0]['details']['books'] != cfg['data']['num_books']:
            raise ValueError(f'{directory}: run is incomplete')
        ends = [r for r in records(directory / 'train.jsonl') if r['event'] == 'run_end']
        if not ends or ends[-1]['status'] != 'complete':
            raise ValueError(f'{directory}: run did not finish successfully')
        limit = cfg['data']['num_books']
    initial, values = None, {}
    resume = cfg.get('_runtime', {}).get('resume')
    if resume:
        checkpoint = Path(resume)
        parent = checkpoint.parent.parent
        previous, events = metadata(parent)
        for field in ('seed', 'model', 'train', 'data', 'eval'):
            if cfg[field] != previous[field]:
                raise ValueError(f'Resumed configuration changes {field}')
        saved = [r for r in events if r['event'] == 'checkpoint' and Path(r['path']) == checkpoint]
        if len(saved) != 1:
            raise ValueError(f'Cannot identify the saved checkpoint prefix: {checkpoint}')
        boundary = saved[0]['sequence']
        if not 0 < boundary < limit:
            raise ValueError('Resume boundary must precede the requested trajectory end')
        _, initial, values = trajectory(parent, boundary, visited)
    for row in records(directory / 'eval.jsonl'):
        if row['event'] != 'eval_summary' or row['eval_scope'] != 'general':
            continue
        value = row['macro_bpt']
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('General BPT must be finite')
        if row['phase'] == 'initialization':
            if initial is not None:
                raise ValueError('Duplicate initialization measurement')
            initial = value
        if row['state'] == 'post_merge' and row['sequence'] <= limit:
            if row['sequence'] in values:
                raise ValueError('Overlapping post-merge measurements')
            values[row['sequence']] = value
    if initial is None or set(values) != set(range(1, limit + 1)):
        raise ValueError(f'{directory}: missing initialization or book measurements')
    return cfg, initial, values


def collect_logs(root):
    rows = []
    expected = paper_configs()
    for path in sorted(Path(root).rglob('metadata.jsonl')):
        cfg, meta = metadata(path.parent)
        if cfg.get('_runtime', {}).get('eval_only') or not any(r['event'] == 'run_complete' for r in meta):
            continue
        identity = key(cfg['train']['learning_rate'], cfg['train']['merge']['base_meta_lr'], cfg['seed'])
        if identity not in expected:
            raise ValueError(f'{path.parent}: unexpected paper configuration')
        reference = expected[identity]
        for field in ('model', 'train', 'eval'):
            if cfg[field] != reference[field]:
                raise ValueError(f'{path.parent}: {field} differs from the paper configuration')
        for field, value in reference['data'].items():
            if field != 'cache_dir' and cfg['data'].get(field) != value:
                raise ValueError(f'{path.parent}: data.{field} differs from the paper configuration')
        starts = [r for r in records(path.parent / 'train.jsonl') if r['event'] == 'run_start']
        if len(starts) != 1 or starts[0]['num_processes'] != 4:
            raise ValueError('Paper configurations require four training processes')
        cfg, initial, values = trajectory(path.parent)
        mean_bpt = statistics.mean(values.values())
        rows.append(dict(base_lr=cfg['train']['learning_rate'],
                         integration_lambda=cfg['train']['merge']['base_meta_lr'],
                         train_data_seed=cfg['data']['train_data_seed'], training_seed=cfg['seed'],
                         eval_data_seed=cfg['data']['eval_data_seed'], n_segments=len(values),
                         initial_general_bpt=initial, mean_general_post_merge_bpt=mean_bpt))
    return rows


def key(lr, lam, seed):
    return round(lr, 12), round(lam, 12), int(seed)


def paper_configs():
    expected = {}
    for path in (Path(__file__).resolve().parents[1] / 'config/paper').glob('*.json'):
        cfg = json.loads(path.read_text())
        expected[key(cfg['train']['learning_rate'], cfg['train']['merge']['base_meta_lr'], cfg['seed'])] = cfg
    return expected


def summarize(rows):
    expected = set(paper_configs())
    runs = {}
    for source in rows:
        r = {k: float(v) for k, v in source.items()}
        for field in ('train_data_seed', 'training_seed', 'eval_data_seed', 'n_segments'):
            if not r[field].is_integer():
                raise ValueError(f'{field} must be an integer')
            r[field] = int(r[field])
        if not all(math.isfinite(v) for v in r.values()):
            raise ValueError('Non-finite run value')
        if r['training_seed'] != r['train_data_seed'] or r['eval_data_seed'] != 0 or r['n_segments'] != 200:
            raise ValueError('Paper runs require 200 books, paired seeds, and evaluation seed 0')
        identity = key(r['base_lr'], r['integration_lambda'], r['training_seed'])
        if identity in runs:
            raise ValueError(f'Duplicate run for {identity}')
        r['effective_lr'] = r['base_lr'] * r['integration_lambda']
        r['mean_general_post_merge_score'] = -r['mean_general_post_merge_bpt']
        r['mean_general_gain_bpt'] = r['initial_general_bpt'] - r['mean_general_post_merge_bpt']
        runs[identity] = r
    if set(runs) != expected or len(expected) != 72:
        raise ValueError(f'Expected all 72 paper runs; missing {len(expected-set(runs))}, unexpected {len(set(runs)-expected)}')
    groups = defaultdict(list)
    for r in runs.values():
        baseline = runs[key(r['effective_lr'], 1, r['training_seed'])]
        r['matched_full_mean_score'] = baseline['mean_general_post_merge_score']
        r['advantage_over_matched_full'] = r['mean_general_post_merge_score'] - r['matched_full_mean_score']
        groups[key(r['base_lr'], r['integration_lambda'], 0)[:2]].append(r)
    summary = []
    metrics = {'initial_general_bpt': 'initial_general_bpt',
               'general_post_merge_bpt': 'mean_general_post_merge_bpt',
               'general_post_merge_score': 'mean_general_post_merge_score',
               'general_gain_bpt': 'mean_general_gain_bpt',
               'matched_full_score': 'matched_full_mean_score',
               'paired_advantage': 'advantage_over_matched_full'}
    for (lr, lam), group in sorted(groups.items()):
        item = dict(base_lr=lr, integration_lambda=lam, effective_lr=lr*lam, n_seeds=len(group),
                    n_segments_per_seed=200, eval_data_seed=0)
        for name, column in metrics.items():
            values = [r[column] for r in group]
            item[name+'_mean'] = statistics.mean(values)
            item[name+'_sd'] = statistics.stdev(values)
            item[name+'_sem'] = item[name+'_sd'] / math.sqrt(len(values))
        summary.append(item)
    return [runs[k] for k in sorted(runs)], summary


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--results-root', type=Path)
    source.add_argument('--runs-csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.runs_csv:
        with args.runs_csv.open() as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = collect_logs(args.results_root)
    runs, summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'runs.csv', runs)
    write_csv(args.output / 'summary.csv', summary)
    write_csv(args.output / 'matched_comparisons.csv', [r for r in summary if r['integration_lambda'] < 1])
    print(f'Wrote {len(runs)} runs, {len(summary)} summaries, and 12 matched comparisons to {args.output}')


if __name__ == '__main__':
    main()

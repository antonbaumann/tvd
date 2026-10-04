"""Uniform JSON configuration and environment-configured artifact roots."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def roots():
    result = {"REPO_ROOT": str(REPO_ROOT)}
    for name in ("TVD_DATA_ROOT", "TVD_RESULTS_ROOT"):
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"Set {name} to an absolute storage directory; see .env.example")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{name} must be absolute")
        result[name] = str(path)
    return result


def resolve(value, variables):
    if isinstance(value, dict):
        return {k: resolve(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, variables) for v in value]
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("${" + key + "}", replacement)
        if "${" in value:
            raise ValueError(f"Unresolved configuration variable: {value}")
    return value


def resolve_config(config):
    variables = roots()
    os.environ.setdefault('HF_HOME', str(Path(variables['TVD_DATA_ROOT']) / 'models'))
    os.environ.setdefault('HF_HUB_CACHE', str(Path(os.environ['HF_HOME']) / 'hub'))
    os.environ.setdefault('HF_DATASETS_CACHE', str(Path(variables['TVD_DATA_ROOT']) / 'cache/datasets'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    return resolve(config, variables)


def load_config(path):
    return resolve_config(json.loads(Path(path).read_text()))


def resolve_run_config(config, *, resume=None, eval_only=False):
    """Copy and resolve a training config, with explicit resume/evaluation options."""
    if '_runtime' in config:
        raise ValueError('Pass resume and eval_only as run() arguments, not _runtime config')
    config = resolve_config(config)
    config['_runtime'] = {'resume': str(resume.resolve()) if resume is not None else None,
                          'eval_only': eval_only}
    if resume is not None or eval_only:
        suffix = '-evaluation' if eval_only else '-resumed'
        config['output_dir'] += suffix
        config['_tvd']['run_id'] += suffix
    return config


def validate_train(config, experiment):
    context = config['_tvd']
    if context['experiment'] != experiment or context['training_seed'] != config['seed']:
        raise ValueError('Run metadata must match the experiment and training seed')
    for key in ('run_id', 'baseline', 'stream_seed', 'data_seed', 'L'):
        context[key]
    if experiment == 'scratch':
        paths = ('output_dir', 'dataset_dir', 'stream_dir')
    elif experiment in {'wildchat', 'pg19'}:
        paths = ('output_dir',)
    else:
        paths = ('output_dir', 'stream')
    for key in paths:
        if not Path(config[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute resolved path')
    if 'out_dir' in config:
        raise ValueError('Use output_dir')
    lam = context['lambda']
    if not isinstance(lam, (int, float)) or not 0 <= lam <= 1:
        raise ValueError('lambda must be between zero and one')
    if experiment == 'scratch':
        actual_lam = 1 if config['stream_name'] == 'iid_mixed' else config['merge_lr']
        if config['init_from'] not in ('scratch', 'resume'):
            raise ValueError('E5 starts from random initialization or resumes its checkpoint')
    else:
        train = config['train']
        for section, removed in ((train, {'lr', 'method'}), (config['eval'], {'final_eval'})):
            if section.keys() & removed:
                raise ValueError(f'Unsupported config keys: {sorted(section.keys() & removed)}')
        for key in ('learning_rate', 'weight_decay', 'adam_betas', 'adam_epsilon'):
            train[key]
        for key in ('enabled', 'final'):
            if type(config['eval'][key]) is not bool:
                raise TypeError(f'eval.{key} must be a boolean')
        merge = train['merge']
        enabled = merge['enabled']
        if type(enabled) is not bool:
            raise TypeError('train.merge.enabled must be a boolean')
        actual_lam = merge['base_meta_lr'] if enabled else 1
        if merge['mode'] != 'static' or merge['alpha'] != 0:
            raise ValueError('Only static lambda is retained')
        if train['optimizer_state'] != 'merge':
            raise ValueError('Paper experiments require merged optimizer state')
        if experiment == 'personas':
            if train['loss_mode'] != 'full_distillation' or train['reference_kl_beta'] != 0:
                raise ValueError('E2 requires feedback-conditioned self-distillation without reference KL')
            if train['distillation_support'] not in {'student_only', 'student_teacher_union'}:
                raise ValueError('distillation_support must be student_only or student_teacher_union')
    if lam != actual_lam:
        raise ValueError('Logged lambda differs from the configured integration coefficient')
    return config

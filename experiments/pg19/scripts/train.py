"""PG-19 book-sequential SFT with synchronized integration and sampled evaluation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from dotenv import load_dotenv
from shared.config import REPO_ROOT, resolve_run_config, validate_train
load_dotenv(REPO_ROOT / '.env', override=False)

import torch
from accelerate import PartialState
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from experiments.pg19.data import Blocks, BookBatchSampler, Collator, prepare, schedule
from shared import checkpoint as checkpoints
from shared.integration import apply_reptile_state, clone_floating_state, clone_optimizer_state, merge_adamw_optimizer_state
from shared.io import RUN_CONTEXT, JsonlLogger, evaluation_phase, prepare_output_dir, read_json, sha256, write_json, write_run_record
from shared.provenance import record_runtime, start_run, tokenizer_metadata


def validate_config(config):
    validate_train(config, 'pg19')
    data, train, evaluation = config['data'], config['train'], config['eval']
    for obj, keys, minimum in ((data, ('train_data_seed', 'eval_data_seed'), 0),
                               (data, ('num_books',), 1), (data, ('block_size',), 2),
                               (train, ('per_device_train_batch_size',), 1),
                               (evaluation, ('batch_size', 'max_batch_tokens'), 1)):
        for key in keys:
            if type(obj[key]) is not int or obj[key] < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
    for key, minimum in (('book_blocks', 1), ('retention_k', 0), ('test_blocks', 1)):
        if evaluation[key] is not None and (type(evaluation[key]) is not int or evaluation[key] < minimum):
            raise ValueError(f'eval.{key} must be null or an integer >= {minimum}')
    if not 0 < data['holdout_fraction'] < 1:
        raise ValueError('holdout_fraction must be between zero and one')
    if train['gradient_accumulation_steps'] != 1 or train['num_train_epochs'] != 1:
        raise ValueError('PG-19 requires accumulation 1 and one epoch')
    if train['lr_scheduler_type'] != 'constant' or train['packing'] is not False:
        raise ValueError('PG-19 requires constant scheduling and packing=false')
    if train['weight_decay'] != 0:
        raise ValueError('PG-19 requires zero weight decay')
    if train['mixed_precision'] not in ('no', 'bf16') or config['model']['torch_dtype'] != 'float32':
        raise ValueError('Use FP32 parameters and no/bf16 mixed precision')
    if not train['merge']['enabled'] or not evaluation['enabled']:
        raise ValueError('Book integration and evaluation must be enabled')
    if train['checkpoint_every_segments'] != 1:
        raise ValueError('Checkpoints are required after every book')
    if config['_tvd']['L'] is not None:
        raise ValueError('Use L=null: books have variable numbers of updates')
    if config['_tvd']['data_seed'] != data['train_data_seed']:
        raise ValueError('Logged data_seed must equal train_data_seed')
    config['_tvd'].update(stream_seed=data['train_data_seed'], train_data_seed=data['train_data_seed'],
                          eval_data_seed=data['eval_data_seed'])
    if evaluation['max_batch_tokens'] < data['block_size']:
        raise ValueError('Evaluation token budget must fit one block')
    if not math.isfinite(train['learning_rate']) or train['learning_rate'] <= 0:
        raise ValueError('Learning rate must be positive and finite')
    if config['wandb']['enabled'] and not config['wandb']['project']:
        raise ValueError('W&B project is required')
    return config


def reduce_sum(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor)
    return tensor


def rank_zero(runtime, fn):
    """Propagate rank-zero preparation errors instead of leaving peers at a barrier."""
    payload = [None, None]
    if runtime.is_main_process:
        try:
            payload[0] = fn()
        except (Exception, SystemExit) as error:
            payload[1] = f'{type(error).__name__}: {error}'
    if runtime.num_processes > 1:
        torch.distributed.broadcast_object_list(payload, src=0)
    if payload[1]:
        raise RuntimeError(payload[1])
    return payload[0]


def nll_sum(logits, labels):
    # Sum avoids NaNs for dummy ranks. Float32 cross entropy even under bf16 autocast.
    total = logits[..., :1, :1].sum() * 0
    for prediction, target in zip(logits[:, :-1], labels[:, 1:]):
        valid = target != -100
        total = total + torch.nn.functional.cross_entropy(prediction[valid].float(), target[valid], reduction='sum')
    return total


def metric(nll, tokens, blocks):
    loss = nll / tokens if tokens else None
    bpt = loss / math.log(2) if loss is not None else None
    return {'nll_sum': nll, 'target_tokens': tokens, 'tokens': tokens, 'num_blocks': blocks,
            'loss': loss, 'bpt': bpt, 'score': -bpt if bpt is not None else None}


def summary(metrics):
    nll = math.fsum(m['nll_sum'] for m in metrics)
    tokens = sum(m['target_tokens'] for m in metrics)
    macro = math.fsum(m['bpt'] for m in metrics) / len(metrics) if metrics else None
    micro = nll / tokens / math.log(2) if tokens else None
    return {'num_tasks': len(metrics), 'num_books': len(metrics), 'nll_sum': nll, 'target_tokens': tokens, 'tokens': tokens,
            'num_blocks': sum(m['num_blocks'] for m in metrics), 'macro_bpt': macro, 'micro_bpt': micro,
            'macro_loss': macro * math.log(2) if macro is not None else None,
            'micro_loss': nll / tokens if tokens else None,
            'score': -macro if macro is not None else None}


@contextmanager
def evaluation_context(model):
    # Saving every visible GPU's RNG on every rank creates unwanted CUDA contexts.
    state = checkpoints.rng_state(cuda_device=next(model.parameters()).device.index)
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        checkpoints.restore_rng(state)
        # Large evaluation logits otherwise leave the allocator holding most HBM.
        if next(model.parameters()).device.type == 'cuda':
            torch.cuda.empty_cache()


class Logger(JsonlLogger):
    def __init__(self, path, runtime, tracking=None):
        super().__init__(path, enabled=runtime.is_main_process)
        self.tracking = tracking

    def write(self, row):
        super().write(row)
        if self.tracking is None:
            return
        if row['event'] == 'eval_summary':
            prefix = f'eval/{row["eval_scope"]}/{row["state"]}'
        elif row['event'] == 'train_batch':
            prefix = 'train'
        elif row['event'] == 'segment_merge':
            prefix = 'merge'
        else:
            return
        values = {f'{prefix}/{k}': v for k, v in row.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
        self.tracking.log({'optimizer_step': row['step'], 'book': row['sequence'], **values})


class Evaluator:
    def __init__(self, config, manifest, dataset, collator, runtime, logger):
        self.config, self.manifest, self.dataset = config, manifest, dataset
        self.collator, self.runtime, self.logger = collator, runtime, logger
        self.cache, self.references = {}, {}
        self.eval_id = 0
        self.current_before = None
        self.model = None

    def score(self, book_ids):
        if not book_ids:
            return {}
        slots = {book_id: i for i, book_id in enumerate(book_ids)}
        work = [(book_id, block) for book_id in book_ids
                for block in self.manifest['books'][book_id]['eval_blocks']]
        work = work[self.runtime.process_index::self.runtime.num_processes]
        totals = torch.zeros((len(book_ids), 3), dtype=torch.float64, device=self.runtime.device)
        settings = self.config['eval']
        batch_size = min(settings['batch_size'], settings['max_batch_tokens'] // self.config['data']['block_size'])
        for start in range(0, len(work), batch_size):
            batch_refs = work[start:start + batch_size]
            inputs = self.collator([self.dataset.block(*ref) for ref in batch_refs])
            inputs = {k: v.to(self.runtime.device) for k, v in inputs.items()}
            labels = inputs.pop('labels')
            with torch.autocast(device_type=self.runtime.device.type, dtype=torch.bfloat16,
                                enabled=self.config['train']['mixed_precision'] == 'bf16'):
                # Accelerate's training wrapper copies ALL logits to FP32 (9.3 GiB for
                # eight Qwen blocks). Evaluation needs FP32 only for one row's CE.
                forward = getattr(self.model, '_original_forward', self.model.forward)
                output = forward(**inputs, use_cache=False)
            for i, (book_id, _) in enumerate(batch_refs):
                loss = nll_sum(output.logits[i:i+1], labels[i:i+1])
                totals[slots[book_id], 0] += loss.double()
                totals[slots[book_id], 1] += (labels[i, 1:] != -100).sum()
                totals[slots[book_id], 2] += 1
            # Do not hold the previous batch's FP32 logits during the next forward.
            del output, loss
        reduce_sum(totals)
        if not torch.isfinite(totals).all():
            raise ValueError('Nonfinite evaluation NLL')
        return {book_id: metric(float(row[0]), int(row[1]), int(row[2]))
                for book_id, row in zip(book_ids, totals.cpu().tolist())}

    def evaluate(self, state, sequence, step, *, test_only=False):
        train_ids = self.manifest['train_ids']
        current = train_ids[sequence - 1] if not test_only else 'test'
        previous = train_ids[:sequence - 1]
        k = self.config['eval']['retention_k']
        retention = previous if k is None else previous[-k:] if k else []
        groups = {'general': self.manifest['general_ids']} if test_only else {
            'current': [current], 'retention': retention, 'general': self.manifest['general_ids']}
        needed = list(dict.fromkeys(i for ids in groups.values() for i in ids))
        self.cache = {i: self.cache[i] for i in needed if i in self.cache}
        missing = [i for i in needed if i not in self.cache]
        started = time.monotonic()
        with evaluation_context(self.model):
            computed = self.score(missing)
        self.cache.update({i: (m, self.eval_id) for i, m in computed.items()})
        context = {'state': state, 'phase': evaluation_phase(state, step), 'sequence': sequence, 'step': step,
                   'targeted_task': f'pg19:{current}', 'book_id': current, 'eval_id': self.eval_id}
        for scope, ids in groups.items():
            metrics, forgetting = [], []
            for book_id in ids:
                row, origin = self.cache[book_id]
                extra = {}
                if scope == 'retention':
                    reference = self.references[book_id]
                    extra = {'reference_post_merge_bpt': reference['bpt'],
                             'books_since_learning': sequence - reference['sequence'],
                             'forgetting_bpt': row['bpt'] - reference['bpt']}
                    forgetting.append(extra['forgetting_bpt'])
                item = self.manifest['books'][book_id]
                self.logger.write({'event': 'eval_task', **context, 'eval_scope': scope,
                                   'evaluated_task': f'pg19:{book_id}', 'evaluated_book_id': book_id,
                                   'split': self.manifest['general_split'] if scope == 'general' else 'book_holdout',
                                   'eligible_blocks': len(item['offsets']) if scope == 'general' else len(item['held_out_blocks']),
                                   'sampled_block_ids': item['eval_blocks'], 'eval_cached': origin != self.eval_id,
                                   'reused_from_eval_id': origin, **row, **extra})
                metrics.append(row)
            extras = {}
            if scope == 'current':
                if state == 'pre_segment':
                    self.current_before = metrics[0]['bpt']
                else:
                    extras['adaptation_gain_bpt'] = self.current_before - metrics[0]['bpt']
            if scope == 'retention':
                extras['mean_forgetting_bpt'] = math.fsum(forgetting) / len(forgetting) if forgetting else None
            self.logger.write({'event': 'eval_summary', **context, 'eval_scope': scope, **summary(metrics), **extras,
                               'cached_books': sum(self.cache[i][1] != self.eval_id for i in ids),
                               'computed_books': sum(i in computed for i in ids),
                               'boundary_seconds': time.monotonic() - started})
        if state == 'post_merge' and not test_only:
            self.references[current] = {'bpt': self.cache[current][0]['bpt'], 'sequence': sequence}
        self.eval_id += 1


class BookCallback(TrainerCallback):
    def __init__(self, config, manifest, steps, runtime, evaluator, logger, checkpoint=None):
        self.config, self.manifest, self.steps, self.runtime = config, manifest, steps, runtime
        self.evaluator, self.logger, self.checkpoint = evaluator, logger, checkpoint
        self.offset = checkpoint['step'] if checkpoint else 0
        self.completed = checkpoint['sequence'] if checkpoint else 0
        self.trainer = None
        self.before_model = self.before_optimizer = self.before_cache = None

    def on_train_begin(self, args, state, control, **kwargs):
        if self.checkpoint:
            checkpoints.restore(self.checkpoint, self.evaluator.model, self.trainer.optimizer, self.trainer.lr_scheduler,
                                cuda_device=self.runtime.device.index)
            self.evaluator.references = self.checkpoint['extra']['references']
            self.evaluator.cache = self.checkpoint['extra']['eval_cache']
            self.evaluator.eval_id = self.checkpoint['extra']['eval_id']
            self.checkpoint = None

    def on_step_begin(self, args, state, control, **kwargs):
        step = self.offset + state.global_step
        item = self.steps[step]
        if item['first']:
            self.evaluator.evaluate('pre_segment', item['sequence'], step)
            self.before_model = clone_floating_state(self.evaluator.model)
            self.before_optimizer = clone_optimizer_state(self.trainer.optimizer)
            self.before_cache = self.evaluator.cache.copy()
        self.step_started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.runtime.device.type == 'cuda':
            torch.cuda.synchronize(self.runtime.device)
        step = self.offset + state.global_step
        item = self.steps[step - 1]
        if self.trainer.accelerator.optimizer_step_was_skipped:
            raise RuntimeError('Skipped optimizer update')
        stats = self.trainer.last_stats
        loss = stats[0] / stats[1]
        context = {'step': step, 'sequence': item['sequence'], 'book_id': item['book_id'],
                   'targeted_task': f'pg19:{item["book_id"]}', 'local_steps': item['local_step']}
        self.logger.write({'event': 'train_batch', **context, 'loss': loss, 'bpt': loss / math.log(2),
                           'nll_sum': stats[0], 'target_tokens': int(stats[1]), 'num_blocks': len(item['indices']),
                           'learning_rate': self.trainer.optimizer.param_groups[0]['lr'],
                           'optimizer_step': step, 'seconds': time.monotonic() - self.step_started})
        self.evaluator.cache = {}
        if not item['last']:
            return
        self.evaluator.evaluate('pre_merge', item['sequence'], step)
        started = time.monotonic()
        lam = self.config['train']['merge']['base_meta_lr']
        after = clone_floating_state(self.evaluator.model)
        apply_reptile_state(self.evaluator.model, self.before_model, after, lam)
        merge_adamw_optimizer_state(self.trainer.optimizer, self.before_optimizer, lam)
        if lam == 0:
            self.evaluator.cache = self.before_cache
        elif lam != 1:
            self.evaluator.cache = {}
        self.before_model = self.before_optimizer = self.before_cache = None
        del after
        self.logger.write({'event': 'segment_merge', **context, 'meta_lr': lam, 'seconds': time.monotonic() - started})
        self.evaluator.evaluate('post_merge', item['sequence'], step)
        self.completed = item['sequence']
        started = time.monotonic()
        checkpoints.save(self.config, self.evaluator.model, self.trainer.optimizer, step, self.completed,
                         scheduler=self.trainer.lr_scheduler, cuda_device=self.runtime.device.index,
                         extra={'manifest': self.manifest, 'references': self.evaluator.references,
                                'eval_cache': self.evaluator.cache, 'eval_id': self.evaluator.eval_id,
                                'global_batch_size': self.runtime.num_processes * args.per_device_train_batch_size})
        def prune():
            files = sorted((Path(self.config['output_dir']) / 'checkpoints').glob('sequence-*.pt'),
                           key=lambda p: int(p.stem.split('-')[1]))
            for path in files[:-2]:
                path.unlink()
        rank_zero(self.runtime, prune)
        self.logger.write({'event': 'checkpoint_saved', **context, 'seconds': time.monotonic() - started})


class BookTrainer(SFTTrainer):
    def get_train_dataloader(self):
        # Already sharded: do not pass through Accelerate's DataLoader sharding.
        return DataLoader(self.train_dataset, batch_sampler=self.book_sampler,
                          collate_fn=self.data_collator, num_workers=0,
                          generator=torch.Generator().manual_seed(self.args.seed))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        output = model(**inputs, use_cache=False)
        local_nll = nll_sum(output.logits, labels)
        totals = torch.stack((local_nll.detach().double(), (labels[:, 1:] != -100).sum().double()))
        reduce_sum(totals)
        if not torch.isfinite(totals).all() or totals[1] <= 0:
            raise ValueError('Nonfinite loss or empty global batch')
        self.last_stats = totals.cpu().tolist()
        # DDP averages gradients; cancel that average to obtain global token-mean gradients.
        loss = local_nll * self.accelerator.num_processes / totals[1].to(local_nll.dtype)
        return (loss, output) if return_outputs else loss


def load_model_and_tokenizer(config):
    settings = config['model']
    kwargs = {'revision': settings.get('revision'), 'cache_dir': os.environ['HF_HUB_CACHE']}
    tokenizer = AutoTokenizer.from_pretrained(settings['name'], **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(settings['name'], dtype=torch.float32, **kwargs)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.init_kwargs['_commit_hash'] = getattr(model.config, '_commit_hash', None)
    return model, tokenizer


def resume_manifest(checkpoint, tokenizer):
    manifest = checkpoint['extra']['manifest']
    if manifest['tokenizer'] != tokenizer_metadata(tokenizer):
        raise ValueError('Resume tokenizer differs from the checkpoint')
    for book in manifest['books'].values():
        if sha256(book['tokens_path']) != book['tokens_sha256']:
            raise ValueError(f'Resume token cache differs for book {book["book_id"]}')
    return manifest


def run(config, *, overwrite=False, dry_run=False, resume=None, eval_only=False):
    config = validate_config(resolve_run_config(config, resume=resume, eval_only=eval_only))
    if dry_run:
        print(json.dumps({'event': 'configuration_validated', 'config': config}, allow_nan=False))
        return 0
    if eval_only and resume is None:
        raise ValueError('Evaluation-only requires --resume CHECKPOINT')
    settings = config['train']
    backend = 'gloo' if settings['use_cpu'] else 'nccl'
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if not settings['use_cpu'] and not distributed and torch.cuda.device_count() != 1:
        raise ValueError('Launch one process per GPU with torchrun; a single-process reference must expose one GPU')
    # Rank zero can wait on the shared preparation lock while peers wait to receive
    # its manifest. Allow cold-cache preparation to exceed NCCL's ten-minute default.
    runtime = PartialState(cpu=settings['use_cpu'], **(
        {'backend': backend, 'timeout': timedelta(minutes=30)} if distributed else {}))
    torch.backends.cuda.enable_cudnn_sdp(False)
    if not settings['use_cpu']:
        torch.cuda.set_device(runtime.device.index if runtime.device.index is not None else 0)
    output = Path(config['output_dir'])
    rank_zero(runtime, lambda: prepare_output_dir(output, overwrite))
    start_run(copy.deepcopy(config))
    tracking = None
    if runtime.is_main_process and config['wandb']['enabled']:
        import wandb
        tracking = wandb.init(project=os.environ.get('WANDB_PROJECT', config['wandb']['project']),
                              name=config['_tvd']['run_id'], config=config, dir=str(output))
        tracking.define_metric('eval/*', step_metric='book')
        tracking.define_metric('train/*', step_metric='optimizer_step')
        tracking.define_metric('merge/*', step_metric='book')
    train_log, eval_log = Logger(output / 'train.jsonl', runtime, tracking), Logger(output / 'eval.jsonl', runtime, tracking)
    success = False
    try:
        if runtime.is_main_process:
            write_run_record(output, 'resolved_config', config)
        train_log.write({'event': 'run_start', 'timestamp': time.time(), 'num_processes': runtime.num_processes,
                         'cudnn_sdp': torch.backends.cuda.cudnn_sdp_enabled(), 'backend': backend,
                         'parameter_dtype': 'float32', 'optimizer_dtype': 'float32',
                         'compute_dtype': settings['mixed_precision'], 'gradient_checkpointing': settings['gradient_checkpointing'],
                         'fixed_training_padding': True, 'release_eval_memory': True,
                         'eval_bypasses_fp32_logits_wrapper': True, 'cuda_rng_scope': 'local_rank',
                         'distributed_timeout_seconds': 1800 if distributed else None})
        set_seed(config['seed'])
        model, tokenizer = load_model_and_tokenizer(config)
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('PG-19 requires FP32 model parameters')
        checkpoint = checkpoints.read_checkpoint(config)
        if checkpoint:
            if not eval_only and any(config[k] != checkpoint['config'][k] for k in ('data', 'eval')):
                raise ValueError('Resume changes data selection or evaluation settings')
            model.load_state_dict(checkpoint['model'])
        started = time.monotonic()
        manifest = rank_zero(runtime, lambda: resume_manifest(checkpoint, tokenizer) if checkpoint and not eval_only
                             else prepare(config, tokenizer, test_only=eval_only))
        train_log.write({'event': 'data_preparation', 'seconds': time.monotonic() - started,
                         'train_books': len(manifest['train_ids']), 'general_books': len(manifest['general_ids']),
                         'train_blocks': sum(len(x['train_blocks']) for x in manifest['books'].values()),
                         'general_blocks': sum(len(manifest['books'][i]['eval_blocks']) for i in manifest['general_ids']),
                         'discarded_tail_tokens': sum(x['discarded_tail_tokens'] for x in manifest['books'].values()),
                         'cache_dir': config['data']['cache_dir'], 'manifest_fingerprint': manifest['fingerprint']})
        rank_zero(runtime, lambda: write_json(output / 'data_manifest.json', manifest))
        if runtime.is_main_process:
            record_runtime(output_dir=output, model_config=model.config.to_dict(),
                           model_revision=getattr(model.config, '_commit_hash', None),
                           tokenizer=tokenizer_metadata(tokenizer), input_files=[output / 'data_manifest.json'])
        if runtime.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(runtime.device)
        dataset, collator = Blocks(manifest), Collator(tokenizer.pad_token_id)
        evaluator = Evaluator(config, manifest, dataset, collator, runtime, eval_log)
        if eval_only:
            evaluator.model = model.to(runtime.device)
            evaluator.evaluate('final', checkpoint['sequence'], checkpoint['step'], test_only=True)
            completed, step = checkpoint['sequence'], checkpoint['step']
        else:
            global_batch = runtime.num_processes * settings['per_device_train_batch_size']
            if checkpoint and checkpoint['extra']['global_batch_size'] != global_batch:
                raise ValueError('Resume changes global batch size')
            steps = schedule(manifest, global_batch)
            callback = BookCallback(config, manifest, steps, runtime, evaluator, train_log, checkpoint)
            checkpoint = None  # The callback releases CPU checkpoint tensors after restoring the optimizer.
            if callback.offset >= len(steps):
                raise ValueError('Checkpoint already completed the selected books')
            args = SFTConfig(output_dir=str(output), use_cpu=settings['use_cpu'], seed=config['seed'],
                             bf16=settings['mixed_precision'] == 'bf16', fp16=False,
                             per_device_train_batch_size=settings['per_device_train_batch_size'],
                             gradient_accumulation_steps=1, num_train_epochs=1,
                             learning_rate=settings['learning_rate'], lr_scheduler_type='constant',
                             weight_decay=settings['weight_decay'], max_grad_norm=settings['gradient_clip_norm'],
                             gradient_checkpointing=settings['gradient_checkpointing'],
                             gradient_checkpointing_kwargs={'use_reentrant': False},
                             ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
                             ddp_backend=backend if distributed else None,
                             packing=False, padding_free=False, completion_only_loss=False, loss_type='nll',
                             dataset_kwargs={'skip_prepare_dataset': True}, max_length=None,
                             remove_unused_columns=False, average_tokens_across_devices=False,
                             save_strategy='no', eval_strategy='no', logging_strategy='no', report_to='none')
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                          betas=tuple(settings['adam_betas']), eps=settings['adam_epsilon'],
                                          weight_decay=settings['weight_decay'])
            trainer = BookTrainer(model=model, args=args, processing_class=tokenizer, train_dataset=dataset,
                                  data_collator=Collator(tokenizer.pad_token_id, config['data']['block_size']),
                                  optimizers=(optimizer, None), callbacks=[callback])
            trainer.model_accepts_loss_kwargs = False
            trainer.book_sampler = BookBatchSampler(steps, runtime.process_index, runtime.num_processes,
                                                    settings['per_device_train_batch_size'], callback.offset)
            callback.trainer = trainer
            evaluator.model = trainer.accelerator.unwrap_model(trainer.model)
            trainer.train()
            completed, step = callback.completed, callback.offset + trainer.state.global_step
            if completed != len(manifest['train_ids']) or step != len(steps):
                raise RuntimeError('Trainer did not finish the exact book schedule')
        memory = {'rank': runtime.process_index,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated(runtime.device) / 2**30 if runtime.device.type == 'cuda' else 0,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved(runtime.device) / 2**30 if runtime.device.type == 'cuda' else 0}
        ranks = [None] * runtime.num_processes
        if runtime.num_processes > 1:
            torch.distributed.all_gather_object(ranks, memory)
        else:
            ranks[0] = memory
        if runtime.is_main_process:
            details = {'steps': step, 'books': completed, 'output_dir': str(output), 'memory': ranks}
            write_run_record(output, 'run_complete', details)
            if tracking:
                tracking.summary.update(details)
            print(json.dumps(details), flush=True)
        success = True
        return 0
    finally:
        train_log.write({'event': 'run_end', 'status': 'complete' if success else 'failed', 'timestamp': time.time()})
        train_log.close()
        eval_log.close()
        RUN_CONTEXT.clear()
        if tracking:
            tracking.finish(exit_code=0 if success else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config/train.json')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--eval-only', action='store_true', help='Evaluate sampled official test books from a checkpoint')
    args = parser.parse_args(argv)
    try:
        result = run(read_json(args.config), overwrite=args.overwrite, dry_run=args.dry_run,
                     resume=args.resume, eval_only=args.eval_only)
    except BaseException:
        if torch.distributed.is_initialized():
            # Collective cleanup after one rank fails can deadlock and hide its traceback.
            # Exit this worker so torchrun promptly terminates its waiting peers.
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        raise
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return result


if __name__ == '__main__':
    raise SystemExit(main())

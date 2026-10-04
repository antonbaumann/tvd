"""Deterministic book selection, disjoint token blocks, and rank-local batches."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from filelock import FileLock

from shared.io import sha256
from shared.provenance import tokenizer_metadata


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rng(seed, purpose, book_id=''):
    return np.random.default_rng(int(digest([seed, purpose, str(book_id)])[:16], 16))


def sample(values, limit, seed, purpose, book_id):
    values = list(values)
    if limit is None or len(values) <= limit:
        return sorted(values)
    return sorted(rng(seed, purpose, book_id).choice(values, limit, replace=False).tolist())


def block_offsets(length, size):
    return [[start, min(start + size, length)] for start in range(0, length, size)
            if min(size, length - start) >= 2]


def partition_blocks(count, fraction, seed, book_id):
    if count < 2:
        raise ValueError(f'Book {book_id} needs at least two usable blocks')
    held = sample(range(count), math.ceil(count * fraction), seed, 'holdout', book_id)
    held_set = set(held)
    training = [i for i in range(count) if i not in held_set]
    if not training:
        raise ValueError(f'Book {book_id} has no training blocks')
    return training, held


class Source:
    """Published split lists plus original texts; optional local mirror for offline use."""
    def __init__(self, config, cache):
        self.config, self.cache = config, Path(cache)
        self.local = Path(config['local_source']) if config.get('local_source') else None
        self.revision = config.get('revision') or 'main'
        if self.local is None:
            from huggingface_hub import HfApi
            self.revision = HfApi().dataset_info('deepmind/pg19', revision=self.revision).sha

    def download(self, relative):
        if self.local is not None:
            return self.local / relative
        path = self.cache / 'raw' / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            with urllib.request.urlopen('https://storage.googleapis.com/deepmind-gutenberg/' + relative,
                                        timeout=60) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            os.replace(temporary, path)
        return path

    def names(self, split):
        if self.local is not None:
            path = self.local / 'data' / f'{split}_files.txt'
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download('deepmind/pg19', f'data/{split}_files.txt', repo_type='dataset',
                                       revision=self.revision, cache_dir=os.environ['HF_HUB_CACHE']))
        names = sorted(path.read_text().splitlines())
        if len(names) != len(set(names)) or any(not re.fullmatch(rf'{split}/[0-9]+\.txt', x) for x in names):
            raise ValueError(f'Invalid PG-19 {split} manifest')
        return names, sha256(path)

    def metadata(self):
        path = self.download('metadata.csv')
        with path.open() as handle:
            rows = {row[0]: {'title': row[1], 'publication_date': row[2]} for row in csv.reader(handle)}
        return rows, sha256(path)


def prepare(config, tokenizer, *, test_only=False):
    """Called on rank zero; token files are reusable across data seeds and run settings."""
    cache = Path(config['data']['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    # Different matched configurations may start together on different Slurm nodes.
    with FileLock(cache / 'prepare.lock'):
        return _prepare(config, tokenizer, test_only=test_only)


def _prepare(config, tokenizer, *, test_only=False):
    settings, evaluation = config['data'], config['eval']
    cache = Path(settings['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    source = Source(settings, cache)
    metadata, metadata_hash = source.metadata()
    tok_meta = tokenizer_metadata(tokenizer)
    tok_key = digest(tok_meta)
    result = {'version': 1, 'dataset': 'PG-19', 'split_manifest_revision': source.revision,
              'metadata_sha256': metadata_hash, 'tokenizer': tok_meta, 'block_size': settings['block_size'],
              'train_data_seed': settings['train_data_seed'], 'eval_data_seed': settings['eval_data_seed'],
              'train_ids': [], 'general_ids': [], 'books': {}, 'excluded': [], 'split_hashes': {}}

    def book(relative):
        book_id = Path(relative).stem
        raw = source.download(relative)
        raw_hash = sha256(raw)
        token_path = cache / 'tokens' / tok_key / f'{book_id}-{raw_hash}.npy'
        if not token_path.exists():
            tokens = tokenizer(raw.read_text(encoding='utf-8'), add_special_tokens=False,
                               return_attention_mask=False, truncation=False, verbose=False)['input_ids']
            token_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = token_path.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, np.asarray(tokens, dtype=np.int32), allow_pickle=False)
            os.replace(temporary, token_path)
        length = len(np.load(token_path, mmap_mode='r'))
        offsets = block_offsets(length, settings['block_size'])
        return {'book_id': book_id, **metadata.get(book_id, {}), 'source': relative,
                'source_sha256': raw_hash, 'tokens_path': str(token_path), 'tokens_sha256': sha256(token_path),
                'num_tokens': length, 'discarded_tail_tokens': length - sum(b-a for a,b in offsets),
                'offsets': offsets, 'train_blocks': [], 'held_out_blocks': [], 'eval_blocks': []}

    if not test_only:
        names, result['split_hashes']['train'] = source.names('train')
        candidates = rng(settings['train_data_seed'], 'book_order').permutation(names).tolist()
        for relative in candidates:
            item = book(relative)
            book_id = item['book_id']
            if len(item['offsets']) < 2:
                result['excluded'].append({'book_id': book_id, 'reason': 'fewer_than_two_blocks'})
                continue
            training, held = partition_blocks(len(item['offsets']), settings['holdout_fraction'],
                                              settings['train_data_seed'], book_id)
            item.update(train_blocks=training, held_out_blocks=held,
                        eval_blocks=sample(held, evaluation['book_blocks'], settings['eval_data_seed'],
                                           'book_evaluation', book_id))
            result['books'][book_id] = item
            result['train_ids'].append(book_id)
            if len(result['train_ids']) == settings['num_books']:
                break
        if len(result['train_ids']) != settings['num_books']:
            raise ValueError('Not enough eligible PG-19 training books')

    split = 'test' if test_only else 'validation'
    names, result['split_hashes'][split] = source.names(split)
    expected = 100 if test_only else 50
    if len(names) != expected:
        raise ValueError(f'PG-19 {split} must contain exactly {expected} books')
    for relative in names:
        item = book(relative)
        book_id = item['book_id']
        if book_id in result['books']:
            raise ValueError(f'Book {book_id} appears in training and {split}')
        limit = evaluation['test_blocks'] if test_only else 2
        if len(item['offsets']) < (limit or 1):
            raise ValueError(f'Book {book_id} in {split} has fewer than {limit or 1} usable blocks')
        item['eval_blocks'] = sample(range(len(item['offsets'])), limit, settings['eval_data_seed'],
                                     split + '_evaluation', book_id)
        result['general_ids'].append(book_id)
        result['books'][book_id] = item
    result['general_split'] = split
    result['fingerprint'] = digest(result)
    return result


class Blocks(Dataset):
    def __init__(self, manifest):
        self.manifest = manifest
        self.refs = [(book_id, block) for book_id in manifest['train_ids']
                     for block in manifest['books'][book_id]['train_blocks']]
        self.arrays = {}

    def block(self, book_id, block):
        item = self.manifest['books'][book_id]
        if book_id not in self.arrays:
            self.arrays[book_id] = np.load(item['tokens_path'], mmap_mode='r')
        start, stop = item['offsets'][block]
        return {'input_ids': self.arrays[book_id][start:stop].tolist(), 'dummy': False}

    def __getitem__(self, index):
        if index == -1:
            result = self.block(*self.refs[0])
            result['dummy'] = True
            return result
        return self.block(*self.refs[index])

    def __len__(self):
        return len(self.refs)


def schedule(manifest, global_batch):
    steps, offset = [], 0
    for sequence, book_id in enumerate(manifest['train_ids'], 1):
        count = len(manifest['books'][book_id]['train_blocks'])
        for local, start in enumerate(range(0, count, global_batch)):
            indices = list(range(offset + start, offset + min(start + global_batch, count)))
            steps.append({'book_id': book_id, 'sequence': sequence, 'indices': indices,
                          'first': start == 0, 'last': start + global_batch >= count,
                          'local_step': local + 1})
        offset += count
    return steps


class BookBatchSampler(Sampler):
    def __init__(self, steps, rank, world, per_device, start_step=0):
        self.steps, self.rank, self.world, self.per_device = steps[start_step:], rank, world, per_device

    def __iter__(self):
        for step in self.steps:
            indices = step['indices'] + [-1] * (self.world * self.per_device - len(step['indices']))
            start = self.rank * self.per_device
            yield indices[start:start + self.per_device]

    def __len__(self):
        return len(self.steps)


class Collator:
    def __init__(self, pad_token_id, block_size=None):
        self.pad_token_id, self.block_size = pad_token_id, block_size

    def __call__(self, rows):
        length = self.block_size or max(len(row['input_ids']) for row in rows)
        if any(len(row['input_ids']) > length for row in rows):
            raise ValueError('Input exceeds the configured block size')
        ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, row in enumerate(rows):
            n = len(row['input_ids'])
            ids[i, :n] = torch.tensor(row['input_ids'])
            attention[i, :n] = 1
            if not row.get('dummy', False):
                labels[i, :n] = ids[i, :n]
            labels[i, 0] = -100
        return {'input_ids': ids, 'attention_mask': attention, 'labels': labels}

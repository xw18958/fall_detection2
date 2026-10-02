#!/usr/bin/env python3
"""Run a predeclared seed, lock validation selection, then evaluate it."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from training_v2 import write_json
from training_v4 import file_hash


def markdown_table(frame):
    columns = list(frame.columns)
    def display(value):
        if pd.isna(value):
            return 'N/A'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '\\|')
    lines = ['| '+' | '.join(map(str, columns))+' |', '| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(display(value) for value in row)+' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True, help='Canonical server repository containing data and original runs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--m5-root', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--resume-completed-training', action='store_true',
                        help='Evaluate already locked, completed training runs without retraining')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    preflight = json.loads(args.preflight.read_text())
    if preflight.get('status') != 'passed':
        raise ValueError('Successful tests and end-to-end smoke run are mandatory')
    for name, digest in preflight['code_sha256'].items():
        if file_hash(here/name) != digest:
            raise ValueError('Code changed after the smoke test: '+name)
    root, out = args.repository.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'experiment_status.json').exists() and not args.resume_completed_training:
        raise FileExistsError('This experiment was already launched; choose a new output directory')
    seeds = args.seeds
    if len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique')
    previous = None
    if args.resume_completed_training:
        previous = json.loads((out/'experiment_status.json').read_text())
        for seed in seeds:
            directory = out/f'seed{seed}'
            progress = json.loads((directory/'progress.json').read_text())
            if (seed not in previous['completed_training_seeds'] or
                    progress.get('status') != 'trained' or progress.get('test_evaluated') or
                    not (directory/'checkpoints/locked_model.pt').is_file()):
                raise ValueError(f'Seed {seed} requires completed, locked training and unopened tests')
            command = json.loads((directory/'train_command.json').read_text())
            if int(command[command.index('--seed')+1]) != seed:
                raise ValueError('Recorded training seed does not match requested seed')
    original = root/'FallDetection30HzTraining/runs/v2_multisource_seed42_20261001'
    common = [sys.executable, '-u', str(here/'train.py'), '--experiment-v4',
              '--zip', str(root/'fd_datasets/30Hz_processed_clean_v2'),
              '--public-root', str(root/'fd_datasets/processed_v2'),
              '--m5-root', str(args.m5_root),
              '--legacy-m5-root', str(root/'fd_datasets/M5_hard_negatives_20261002'),
              '--split-from', str(original/'splits/group_splits.json'),
              '--init-ssl', str(original/'checkpoints/ssl_pretrained.pt'),
              '--normalization-from', str(original/'checkpoints/locked_model.pt'),
              '--baseline-checkpoint', str(original/'checkpoints/locked_model.pt'),
              '--cache-root', str(out/'signal_cache'), '--ssl-epochs', '30',
              '--head-epochs', '3', '--all-epochs', '50', '--batch', '128',
              '--ssl-min-epochs', '10', '--ssl-patience', '8', '--min-full-epochs', '10',
              '--patience', '10', '--stride-sec', '.25', '--device', 'cuda:0']
    started = time.monotonic()
    status = {'status': 'training', 'pid': os.getpid(), 'seeds': seeds,
              'completed_training_seeds': [], 'completed_test_seeds': [],
              'preflight': str(args.preflight), 'test_selection_policy': 'primary seed chosen on validation before opening any production test'}
    if previous is not None:
        status['completed_training_seeds'] = list(seeds)
        status['cancelled_seeds'] = previous.get('cancelled_seeds', [])
        status['resumed_completed_training'] = True
        status['test_selection_policy'] = 'predeclared single seed; checkpoint and threshold chosen on validation before testing'
    write_json(out/'experiment_status.json', status)
    def execute(seed, mode):
        directory = out/f'seed{seed}'; directory.mkdir(exist_ok=True)
        command = common+['--seed', str(seed), '--work', str(directory), '--mode', mode]
        write_json(directory/f'{mode}_command.json', command)
        with (directory/f'{mode}.log').open('w') as log:
            subprocess.run(command, cwd=here, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for seed in seeds:
            if seed in status['completed_training_seeds']:
                continue
            status.update(active_seed=seed, active_stage='train')
            write_json(out/'experiment_status.json', status)
            print('START TRAIN', seed, flush=True); execute(seed, 'train')
            status['completed_training_seeds'].append(seed)
            write_json(out/'experiment_status.json', status)
        def rank(seed):
            selection = json.loads((out/f'seed{seed}/results/validation_metrics.json').read_text())
            chosen = selection['selected']
            return (selection['acceptable_improvement'], chosen['valid_under_constraints'], -chosen['weighted_error'])
        primary = max(seeds, key=rank)
        for seed in seeds:
            directory = out/f'seed{seed}'
            write_json(directory/'training_completion.json', json.loads((directory/'progress.json').read_text()))
        write_json(out/'locked_primary_seed.json', {'seed': primary, 'policy': 'validation only',
                   'ranks': {str(seed): list(rank(seed)) for seed in seeds}, 'test_opened': False})
        status.update(status='testing', primary_seed=primary)
        for seed in seeds:
            status.update(active_seed=seed, active_stage='test')
            write_json(out/'experiment_status.json', status)
            print('START TEST', seed, flush=True); execute(seed, 'test')
            status['completed_test_seeds'].append(seed)
        tables = []
        for seed in seeds:
            table = pd.read_csv(out/f'seed{seed}/results/test_summary.csv')
            table.insert(0, 'seed', seed); table.insert(1, 'primary', seed == primary); tables.append(table)
        pd.concat(tables, ignore_index=True).to_csv(out/'all_seed_test_summary.csv', index=False)
        columns = ['accuracy','precision','recall','specificity','f1','f2','mcc','ap','negative_window_fpr']
        if len(seeds) > 1:
            summary = pd.concat(tables).groupby('dataset')[columns].agg(['mean','std'])
            summary.to_csv(out/'seed_mean_std.csv')
        # Fair historical baseline: same restored partitions and dense grid,
        # its own legacy M5 conversion and validation-calibrated threshold.
        import torch
        import train
        import training_v2 as v2
        from training_v4 import inherited_splits, test_report
        train.seed_all(42); torch.set_num_threads(4)
        baseline = torch.load(original/'checkpoints/locked_model.pt', map_location='cpu', weights_only=False)
        cfg = train.Cfg(**baseline['config']); cfg.stride_sec = .25
        records = v2.own_records(root/'fd_datasets/30Hz_processed_clean_v2')
        records += v2.m5_records(root/'fd_datasets/M5_hard_negatives_20261002')
        records += v2.public_records(root/'fd_datasets/processed_v2')
        baseline_out = out/'baseline'; baseline_out.mkdir(exist_ok=True)
        splits, _ = inherited_splits(records, original/'splits/group_splits.json', baseline_out/'group_splits.json')
        threshold = json.loads((out/f'seed{primary}/baseline_validation.json').read_text())['joint_threshold']['threshold']
        model = train.Net(cfg).to('cuda:0'); model.load_state_dict(baseline['model'])
        arr = v2.arrays(splits['test'], out/'signal_cache')
        test_report(model, splits['test'], arr, baseline['normalization'], cfg, torch.device('cuda:0'),
                    train, threshold, baseline_out, splits['train'], out/'signal_cache', False)
        primary_table = pd.read_csv(out/f'seed{primary}/results/test_summary.csv')
        baseline_table = pd.read_csv(baseline_out/'test_summary.csv')
        comparison = baseline_table.merge(primary_table, on='dataset', suffixes=('_baseline','_new'))
        comparison.to_csv(out/'baseline_vs_primary.csv', index=False)
        report_columns = ['dataset','n','tp','fn','fp','tn','accuracy','precision','recall','specificity','f1','negative_window_fpr']
        comparison_columns = ['dataset','accuracy_baseline','accuracy_new','recall_baseline','recall_new',
                              'fp_baseline','fp_new','negative_window_fpr_baseline','negative_window_fpr_new']
        report = ['# Seven-source V2 retraining results', '', f'Primary seed: {primary}; checkpoint and threshold selected using validation before production tests.',
                  '', '## Baseline and primary model', '', markdown_table(comparison[comparison_columns]), '',
                  '## Test results', '', markdown_table(primary_table[report_columns]), '',
                  '## Interpretation', '',
                  'Private V2 retains its original exported representation; M5 uses full-range documented SI conversion and its own training-only normalization.',
                  'Public participants retain the original six-source checkpoint partitions. All eligible training recordings are visited every epoch.',
                  'M5 contains negatives only. Its results measure false alarms, not device-specific fall recall. The test contains one M5 boot.',
                  'All seven sources were trained; zero-shot results are unavailable. These are repeated benchmarks rather than newly collected blind deployment trials.',
                  'The acceptable-improvement flag refers to validation constraints. Test false alarms and missed falls must still be examined separately.']
        test_metrics = json.loads((out/f'seed{primary}/results/final_test_metrics.json').read_text())
        retrieval = test_metrics['own_class_retrieval']
        report.extend(['', '## Retrieval', '', f'Private binary-class retrieval: P@1 {retrieval["precision_at_1"]:.4f}; hit@5 {retrieval["hit_rate_at_5"]:.4f}; mAP {retrieval["mAP"]:.6f}.',
                       '', '## M5 offline alarms', '', markdown_table(pd.DataFrame(test_metrics['m5_boots'])),
                       '', '## Checkpoints', ''])
        report.extend(f'- `{out}/seed{seed}/checkpoints/locked_model.pt`' for seed in seeds)
        for seed in seeds:
            selection = json.loads((out/f'seed{seed}/results/validation_metrics.json').read_text())
            report.append(f'- Seed {seed}: selected {selection["selected"]["stage"]} epoch {selection["selected"]["epoch"]}; acceptable validation improvement: {selection["acceptable_improvement"]}.')
        (out/'REPORT.md').write_text('\n'.join(report)+'\n')
        status.update(status='complete', elapsed_seconds=time.monotonic()-started,
                      results=str(out/'REPORT.md'))
        write_json(out/'experiment_status.json', status)
        print('EXPERIMENTS COMPLETE', json.dumps(status), flush=True)
    except Exception as error:
        status.update(status='failed', error=str(error), elapsed_seconds=time.monotonic()-started)
        write_json(out/'experiment_status.json', status)
        raise


if __name__ == '__main__':
    main()

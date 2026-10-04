"""Review a predeclared replay-tolerance margin for the validation-locked winner.

Does not retrain, switch winners, change labels/splits, or qualify hardware.
Previously examined test data remains an exploratory audit.
"""
import argparse, json
from pathlib import Path
import numpy as np
from trigger_data_v5 import load, evaluation, digest
from power_events_v5 import simulate, quantized_threshold_margin

def scores(path, scale):
    raw = np.fromfile(path, np.int8).reshape(-1, 2).astype(np.float64)
    difference = (raw[:, 1] - raw[:, 0]) * scale
    return np.exp(-np.logaddexp(0., -difference))

def main():
    p = argparse.ArgumentParser()
    for name in ('own-root', 'public-root', 'm5-root', 'checkpoint', 'split-from', 'experiment', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    lock = json.loads((a.experiment / 'selection_lock.json').read_text())
    winner = lock['winner']
    if not winner: raise ValueError('A validation-locked winner is required')
    candidate = lock['candidates'][winner]
    scale = candidate['metadata']['output_scale']
    threshold = quantized_threshold_margin(candidate['selection']['threshold'], scale, 1)
    splits, arrays, ck, cfg, audit = load(a.own_root, a.public_root, a.m5_root,
                                        a.checkpoint, a.split_from, a.experiment / 'cache')
    if audit['checkpoint_sha256'] != lock['checkpoint_sha256'] or audit['split_sha256'] != lock['split_sha256']:
        raise ValueError('Frozen experiment contract differs')
    # The winner and tolerance budget are fixed before examining test scores.
    report = dict(winner=winner, original_threshold=candidate['selection']['threshold'],
        adjusted_threshold=threshold, output_lsb_tolerance=1,
        selection_lock_sha256=digest(a.experiment / 'selection_lock.json'),
        policy='Predeclared +/-1 LSB per logit; lower logit difference by 2 output scales',
        test_used_for_selection=False, previous_test_seen=True,
        deployment_eligible=False, commercial_qualified=False, hardware_deadline_qualified=False)
    for role, prefix in (('val', 'validation'), ('test', 'test')):
        _, rows = evaluation(splits, arrays, ck['normalization'], cfg, role)
        trigger = scores(a.experiment / winner / (prefix + '_micro') / 'output.bin', scale)
        # Full model output metadata can be recovered from the frozen model.
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        from train_power_trigger import lite_metadata
        fullmeta = lite_metadata(tf, a.experiment.parent / 'collector/firmware/main/model.tflite')
        full = scores(a.experiment / (prefix + '_full') / 'output.bin', fullmeta['output_scale'])
        result, _ = simulate(rows, trigger, full, threshold, lock['full_threshold'])
        recalls = [v for v in result['per_source_trigger_recording_recall'].values() if v is not None]
        result['data_gate_passed'] = bool(result['lost_or_late_reference_events'] == 0
            and recalls and min(recalls) >= .995 and result['negative_active_fraction'] < 1/3)
        report[role] = result
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({r: {k: report[r][k] for k in ('data_gate_passed', 'lost_or_late_reference_events',
        'negative_active_fraction', 'false_wakes_per_hour')} for r in ('val', 'test')}, indent=2))

if __name__ == '__main__': main()

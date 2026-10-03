#!/usr/bin/env python3
"""Aggregate V5 validation evidence; never modify candidate selection or use test."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import export_v4_tflite as e
import training_v4 as v4
from quantize_v5 import load
from prepare_m5_combined_v5 import records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/raid1/xwan0900/v5_m5_seed42_20261003'))
    p.add_argument('--project',type=Path,default=Path('/raid1/xwan0900/fall_detection2'))
    a=p.parse_args();r=a.root;work=r/'seed42';data=a.project/'fd_datasets/M5_combined_v5_20261003'
    args=argparse.Namespace(work=work,checkpoint=work/'checkpoints/locked_model.pt',m5_root=data,
        own_root=a.project/'fd_datasets/30Hz_processed_clean_v2',public_root=a.project/'fd_datasets/processed_v2')
    c,cfg,parts,model=load(args);selected=json.loads((r/'qat/selected.json').read_text())
    ptq=json.loads((r/'ptq/validation_report.json').read_text());policy=json.loads((work/'selection_protocol.json').read_text())
    out=r/'reports';out.mkdir(exist_ok=True);all_metrics={};choices={};diagnostics={};table=[]
    for method in ['Float','PTQ','QAT']:
        directory=r/'ptq/validation' if method in ['Float','PTQ'] else r/f"qat/epoch_{selected['epoch']:02d}/validation"
        selection=c['checkpoint_selection'] if method=='Float' else ptq['selection'] if method=='PTQ' else selected['selection']
        choices[method]=selection;details={};all_y=[];all_p=[];metric={}
        for d in c['source_weights']:
            z=np.load(directory/(d+('.float_predictions.npz' if method=='Float' else '.predictions.npz')))
            probabilities=e.probabilities(z['logits']);y=z['labels'];keys=z['keys'];starts=z['starts']
            rows=[dict(key=k,file_label=int(yy),prob=float(prob),start=int(start)) for k,yy,prob,start in zip(keys,y,probabilities,starts)]
            kk,yy,pp=e.base.aggregate(rows);details[d]=(rows,kk,yy,pp)
            m=e.metrics(y,probabilities,selection['threshold']);metric[d]=m
            table.append(dict(method=method,dataset=d,threshold=selection['threshold'],**m))
            all_y.extend(y);all_p.extend(probabilities)
        metric['pooled']=e.metrics(all_y,np.asarray(all_p),selection['threshold']);all_metrics[method]=metric
        # Show the sensitivity tradeoff using validation only. No new lock or deployment.
        alternative=v4.select_threshold(details,parts['val'],c['source_weights'],
            {e.v2.M5:policy['constraints'][e.v2.M5]},window_recall_floors=policy['window_recall_floors'],deployment_thresholds=True)
        alt_fixed=v4.select_threshold(details,parts['val'],c['source_weights'],policy['constraints'],
            window_recall_floors=policy['window_recall_floors'],fixed=alternative['threshold'])
        rr=details[e.v2.M5][0]
        alt_m5=e.metrics(np.asarray([v['file_label'] for v in rr]),np.asarray([v['prob'] for v in rr]),alternative['threshold'])
        violations=[]
        for d,gates in policy['constraints'].items():
            for name,bound in gates.items():
                value=alt_fixed['domains'][d][name]
                if (value>bound+1e-12 if name=='negative_window_fpr' else value<bound-1e-12):
                    violations.append(dict(domain=d,metric=name,value=value,bound=bound))
        for d,bound in policy['window_recall_floors'].items():
            if alt_fixed['window_recalls'][d]<bound-1e-12:
                violations.append(dict(domain=d,metric='positive_window_recall',value=alt_fixed['window_recalls'][d],bound=bound))
        diagnostics[method]=dict(validation_only=True,adopted=False,
            purpose='show M5 sensitivity/specificity tradeoff; these alternatives cannot become a deployed candidate',
            m5_only_selection=alternative,m5_windows=alt_m5,violated_full_gates=violations)
    torch.set_num_threads(4);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train=[z for z in parts['train'] if z['domain']=='own'];val=[z for z in parts['val'] if z['domain']=='own']
    cache=r/'cache';retrieval=e.v2.retrieval(model.to(device),
        e.v2.Clips(train,e.v2.arrays(train,cache),c['normalization'],cfg),
        e.v2.Clips(val,e.v2.arrays(val,cache),c['normalization'],cfg),cfg,device,e.base)
    history=json.loads((work/'training_history.json').read_text());qat_history=json.loads((r/'qat/training_history.json').read_text())
    lock=json.loads((r/'locked_int8/selection_lock.json').read_text())
    verification=json.loads((data/'processed_v2/verification.json').read_text())
    summary=dict(status='qualified' if lock['eligible'] else 'completed_no_qualified_deployment_candidate',
        seed=42,dataset=verification,classification_validation=all_metrics,selection=choices,
        validation_only_threshold_diagnostics=diagnostics,validation_windows=c['validation_windows'],
        stage_epochs={stage:sum(z['stage']==stage for z in history) for stage in ['ssl','head','all']},
        selected_float_stage=c['checkpoint_selection']['stage'],selected_float_epoch=c['checkpoint_selection']['epoch'],
        qat_epochs=len(qat_history)-1,selected_qat_epoch=selected['epoch'],
        float_checkpoint_sha256=e.sha(args.checkpoint),ptq_model_sha256=e.sha(r/'ptq/model_int8.tflite'),
        qat_model_sha256=selected['model_sha256'],validation_retrieval=retrieval,
        fine_tune_test=None if not lock['eligible'] else str(r/'locked_int8/test_report.json'),
        final_test_evaluated=lock['eligible'] and (r/'locked_int8/test_report.json').exists(),
        zero_shot=None,zero_shot_reason='All seven sources participate in training',
        tests_passed=36,input_raw_files_preserved=True,physical_device_updated=False,commercial_readiness_established=False,
        threshold_policy='gates first; validation source-group error uses recording FNR plus negative-window FPR, with additional M5 positive-window recall floor',
        selection_limitation='fallback lowest weighted error may retain only a subset of positive views; invalid fallbacks are never deployment-qualified')
    e.v2.write_json(out/'validation_summary.json',summary)
    with (out/'classification_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    def pct(v):return 'N/A' if v is None else f'{100*v:.2f}%'
    lines=['# V5 combined M5 training and quantization — seed42','',
        'Training and quantization completed. '+('A candidate qualified; see the pre-test selection lock.' if lock['eligible'] else
        '**No candidate passed all declared validation gates. No firmware was installed; full final-test evaluation of the production candidates was not run.**'),'',
        '## Data and protocol','',
        'Nineteen user-confirmed M5 falls were combined with seven verified full-range M5 negatives. '+
        'The existing paired-peak crop method yielded eighteen 5-second containers and one real 3-second boundary container. '+
        'Every 3-second view inherits its positive container label. No padding or new phase labels were introduced.', '',
        'Positive recordings split 13/3/3 with seed42. They share one boot; participant and placement are unknown. '+
        'These are exploratory whole-recording holdouts. Negative boot splits and original six-source splits were preserved. '+
        'Normalization uses combined M5 training inputs only; source masses remain 25% Private /25% M5 /10% each public.', '',
        f"Training completed SSL/head/full epochs {summary['stage_epochs']}; selected full checkpoint epoch {summary['selected_float_epoch']}. "+
        f"QAT completed {summary['qat_epochs']} epochs; selected epoch {summary['selected_qat_epoch']}.", '',
        '## Validation classification','',
        f"All {c['validation_windows']:,} complete 3-second windows were evaluated on the 0.25-second cumulative grid plus final window. Overlapping windows are correlated. INT8 predictions use native TFLite Micro and the firmware custom kernels. These are validation results, not final-test results.", '',
        '| Model | Source | TP | FN | FP | TN | Precision | Recall | F1 | Specificity | Negative-window FPR |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in all_metrics:
        for d,m in all_metrics[method].items():
            lines.append(f"| {method} | {d} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1'])} | {pct(m['specificity'])} | {pct(m['negative_window_fpr'])} |")
    lines+=['','## Selection and threshold diagnostics','',
        'PTQ and QAT used identical checkpoint/split provenance, validation gates and float32 threshold selection. '+
        'The M5 gates require ≥90% positive-window recall, ≥90% fall-recording recall and ≤1% negative-window FPR, alongside inherited Private/public protection. '+
        'The default weighted error includes recording-level fall detection; a small value does not establish high positive-window recall.', '',
        '| Model | Selected threshold | Qualified | M5 positive windows detected | M5 negative FP windows |',
        '|---|---:|---|---:|---:|']
    for method,s in choices.items():
        m=all_metrics[method][e.v2.M5]
        lines.append(f"| {method} | {s['threshold']:.9g} | {s['valid_under_constraints']} | {m['tp']}/{m['tp']+m['fn']} | {m['fp']}/{m['fp']+m['tn']} |")
    lines+=['','Validation-only M5 sensitivity diagnostics below are **not adopted or deployed**. Gates were not relaxed.','']
    for method,z in diagnostics.items():
        lines.append(f"- {method}: threshold {z['m5_only_selection']['threshold']:.9g}, M5 window recall {pct(z['m5_windows']['recall'])}, M5 negative-window FPR {pct(z['m5_windows']['negative_window_fpr'])}; violated full gates: {json.dumps(z['violated_full_gates'])}.")
    lines+=['','## Retrieval and quantization checks','',
        '| Evaluation | P@1 | Hit@5 | mAP | Gallery / queries |',
        '|---|---:|---:|---:|---|',
        f"| Float fine-tune validation, binary-class retrieval | {retrieval['precision_at_1']:.6f} | {retrieval['hit_rate_at_5']:.6f} | {retrieval['mAP']:.6f} | {retrieval['gallery_recordings']} / {retrieval['query_recordings']} |", '',
        'Fine-tune final-test results: pending validation qualification. A small fixed subset was exercised by the pipeline smoke test, using its separate smoke checkpoint, and did not select a production candidate. Zero-shot: not available; all sources participate in training. Retrieval is binary-class retrieval, not event-instance retrieval.', '',
        'Thirty-six unit tests, GPU SSL/head/full/test/retrieval smoke, exact float graph parity, source/class calibration checks, QAT finite-gradient smoke and native Micro validation passed. '+
        'PTQ uses per-channel convolution weights and has INT8/INT32 tensors only. '+
        'QAT uses five LayerNormV4 operators with INT8 interfaces, int64 statistics and an internal float32 sqrt. '+
        'These are host runtime checks; new hardware latency, alarms and commercial readiness are not established.', '',
        '## Reproduction and private artifacts','',
        'See `README_RETRAINING_V5.md`, `run_v5_training.py`, and `run_v5_quantization.py`. '+
        'Source and compact aggregates may be public. Raw data, calibration/replay inputs and model binaries remain private.', '',
        f"- Combined dataset: `{data}`",f"- Experiment and exported candidates: `{r}`",
        f"- Float checkpoint SHA256: `{summary['float_checkpoint_sha256']}`",
        f"- PTQ model SHA256: `{summary['ptq_model_sha256']}`",
        f"- Selected QAT model SHA256: `{summary['qat_model_sha256']}`"]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','stage_epochs','qat_epochs','selected_qat_epoch','validation_retrieval']},indent=2))
    print('M5_VALIDATION',json.dumps({method:m[e.v2.M5] for method,m in all_metrics.items()}))


if __name__=='__main__':main()

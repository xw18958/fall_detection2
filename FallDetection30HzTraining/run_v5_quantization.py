#!/usr/bin/env python3
"""Run PTQ and the established fused QAT recipe, lock on validation, then test."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/raid1/xwan0900/fall_detection2'))
    p.add_argument('--work-root',type=Path,default=Path('/raid1/xwan0900/v5_m5_seed42_20261003'))
    p.add_argument('--runner',type=Path,default=Path('/raid1/xwan0900/fall_detection2_v4_qat/M5BLECollector/artifacts/native-linux/replay'))
    p.add_argument('--start',choices=['export','validation','qat_smoke','qat_train','lock','test'],default='export')
    a=p.parse_args();root=a.project_root;work=a.work_root/'seed42'
    if not (work/'training_lock.json').is_file():raise ValueError('Training is not locked/completed')
    if not a.runner.is_file():raise FileNotFoundError('Build the audited native Micro runner first')
    ptq=a.work_root/'ptq';qat=a.work_root/'qat';package=a.work_root/'locked_int8'
    common=['--checkpoint',str(work/'checkpoints/locked_model.pt'),'--work',str(work),
         '--own-root',str(root/'fd_datasets/30Hz_processed_clean_v2'),
         '--public-root',str(root/'fd_datasets/processed_v2'),
         '--m5-root',str(root/'fd_datasets/M5_combined_v5_20261003'),'--runner',str(a.runner),
         '--ptq-export',str(ptq)]
    convert=[sys.executable,str(Path(__file__).with_name('quantize_v5.py')),*common,
        '--qat-run',str(qat),'--comparison-checkpoint',str(root/'FallDetection30HzTraining/runs/v4_weighted_20261003/seed42/checkpoints/locked_model.pt')]
    qatcmd=[sys.executable,str(Path(__file__).with_name('qat_v4.py')),*common,'--output',str(qat),
        '--protocol-v5','--fused-norm','--input-percentile','99.9','--distillation-weight','2',
        '--epochs','12','--patience','4','--learning-rate','1e-5','--batch','128']
    jobs=[('export',[*convert,'--phase','export','--output',str(ptq)]),
          ('validation',[*convert,'--phase','validation','--output',str(ptq)]),
          ('qat_smoke',[*qatcmd,'--phase','smoke']),('qat_train',[*qatcmd,'--phase','train']),
          ('lock',[*convert,'--phase','lock','--output',str(package)]),
          ('test',[*convert,'--phase','test','--output',str(package)])]
    a.work_root.mkdir(exist_ok=True);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
    env['OMP_NUM_THREADS']='4';env['TF_CPP_MIN_LOG_LEVEL']='2';env['TF_ENABLE_ONEDNN_OPTS']='0'
    start=time.monotonic();index=[name for name,_ in jobs].index(a.start)
    for name,cmd in jobs[index:]:
        status=dict(status='running',stage=name,elapsed_seconds=time.monotonic()-start,command=cmd)
        (a.work_root/'quantization_progress.json').write_text(json.dumps(status,indent=2)+'\n')
        print('V5_QUANTIZATION_STAGE',name,flush=True)
        with (a.work_root/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            if name=='lock' and (package/'selection_lock.json').exists() and not json.loads((package/'selection_lock.json').read_text())['eligible']:
                status.update(status='completed_no_qualified_deployment_candidate',test_evaluated=False,
                              quality_rejection='No candidate met the unchanged validation gates',log=str(a.work_root/(name+'.log')))
                (a.work_root/'quantization_progress.json').write_text(json.dumps(status,indent=2)+'\n')
                print('V5_VALIDATION_REJECTED_NO_DEPLOYMENT',flush=True)
                return
            status.update(status='failed',exit_code=result.returncode,log=str(a.work_root/(name+'.log')))
            (a.work_root/'quantization_progress.json').write_text(json.dumps(status,indent=2)+'\n')
            raise RuntimeError('Stage failed: '+name)
    (a.work_root/'quantization_progress.json').write_text(json.dumps(dict(status='complete',elapsed_seconds=time.monotonic()-start,
        result=str(package/'test_report.json')),indent=2)+'\n')
    print('V5_QUANTIZATION_COMPLETE',flush=True)


if __name__=='__main__':main()

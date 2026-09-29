import os, subprocess, shutil, json, time
from pathlib import Path

W = Path('/kaggle/working')
BASE = W/'repo_base'
REL = Path('FallDetection30HzTraining/experiments/exp2_converged')

def run(cmd, **kw):
    print('RUN', ' '.join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)

run(['git','clone','--depth','1','https://github.com/xw18958/fall_detection2.git',str(BASE)])

def patch_repo(tag, ssl_lr=None, all_lr=None, short_ssl=False, short_stage2=False):
    dst = W/f'repo_{tag}'
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(BASE, dst)
    p = dst/REL/'train_exp2_v2.py'
    s = p.read_text()
    s = s.replace('st=copy.deepcopy(model.state_dict())',
                  "st={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}")
    if ssl_lr is not None:
        s = s.replace('c.ssl_lr=3e-4', f'c.ssl_lr={ssl_lr}')
    if all_lr is not None:
        s = s.replace('c.all_lr=8e-5', f'c.all_lr={all_lr}')
    if short_ssl:
        s = s.replace('for step in range(1,5001):', 'for step in range(1,2001):')
    if short_stage2:
        s = s.replace('for ep in range(1,31):', 'for ep in range(1,11):')
        s = s.replace(
            "[('partial_freeze',10,c.head_lr,False),('all',50,c.all_lr,True)]",
            "[('partial_freeze',5,c.head_lr,False),('all',10,c.all_lr,True)]"
        )
    p.write_text(s)
    compile(s, str(p), 'exec')
    return dst

SSL_TRIAL = r"""
import json, sys, torch
from pathlib import Path
import train_exp2_v2 as v
lr=float(sys.argv[1]); out=Path(sys.argv[2])
(out/'checkpoints').mkdir(parents=True,exist_ok=True); (out/'splits').mkdir(exist_ok=True)
c=v.cfg_v2(); c.ssl_lr=lr; v.b.seed_all(c.seed)
_,_,_,st,ossl,otr,ov,ptr,pv=v.prepare(c,Path('/kaggle/input'),out)
dev=torch.device('cuda')
model=v.b.Net(c).to(dev)
ch,hist=v.ssl_stage(model,c,ossl,otr,ov,ptr,pv,dev,out)
state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
torch.save({'model':state,'metrics':ch['metrics'],'ssl_lr':lr},out/'ssl_result.pt')
(out/'ssl_result.json').write_text(json.dumps({'ssl_lr':lr,'id':ch['id'],"metrics":ch['metrics']},indent=2))
print('SSL_TRIAL_DONE',json.dumps({'ssl_lr':lr,'id':ch['id'],'metrics':ch['metrics']}),flush=True)
"""

STAGE2_TRIAL = r"""
import json, sys, torch
from pathlib import Path
import train_exp2_v2 as v
ssl_lr=float(sys.argv[1]); all_lr=float(sys.argv[2]); ssl_ck=Path(sys.argv[3]); out=Path(sys.argv[4])
(out/'checkpoints').mkdir(parents=True,exist_ok=True); (out/'splits').mkdir(exist_ok=True)
c=v.cfg_v2(); c.ssl_lr=ssl_lr; c.all_lr=all_lr; v.b.seed_all(c.seed)
_,_,_,st,ossl,otr,ov,ptr,pv=v.prepare(c,Path('/kaggle/input'),out)
dev=torch.device('cuda')
model=v.b.Net(c).to(dev)
ck=torch.load(ssl_ck,map_location='cpu'); model.load_state_dict(ck['model'])
pc,ph=v.public_stage(model,c,ptr,ov,pv,dev,out)
fc,fh=v.final_stage(model,c,otr,ptr,ov,pv,dev,out)
state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
torch.save({'model':state,'metrics':fc['metrics'],'ssl_lr':ssl_lr,'all_lr':all_lr},out/'stage2_result.pt')
(out/'stage2_result.json').write_text(json.dumps({'ssl_lr':ssl_lr,'all_lr':all_lr,'public':pc['metrics'],'final':fc['metrics']},indent=2))
print('STAGE2_TRIAL_DONE',json.dumps({'ssl_lr':ssl_lr,'all_lr':all_lr,'final':fc['metrics']}),flush=True)
"""

def make_trial(tag, ssl_lr=None, all_lr=None, stage='ssl'):
    repo = patch_repo(tag, ssl_lr=ssl_lr, all_lr=all_lr,
                      short_ssl=(stage=='ssl'), short_stage2=(stage=='stage2'))
    runner = repo/REL/('ssl_trial.py' if stage=='ssl' else 'stage2_trial.py')
    runner.write_text(SSL_TRIAL if stage=='ssl' else STAGE2_TRIAL)
    return repo, runner

def launch(gpu, tag, repo, runner, args):
    out = W/tag
    if out.exists(): shutil.rmtree(out)
    log = W/f'{tag}.log'
    f = open(log,'w',buffering=1)
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['PYTHONUNBUFFERED']='1'
    cmd=['python',runner.name,*map(str,args),str(out)]
    p=subprocess.Popen(cmd,cwd=repo/REL,env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
    return {'tag':tag,'p':p,'f':f,'log':log,'out':out}

def wait_group(jobs):
    while True:
        alive=False
        for j in jobs:
            rc=j['p'].poll()
            if rc is None: alive=True
            try:
                lines=j['log'].read_text(errors='ignore').splitlines()
                tail=lines[-1] if lines else 'starting'
            except Exception:
                tail='starting'
            print(f"{j['tag']}: {'RUNNING' if rc is None else 'DONE rc='+str(rc)} | {tail[-600:]}",flush=True)
        if not alive: break
        time.sleep(30)
    for j in jobs: j['f'].close()
    bad=[j for j in jobs if j['p'].returncode!=0]
    if bad:
        for j in bad:
            print(f"--- {j['tag']} FAILED TAIL ---",flush=True)
            print('\n'.join(j['log'].read_text(errors='ignore').splitlines()[-120:]),flush=True)
        raise SystemExit(1)

def pick(rows):
    best_own=max(r['metrics']['own_ap'] for r in rows)
    elig=[r for r in rows if r['metrics']['own_ap'] >= best_own-0.02]
    return max(elig,key=lambda r:(r['metrics']['public_mean_ap'],r['metrics']['own_ap']))

ssl_vals=[1e-4,3e-4,1e-3]
ssl_jobs=[]
for gpu,(idx,lr) in enumerate(list(enumerate(ssl_vals))[:2]):
    tag=f'ssl_tune_{idx}'
    repo,runner=make_trial(tag,ssl_lr=lr,stage='ssl')
    ssl_jobs.append(launch(gpu,tag,repo,runner,[lr]))
print('GREEDY_STAGE1_BATCH1_STARTED',ssl_vals[:2],flush=True)
wait_group(ssl_jobs)

idx=2; lr=ssl_vals[2]; tag=f'ssl_tune_{idx}'
repo,runner=make_trial(tag,ssl_lr=lr,stage='ssl')
ssl_job3=launch(0,tag,repo,runner,[lr])
print('GREEDY_STAGE1_BATCH2_STARTED',[lr],flush=True)
wait_group([ssl_job3])

ssl_rows=[]
for idx,lr in enumerate(ssl_vals):
    d=json.loads((W/f'ssl_tune_{idx}'/'ssl_result.json').read_text())
    ssl_rows.append({'idx':idx,'ssl_lr':lr,'metrics':d['metrics'],'id':d['id']})
ssl_win=pick(ssl_rows)
selected_ssl_lr=ssl_win['ssl_lr']
selected_ssl_ck=W/f"ssl_tune_{ssl_win['idx']}"/'ssl_result.pt'
print('GREEDY_STAGE1_RESULT',json.dumps({'trials':ssl_rows,'winner':ssl_win},indent=2),flush=True)

all_vals=[3e-5,8e-5,2e-4]
s2_jobs=[]
for gpu, (idx,lr) in enumerate(list(enumerate(all_vals))[:2]):
    tag=f'ft_tune_{idx}'
    repo,runner=make_trial(tag,ssl_lr=selected_ssl_lr,all_lr=lr,stage='stage2')
    s2_jobs.append(launch(gpu,tag,repo,runner,[selected_ssl_lr,lr,selected_ssl_ck]))
print('GREEDY_STAGE2_BATCH1_STARTED',all_vals[:2],flush=True)
wait_group(s2_jobs)

idx=2; lr=all_vals[2]; tag=f'ft_tune_{idx}'
repo,runner=make_trial(tag,ssl_lr=selected_ssl_lr,all_lr=lr,stage='stage2')
s2_job3=launch(0,tag,repo,runner,[selected_ssl_lr,lr,selected_ssl_ck])
print('GREEDY_STAGE2_BATCH2_STARTED',[lr],flush=True)
wait_group([s2_job3])

s2_rows=[]
for idx,lr in enumerate(all_vals):
    d=json.loads((W/f'ft_tune_{idx}'/'stage2_result.json').read_text())
    s2_rows.append({'idx':idx,'all_lr':lr,'metrics':d['final'],'public_stage':d['public']})
s2_win=pick(s2_rows)
selected_all_lr=s2_win['all_lr']
print('GREEDY_STAGE2_RESULT',json.dumps({'trials':s2_rows,'winner':s2_win},indent=2),flush=True)

greedy_summary={
    'protocol':'sequential_greedy',
    'ssl_lr_trials':ssl_rows,
    'selected_ssl_lr':selected_ssl_lr,
    'fine_tune_lr_trials':s2_rows,
    'selected_all_lr':selected_all_lr,
    'selection_rule':'Own AP within 0.02 of best, then public mean AP, then Own AP',
    'own_test_used_for_tuning':False
}
(W/'greedy_search.json').write_text(json.dumps(greedy_summary,indent=2))

full_repo=patch_repo('full_selected',ssl_lr=selected_ssl_lr,all_lr=selected_all_lr)
full_out=W/'exp2_v2_selected'
if full_out.exists(): shutil.rmtree(full_out)
env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='0'; env['PYTHONUNBUFFERED']='1'
full_log=W/'full_selected.log'
with open(full_log,'w',buffering=1) as f:
    p=subprocess.Popen(['python','train_exp2_v2.py','--mode','train','--input-root','/kaggle/input','--output-root',str(full_out)],cwd=full_repo/REL,env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
    while p.poll() is None:
        lines=full_log.read_text(errors='ignore').splitlines()
        print('FULL_TRAIN_RUNNING | '+(lines[-1][-700:] if lines else 'starting'),flush=True)
        time.sleep(30)
if p.returncode!=0:
    print('\n'.join(full_log.read_text(errors='ignore').splitlines()[-160:]),flush=True)
    raise SystemExit(p.returncode)
print('FULL_TRAIN_COMPLETE',flush=True)

shutil.copy2(W/'greedy_search.json',full_out/'greedy_search.json')

test_log=W/'final_test.log'
with open(test_log,'w',buffering=1) as f:
    p=subprocess.Popen(['python','train_exp2_v2.py','--mode','final_test','--input-root','/kaggle/input','--output-root',str(full_out)],cwd=full_repo/REL,env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
    while p.poll() is None:
        lines=test_log.read_text(errors='ignore').splitlines()
        print('FINAL_TEST_RUNNING | '+(lines[-1][-700:] if lines else 'starting'),flush=True)
        time.sleep(15)
if p.returncode!=0:
    print('\n'.join(test_log.read_text(errors='ignore').splitlines()[-120:]),flush=True)
    raise SystemExit(p.returncode)

print('PIPELINE_COMPLETE',flush=True)
print((full_out/'final_test_metrics.json').read_text(),flush=True)

for p in W.glob('ssl_tune_*/checkpoints'):
    shutil.rmtree(p,ignore_errors=True)
for p in W.glob('ft_tune_*/checkpoints'):
    shutil.rmtree(p,ignore_errors=True)
for p in W.glob('repo_*'):
    shutil.rmtree(p,ignore_errors=True)

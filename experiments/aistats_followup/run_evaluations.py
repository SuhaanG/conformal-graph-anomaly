"""Consume completed score files while training runs; bounded local batch job."""
from pathlib import Path
import sys,time,subprocess,concurrent.futures,json
OUT=Path(__file__).resolve().parent
JOBS=[(d,m,s) for d in ('amazon','tolokers') for s in range(10) for m in ('dominant_pygod','isolation_forest')]

def run(job):
    d,m,s=job;name=f'{d}_{m}_{s}'
    with (OUT/f'evaluation_{name}.log').open('w') as f:
        r=subprocess.run([sys.executable,'-u',str(OUT/'evaluate_headlines.py'),d,m,str(s)],stdout=f,stderr=subprocess.STDOUT)
    return job,r.returncode

def main():
    start=time.time();pending=set(JOBS);active={};failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        while pending or active:
            for job in sorted(pending):
                d,m,s=job
                if (OUT/f'evaluation_{d}_{m}_{s}_summary.csv').exists():pending.remove(job);continue
                if len(active)<2 and (OUT/f'{d}_{m}_scores_{s}.npz').exists():
                    active[ex.submit(run,job)]=job;pending.remove(job);print('DISPATCH',job,flush=True)
            for f in list(active):
                if f.done():
                    job,code=f.result();del active[f]
                    if code:failures.append(dict(job=job,exit_code=code))
                    print('FINISHED',job,code,flush=True)
            (OUT/'evaluation_batch_status.json').write_text(json.dumps(dict(pending=len(pending),running=len(active),failures=failures,elapsed_seconds=time.time()-start),indent=2))
            if time.time()-start>8*3600:raise TimeoutError('Training/evaluation batch exceeded eight hours')
            if pending or active:time.sleep(5)
    assert not failures,failures
    print('ALL EVALUATIONS COMPLETE',flush=True)
if __name__=='__main__':main()

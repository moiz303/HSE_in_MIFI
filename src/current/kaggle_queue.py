"""One explicitly selected submission, after UTC quota reset; never retries upload.

Waiting happens in a separate bounded process in <=30-second slices, not in
model reasoning. Any ambiguous upload error stops without consuming another try.
"""
import argparse
import os
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import time
import threading
import numpy as np
import pandas as pd
from src.current.data import ROOT,sha


def utc(value):return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(timezone.utc)
def now():return datetime.now(timezone.utc)
def field(row,name):return row.get(name) if isinstance(row,dict) else getattr(row,name,None)
def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


def network_call(state_path,phase,function,*args):
    def expired():
        write(state_path,{'status':'ambiguous_network_timeout','phase':phase,'time_utc':now().isoformat(),
            'note':'Stopped after60seconds; no automatic retry; inspect remote history first'})
        os._exit(124)
    watchdog=threading.Timer(60,expired);watchdog.daemon=True;watchdog.start()
    try:return function(*args)
    finally:watchdog.cancel()


def count_today(rows,instant):
    day=instant.astimezone(timezone.utc).date().isoformat()
    return sum(str(field(row,'date') or '').startswith(day) for row in rows)


def duplicate(rows,digest,filename):
    return next((row for row in rows if digest in str(field(row,'description') or '') or
        str(field(row,'file_name') or field(row,'fileName') or '')==filename),None)


def validate(plan,instant,history,remote):
    if instant<utc(plan['not_before_utc']):raise ValueError('Quota reset has not occurred')
    if instant>utc(plan['expires_utc']):raise ValueError('Queued authorization expired')
    if count_today(remote,instant)>=plan['max_daily_submissions']:raise ValueError('No live daily quota')
    if any(row['submission_sha256']==plan['submission_sha256'] for row in history['submissions']):raise ValueError('Already in submission registry')
    if duplicate(remote,plan['submission_sha256'],Path(plan['submission']).name):raise ValueError('Remote duplicate detected')
    for key in ['submission','artifact','verification']:
        if sha(ROOT/plan[key])!=plan[key+'_sha256']:raise ValueError('Changed queued input: '+key)
    verification=json.loads((ROOT/plan['verification']).read_text())
    if verification.get('status')!='passed' or not any(r['candidate']==plan['candidate'] and r.get('passed') for r in verification.get('checks',[])):
        raise ValueError('Candidate has no passing fresh-process verification')
    prediction=pd.read_csv(ROOT/plan['submission']);template=pd.read_csv(ROOT/'data/raw/kaggle/sample_submit.csv',usecols=['index'])
    if list(prediction.columns)!=['index','SI VACV'] or len(prediction)!=597:raise ValueError('Bad submission schema')
    np.testing.assert_array_equal(prediction['index'],template['index'])
    if not np.isfinite(prediction['SI VACV']).all() or prediction['SI VACV'].lt(0).any():raise ValueError('Invalid SI predictions')


def run(plan_path):
    plan=json.loads(plan_path.read_text());state_path=ROOT/plan['state_file']
    state={'candidate':plan['candidate'],'status':'waiting','not_before_utc':plan['not_before_utc'],'max_uploads':1,'started_utc':now().isoformat()}
    write(state_path,state);print(json.dumps(state),flush=True)
    start=now()
    if utc(plan['not_before_utc'])-start>timedelta(hours=6):raise ValueError('Unreasonably long queue')
    while now()<utc(plan['not_before_utc']):
        if (ROOT/plan['cancel_file']).exists():
            state.update(status='cancelled_before_upload');write(state_path,state);return
        time.sleep(min(30.,max(.1,(utc(plan['not_before_utc'])-now()).total_seconds())))
    if (ROOT/plan['cancel_file']).exists():
        state.update(status='cancelled_before_upload');write(state_path,state);return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();network_call(state_path,'authenticate',api.authenticate)
    remote=network_call(state_path,'quota_preflight',api.competition_submissions,plan['competition'])
    registry_path=ROOT/'reports/current/kaggle_goal_submissions.json';history=json.loads(registry_path.read_text())
    validate(plan,now(),history,remote)
    state.update(status='preflight_passed',live_today_before_upload=count_today(remote,now()),checked_utc=now().isoformat());write(state_path,state)
    print(json.dumps(state),flush=True)
    if (ROOT/plan['cancel_file']).exists():
        state.update(status='cancelled_before_upload');write(state_path,state);return
    # Exactly one upload call: a network exception is not automatically retried.
    result=network_call(state_path,'upload_once',api.competition_submit,str(ROOT/plan['submission']),plan['message'],plan['competition'])
    ref=int(json.loads(str(result))['ref'])
    record={k:plan[k] for k in ['candidate','submission','submission_sha256','mean_oof_mse']}
    record.update(ref=ref,result=str(result),official_score=None,status='submitted',submitted_utc=now().isoformat(),submission_attempt_in_current_goal=len(history['submissions'])+1,
        quota_window_utc=now().date().isoformat(),explicitly_selected_before_reset=True)
    history['submissions'].append(record);history['current_window_remaining_after_upload']=plan['max_daily_submissions']-count_today(remote,now())-1
    write(registry_path,history);state.update(status='submitted',ref=ref);write(state_path,state);print(json.dumps(state),flush=True)
    deadline=now()+timedelta(minutes=10)
    while now()<deadline:
        rows=network_call(state_path,'poll_score',api.competition_submissions,plan['competition']);row=next((r for r in rows if int(field(r,'ref'))==ref),None)
        if row is not None:
            score=field(row,'public_score') or field(row,'publicScore')
            status=str(field(row,'status'))
            if score is not None and str(score)!='':
                history=json.loads(registry_path.read_text())
                for entry in history['submissions']:
                    if entry.get('ref')==ref:entry.update(official_score=float(score),status=status)
                write(registry_path,history);state.update(status='scored',official_score=float(score));write(state_path,state);print(json.dumps(state),flush=True);return
            if 'ERROR' in status or 'CANCEL' in status:
                state.update(status='submission_failed',remote_status=status);write(state_path,state);return
        time.sleep(30)
    state.update(status='submitted_score_pending',note='Polling deadline reached; do not upload again');write(state_path,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',default='reports/current/queued_submission_plan.json');args=parser.parse_args()
    path=ROOT/args.plan
    plan=json.loads(path.read_text());lock=ROOT/plan['lock_file'];lock.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lock.open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'started_utc':now().isoformat(),'plan':args.plan},stream)
    except FileExistsError:
        raise SystemExit('Queue already launched; inspect its live process/history, do not automatically relaunch')
    try:run(path)
    except Exception as error:
        plan=json.loads(path.read_text());state={'candidate':plan['candidate'],'status':'stopped_without_upload_retry','error':type(error).__name__+': '+str(error),'time_utc':now().isoformat(),
            'note':'An upload exception may be ambiguous; inspect remote submissions before any manual retry'}
        write(ROOT/plan['state_file'],state);print(json.dumps(state),flush=True);raise


if __name__=='__main__':main()

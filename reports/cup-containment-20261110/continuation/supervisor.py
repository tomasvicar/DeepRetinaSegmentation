"""One approved CPU cup continuation; signal only its own newly isolated group."""
import datetime, hashlib, json, os, pathlib, signal, subprocess, time
run=pathlib.Path.home()/'code/DeepRetinaSegmentation_cup_containment_runs/fresh-paired-20261110T183601Z'
source=pathlib.Path.home()/'code/DeepRetinaSegmentation_cup_containment'
old=pathlib.Path.home()/'code/DeepRetinaSegmentation_cup_containment_runs/full-20261110T174434Z'
python=pathlib.Path.home()/'code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
review=json.loads((run/'root-review.json').read_text())
assert review['execution_authorized']['one_fresh_output_continuation_on_ada'] is True
assert sha(run/'helper.py')==review['helper_sha256']=='0379fdfc146a693b26f0623986e5cb8593356cd5289c4929a87caf8696334c47'
assert sha(run/'plan.json')==review['proposal_sha256']=='3f82e625b62f089bc7a3a36a9cfb3072e8367821147fba48e1f9831826d5f4ee'
assert not (run/'result').exists() and not (run/'launch.json').exists()
command=['nice','-n','10',str(python),'-u',str(run/'helper.py'),'--source',str(source),'--plan',str(run/'plan.json'),'--manifest',str(pathlib.Path.home()/'code/DeepRetinaSegmentation_for_Leipzig/data/unified25/manifest.json'),'--reference-dir',str(source/'reports/disc-reference'),'--prefix',str(old/'result/per-photo.jsonl'),'--prefix-terminal',str(old/'terminal.json'),'--output',str(run/'result'),'--expected-helper-sha256',review['helper_sha256']]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(source/'src')+':'+str(source/'scripts'))
started=time.monotonic()
with (run/'continuation.log').open('x') as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=run,start_new_session=True)
 metadata=dict(child_pid=process.pid,child_process_group=process.pid,supervisor_pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH']},helper_sha256=sha(run/'helper.py'),proposal_sha256=sha(run/'plan.json'),root_review_sha256=sha(run/'root-review.json'),root_review_url='https://github.com/tomasvicar/DeepRetinaSegmentation/blob/1002a00/reports/cup-containment-20261110/continuation-root-review.json',prefix_sha256=sha(old/'result/per-photo.jsonl'),prefix_terminal_sha256=sha(old/'terminal.json'),source_commit=review['forward_package_commit'],wall_limit_seconds=4800,rss_limit_kib=2097152,max_observed_rss_kib=0,no_gpu=True,no_test=True,automatic_retry=False,prior_aborted_full_seconds=json.loads((old/'terminal.json').read_text())['elapsed_seconds'])
 (run/'launch.json').write_text(json.dumps(metadata,indent=2)+'\n')
 reason=None
 while process.poll() is None:
  try:
   status=pathlib.Path(f'/proc/{process.pid}/status').read_text();rss=next((int(line.split()[1]) for line in status.splitlines() if line.startswith('VmRSS:')),0)
  except FileNotFoundError:rss=0
  metadata['max_observed_rss_kib']=max(metadata['max_observed_rss_kib'],rss)
  if rss>=2097152 or time.monotonic()-started>=4800:
   reason='rss_limit' if rss>=2097152 else 'wall_limit'
   try:os.killpg(process.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()
   break
  time.sleep(2)
 metadata.update(returncode=process.wait(),termination_reason=reason,elapsed_seconds=time.monotonic()-started,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_report_exists=(run/'result/report.json').exists())
 if (run/'result/report.json').exists():metadata['full_report_sha256']=sha(run/'result/report.json')
 (run/'terminal.json').write_text(json.dumps(metadata,indent=2)+'\n')

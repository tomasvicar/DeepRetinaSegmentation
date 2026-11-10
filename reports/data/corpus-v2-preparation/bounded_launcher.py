from pathlib import Path
import datetime,hashlib,json,os,resource,signal,subprocess,time
home=Path.home();source=home/'code/DeepRetinaSegmentation_corpus_v2_run_fa1cf03'
assert hashlib.sha256((source/'scripts/prepare_corpus_v2.py').read_bytes()).hexdigest()=='8a745d3bddff4e94b31e855e7b8e6b10bc6562444992c78ac5243872cf5837e0'
assert hashlib.sha256((source/'reports/data/corpus-v2-geometry-policy/policy.json').read_bytes()).hexdigest()=='2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d'
for p in source.rglob('*'):
 if p.is_file():p.chmod(0o444)
for p in sorted(source.rglob('*'),reverse=True):
 if p.is_dir():p.chmod(0o555)
source.chmod(0o555)
root=home/'code/DeepRetinaSegmentation_proposed_storage';run=root/'runs/corpusv2_fa1cf03';run.mkdir(parents=True,exist_ok=False)
output=root/'corpusv2_2591_fa1cf03'
assert not output.exists()
cmd=[str(home/'code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python'),str(source/'scripts/prepare_corpus_v2.py'),'--parent',str(home/'code/DeepRetinaSegmentation_for_Leipzig/data/unified25'),'--output',str(output),'--data-root',str(home/'ownCloud/fundus_databases'),'--geometry-policy',str(source/'reports/data/corpus-v2-geometry-policy/policy.json'),'--biomisa-label-policy','named_cup_is_disc_named_disc_is_cup']
env=os.environ.copy();env.update(PYTHONPATH=str(source/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
start=time.monotonic()
with (run/'stdout.log').open('x') as out,(run/'stderr.log').open('x') as err:
 child=subprocess.Popen(cmd,cwd=source,env=env,stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.setpriority(os.PRIO_PROCESS,0,10))
 launch={'source_commit':'fa1cf03','producer_sha256':'8a745d3bddff4e94b31e855e7b8e6b10bc6562444992c78ac5243872cf5837e0','geometry_policy_sha256':'2591d8461b56433529105532c20203e0b17a48c781c1a74fd3a2901bcb301e3d','pid':child.pid,'supervisor_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.UTC).isoformat(),'command':cmd,'env':{k:env[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},'observed_child_nice':os.getpriority(os.PRIO_PROCESS,child.pid),'wall_cap_seconds':1800,'rss_cap_bytes':4*1024**3,'output':str(output),'log':str(run/'stdout.log'),'supervisor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
 peak=0;reason=None
 while child.poll() is None:
  todo=[child.pid];seen=set();rss=0
  while todo:
   pid=todo.pop()
   if pid in seen:continue
   seen.add(pid)
   try:
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    if int(fields[2])!=child.pid:continue
    rss+=int(fields[21])*os.sysconf('SC_PAGE_SIZE')
    todo.extend(int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
   except (FileNotFoundError,ProcessLookupError):pass
  peak=max(peak,rss)
  if peak>4*1024**3:reason='rss_cap'
  if time.monotonic()-start>1800:reason='wall_cap'
  if reason:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
   break
  time.sleep(.5)
 code=child.wait()
 terminal={'returncode':code,'stopped_reason':reason,'wall_seconds':time.monotonic()-start,'peak_group_rss_bytes':peak,'kernel_child_maxrss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'completed_utc':datetime.datetime.now(datetime.UTC).isoformat()}
 (run/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n');print(json.dumps(terminal),flush=True)
 raise SystemExit(code if reason is None else 124)

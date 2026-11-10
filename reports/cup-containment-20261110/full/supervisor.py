"""Bound only this explicitly launched CPU cup study; never touch other jobs."""
import datetime, hashlib, json, os, pathlib, signal, subprocess, time
root = pathlib.Path.home() / 'code/DeepRetinaSegmentation_cup_containment'
run = pathlib.Path.home() / 'code/DeepRetinaSegmentation_cup_containment_runs/full-20261110T174434Z'
run.mkdir(parents=True, exist_ok=True)
source = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
if source != 'b4134160983b7358be1ffd06a97ff497076ddb5e':
    raise RuntimeError('Frozen source mismatch')
python = pathlib.Path.home() / 'code/DeepRetinaSegmentation_unified_run_87cf045/.venv/bin/python'
command = ['nice','-n','10',str(python),'-u',str(root/'scripts/evaluate_cup_containment.py'),'--manifest',str(pathlib.Path.home()/'code/DeepRetinaSegmentation_for_Leipzig/data/unified25/manifest.json'),'--reference-dir',str(root/'reports/disc-reference'),'--mode','full','--output',str(run/'result')]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONPATH=str(root/'src'))
started = time.monotonic()
with (run/'full.log').open('x') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=root, start_new_session=True)
    metadata = dict(source_commit=source, script_sha256=hashlib.sha256((root/'scripts/evaluate_cup_containment.py').read_bytes()).hexdigest(), command=command, child_pid=process.pid, supervisor_pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), wall_limit_seconds=6000, rss_limit_kib=2097152, environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH']}, max_observed_rss_kib=0, gpu=False, split='val')
    (run/'launch.json').write_text(json.dumps(metadata, indent=2)+'\n')
    reason = None
    while process.poll() is None:
        try:
            status = pathlib.Path(f'/proc/{process.pid}/status').read_text()
            rss = next((int(line.split()[1]) for line in status.splitlines() if line.startswith('VmRSS:')), 0)
        except FileNotFoundError:
            rss = 0
        metadata['max_observed_rss_kib'] = max(metadata['max_observed_rss_kib'], rss)
        elapsed = time.monotonic() - started
        if rss > 2097152 or elapsed >= 6000:
            reason = 'rss_limit' if rss > 2097152 else 'wall_limit'
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        time.sleep(5)
    metadata.update(returncode=process.wait(), termination_reason=reason, elapsed_seconds=time.monotonic()-started, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), full_report_exists=(run/'result/report.json').exists())
    (run/'terminal.json').write_text(json.dumps(metadata, indent=2)+'\n')

"""Explicitly authorized full-frame native AV CPU smoke, bounded to one owned group."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

MIB = 1024


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))


def group_rss(pgid):
    total = 0
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if os.getpgid(int(entry.name)) == pgid:
                total += int(next((x.split()[1] for x in (entry / 'status').read_text().splitlines() if x.startswith('VmRSS:')), '0'))
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            pass
    return total


def stop_group(child):
    # Never signal a foreign group; child was created with start_new_session=True.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGKILL:
            return
        time.sleep(.1)  # Also terminate surviving owned descendants after leader exit.
    child.wait(timeout=3)


def bounded(command, env, launch, terminal, logpath, *, seconds=293, rss_limit=3072*MIB, memory_floor=3072*MIB):
    child = None
    started, peak, reason, error = time.monotonic(), 0, None, None
    old = {}
    def interrupt(signum, frame):
        raise InterruptedError('operator_signal_' + str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old[sig] = signal.signal(sig, interrupt)
        with logpath.open('x') as log:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            launch.write_text(json.dumps({'pid':child.pid,'command':command,'maximum_seconds':seconds,'max_group_RSS_KiB':rss_limit,'minimum_MemAvailable_KiB':memory_floor}, indent=2)+'\n')
            while child.poll() is None:
                peak = max(peak, group_rss(child.pid))
                if time.monotonic()-started >= seconds:
                    reason = 'deadline'
                elif peak > rss_limit:
                    reason = 'own_group_RSS_limit'
                elif available() < memory_floor:
                    reason = 'system_MemAvailable_limit'
                if reason:
                    break
                time.sleep(.1)
    except BaseException as exc:
        reason, error = 'supervisor_exception', type(exc).__name__ + ': ' + str(exc)
    finally:
        if child is not None:
            stop_group(child)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    result = {'kind':'actual_external_native_AV_real_CPU_smoke_exit','observed_returncode':child is not None,'returncode':None if child is None else child.returncode,'elapsed_seconds':time.monotonic()-started,'peak_group_RSS_KiB':peak,'stop_reason':reason,'error':error,'final_quality_or_full_data_training':False,'automatic_retry':False}
    terminal.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if child is not None and child.returncode == 0 and reason is None else 75


def worker(phase, data, runs):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from deepretina import legacy_training as m
    if phase == 'train':
        from deepretina.legacy_training_cli import main
        main(['--manifest',str(data/'manifest.json'),'--config',str(Path(__file__).resolve().parents[1]/'configs/native-legacy-av.toml'),'--epochs','1','--device','cpu','--output',str(runs/'av')])
        return
    from deepretina.legacy.av import load_av
    previous = json.loads((runs/'av-train-observed-exit.json').read_text())
    if previous['returncode'] != 0 or previous['stop_reason'] is not None:
        raise ValueError('Require original successful observed training exit')
    model, payload = load_av(runs/'av/last.pt', 'cpu')
    m.require_finite(payload['state_dict'])
    m.require_finite(payload['training_state']['optimizer'])
    recipe = payload['training_recipe']
    if not recipe['fresh_initialization'] or not recipe['architecture_matches_original_default'] or payload['preprocessing']['profile'] != 'legacy25_raw_av4':
        raise ValueError('Wrong architecture/lineage/preprocessing')
    config = m.LegacyTrainingConfig(task='av', epochs=1)
    torch.manual_seed(42)
    initial = m.build_fresh(config).state_dict()
    names = set(dict(model.named_parameters()))
    changed = sum(not torch.equal(initial[k], payload['state_dict'][k]) for k in names)
    del initial
    if not changed:
        raise ValueError('No actual same-seed parameter update')
    dataset = m.NativeDataset(data/'manifest.json', 'val', config)
    image, target, valid, ids = dataset.sample(dataset.photo_ids[0], crop=False, observer_index=0)
    if tuple(image.shape) != (4,1031,1033):
        raise ValueError('Require exact full1031x1033 canonical AV frame, no resize/crop')
    with torch.inference_mode():
        logits = model(image[None])
    if tuple(logits.shape) != (1,3,1031,1033) or not torch.isfinite(logits).all():
        raise ValueError('Invalid full-frame safe reload')
    coverage = payload['coverage']
    summary = json.loads((runs/'av/training_summary.json').read_text())
    if not (coverage['all_eligible_annotations_contributed'] and coverage['all_eligible_photos_seen'] and summary['completed_epochs']==1 and summary['steps']>0 and summary['test_evaluation_performed'] is False):
        raise ValueError('Incomplete functional subset coverage')
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    proof = {'kind':'native_AV_production_fullframe_functional_smoke_only','architecture':payload['architecture'],'preprocessing':payload['preprocessing'],'changed_trainable_parameter_tensors_vs_seed42':changed,'reload_eval_shape':list(logits.shape),'VAL_annotation_ids':ids,'coverage':coverage,'finite_optimizer_and_weights':True,'last_sha256':hashlib.sha256((runs/'av/last.pt').read_bytes()).hexdigest(),'CUDA_initialized':False,'TEST_evaluation':False,'quality_or_full_data_claim':False}
    with (runs/'av-safe-verification.json').open('x') as stream:
        json.dump(proof,stream,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['train','verify'],required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.inside:
        if os.environ.get('NATIVE_AV_SMOKE_PARENT') != str(os.getppid()):
            raise ValueError('Internal child only')
        worker(a.phase,a.data,a.runs)
        return 0
    if not a.execute:
        raise ValueError('Pending helper: explicit root authorization/--execute required')
    if 4 not in os.sched_getaffinity(0) or not a.runs.is_dir():
        raise ValueError('Require existing runs directory and CPU4')
    paths=[a.runs/f'av-{a.phase}-{s}' for s in ['launch.json','observed-exit.json','cli.log']]
    if any(x.exists() for x in paths) or (a.phase=='train' and (a.runs/'av').exists()):
        raise FileExistsError('No overwrite/automatic retry')
    if available() < 5500*MIB:
        print('DEFER: MemAvailable below5500MiB; no child/output created')
        return 75
    source=Path(__file__).resolve().parents[1]
    command=['nice','-n','17','ionice','-c','3','taskset','-c','4',sys.executable,str(Path(__file__).resolve()),'--inside','--phase',a.phase,'--data',str(a.data),'--runs',str(a.runs)]
    env=dict(os.environ,NATIVE_AV_SMOKE_PARENT=str(os.getpid()),CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return bounded(command,env,*paths)


if __name__=='__main__':
    raise SystemExit(main())

"""Exercise the installed wheel only, using a previously verified TRAIN image."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/var/home/tom/code/DeepRetinaSegmentation_model_backups/release-installed-20261110')
REFERENCE = Path('/tmp/deepretina-source-av-release-20261110-2052')
SITE = ROOT / 'site'
OUTPUT = ROOT / 'outputs'
assert not OUTPUT.exists()
wheel = next((ROOT / 'dist').glob('*.whl'))
with zipfile.ZipFile(wheel) as archive:
    catalog = json.loads(archive.read('deepretina/pretrained/registry.json'))
    hashes = {}
    for name, model in catalog['models'].items():
        data = archive.read('deepretina/pretrained/' + model['filename'])
        hashes[name] = hashlib.sha256(data).hexdigest()
        assert hashes[name] == model['sha256']
    assert len(hashes) == 9
wrapper = '''import json,pathlib,deepretina,torch
site=pathlib.Path(__import__('os').environ['PYTHONPATH']).resolve()
assert pathlib.Path(deepretina.__file__).resolve().is_relative_to(site)
from deepretina.cli import main
main()
r={'module_origin':deepretina.__file__,'cuda_initialized':torch.cuda.is_initialized(),'intra_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads()}
pathlib.Path('installed-runtime.json').write_text(json.dumps(r)+'\\n')
'''
command = [sys.executable, '-c', wrapper, 'predict', '--model', 'legacy-av', '--av-raw-matlab-source', '--input', str(REFERENCE / 'im0002.ppm'), '--dataset', 'stare', '--output', str(OUTPUT), '--device', 'cpu']
env = os.environ.copy()
env.update(PYTHONPATH=str(SITE), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
start = time.monotonic()
with (ROOT / 'stdout.json').open('x') as stdout, (ROOT / 'stderr.log').open('x') as stderr:
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
    launch = {'utc':datetime.datetime.now(datetime.UTC).isoformat(), 'pid':child.pid, 'nice':os.getpriority(os.PRIO_PROCESS, child.pid), 'command':command, 'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(), 'model_sha256':hashes}
    (ROOT / 'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    peak = 0
    while child.poll() is None:
        try:
            status = Path(f'/proc/{child.pid}/status').read_text()
            peak = max(peak, int(next(line for line in status.splitlines() if line.startswith('VmRSS:')).split()[1])*1024)
        except (FileNotFoundError, StopIteration):
            pass
        if time.monotonic()-start > 180 or peak > 6*1024**3:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            raise RuntimeError('Installed CLI resource watchdog triggered')
        time.sleep(.2)
    assert child.wait() == 0, (ROOT / 'stderr.log').read_text()
import numpy as np
from PIL import Image
comparisons = {}
with np.load(REFERENCE / 'outputs/probabilities.npz', allow_pickle=False) as original, np.load(OUTPUT / 'probabilities.npz', allow_pickle=False) as installed:
    assert set(original.files) == set(installed.files)
    for name in original.files:
        assert np.array_equal(original[name], installed[name]), name
        comparisons[name] = {'shape':list(installed[name].shape), 'dtype':str(installed[name].dtype), 'array_exact':True}
for name in ('artery.png','vein.png','vessel.png','classes.png','overlay.png'):
    with Image.open(REFERENCE / 'outputs' / name) as a, Image.open(OUTPUT / name) as b:
        assert np.array_equal(np.asarray(a), np.asarray(b)), name
metadata = json.loads((OUTPUT / 'prediction.json').read_text())
for outputs in (metadata['outputs'], metadata['vessel_prediction']['outputs']):
    for value in outputs.values():
        if value is not None:
            assert Path(value).is_file(), value
runtime = json.loads((ROOT / 'installed-runtime.json').read_text())
assert not runtime['cuda_initialized']
proof = {'scope':'Actual installed wheel CPU source-AV CLI on canonical TRAIN STARE im0002, automatic vessels/FOV, no MATLAB/oracle/GT scoring/TEST', 'wheel_sha256':launch['wheel_sha256'], 'wheel_bytes':wheel.stat().st_size, 'model_sha256':hashes, 'wall_seconds':time.monotonic()-start, 'peak_sampled_rss_bytes':peak, 'kernel_maxrss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024, 'runtime':runtime, 'arrays':comparisons, 'all_png_pixels_exact_to_source_release':True, 'all_output_paths_exist':True, 'source_release_commit':'ad72647', 'nice':launch['nice'], 'cuda_visible_devices':'', 'input_sha256':hashlib.sha256((REFERENCE / 'im0002.ppm').read_bytes()).hexdigest(), 'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(ROOT / 'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
print(json.dumps(proof), flush=True)

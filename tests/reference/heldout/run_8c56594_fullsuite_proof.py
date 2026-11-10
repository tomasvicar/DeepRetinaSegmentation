"""External CPU full-suite receipt; never supply actual evaluation inputs."""
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

root=Path('/tmp/deepretina-heldout-full-8c56594')
expected='8c56594f36a92e569e901fcce0ed99ab2179629f'
log=Path('/tmp/deepretina-heldout-full-8c56594-pytest.log')
proof=Path('/tmp/deepretina-heldout-full-8c56594-proof.json')
os.chdir(root)
sys.path.insert(0,str(root/'src'))
from deepretina.selection import evaluation_source_identity

def git(*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if git('rev-parse','HEAD')!=expected or git('status','--porcelain'):
    raise RuntimeError('Expected clean exact source commit')
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
torch.set_num_threads(1)
before=evaluation_source_identity()
started=datetime.now(UTC).isoformat()
with log.open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
    print('Exact source',expected,'UTC',started,flush=True)
    result=pytest.main(['-q','-p','no:cacheprovider'])
finished=datetime.now(UTC).isoformat()
after=evaluation_source_identity()
source_unchanged=before['evaluation_source_sha256']==after['evaluation_source_sha256']
tracked_clean=not git('status','--porcelain')
observed={
 'kind':'unified_stage2_fullsuite_cpu_proof','source_commit':expected,'source_tree':git('rev-parse','HEAD^{tree}'),
 'started_at_utc':started,'finished_at_utc':finished,'pytest_exit_code':int(result),
 'source_checkout':str(root),'python_executable':sys.executable,'pytest_version':pytest.__version__,
 'command':['python','EXTERNAL_PROOF_RUNNER','pytest.main(-q -p no:cacheprovider)'],
 'runner_sha256':sha(__file__),'pytest_log_sha256':sha(log),'pytest_log_bytes':log.stat().st_size,
 'environment':{key:os.environ.get(key) for key in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONPATH']},
 'nice':os.getpriority(os.PRIO_PROCESS,0),'initial_torch_cpu_threads':1,'final_torch_cpu_threads':torch.get_num_threads(),
 'cuda_initialized_after_suite':torch.cuda.is_initialized(),
 'source_runtime_before':before,'source_runtime_after':after,
 'source_bytes_unchanged':source_unchanged,'tracked_git_tree_clean_after':tracked_clean,
 'runtime_note':'Tests may intentionally change backend flags/thread settings; both actual observations are retained, not a claim of one immutable in-memory runtime. Source byte digest excludes .pyc.',
 'scope':'Full suite using synthetic data and available original/bundled weight references; no actual corpus VAL/TEST evaluation, selection or GPU experiment.',
 'prior_failed_pass':{'source_commit':'544e399a1ef658d8b28b53c472ed520be5e84550','log_sha256':sha('/tmp/deepretina-heldout-full-544e399-pytest.log'),'result':'682 passed,4 failed','causes':['git-archive-only tree lacked Git HEAD required by3 HPO tests','synthetic stage2 CLI fixture inherited earlier deterministic flags unlike fresh interpreter; production gate correctly rejected mismatch'],'resolution':'Real isolated shallow Git checkout plus fixture default/reset isolation; backend mismatch remains explicitly tested and rejected.'},
}
with proof.open('x') as stream:json.dump(observed,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({'exit_code':int(result),'source_bytes_unchanged':source_unchanged,'tracked_clean':tracked_clean,'proof':str(proof),'log':str(log)}))
if not source_unchanged or not tracked_clean or torch.cuda.is_initialized():raise RuntimeError('Source/GPU invariant failed')
raise SystemExit(result)

from pathlib import Path
import hashlib,json,subprocess,time
root=Path.home()/'code/DeepRetinaSegmentation_unified_run_3157bb9'
data=Path.home()/'code/DeepRetinaSegmentation_proposed_storage/corpusv2_2591_fa1cf03'
run=data.parent/'runs/corpusv2_fa1cf03'
expected='00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52'
h=hashlib.sha256()
for p in sorted([*root.glob('src/**/*.py'),*root.glob('scripts/*.py'),root/'pyproject.toml',root/'uv.lock']):
 h.update(p.relative_to(root).as_posix().encode());h.update(b'\0');h.update(p.read_bytes())
assert h.hexdigest()=='8e0481fff8ef53beca760acf074e96b5b73305e3adea2849e79805fb961cb30b'
import torch
from deepretina.integrity import verify_prepared_content
from deepretina.unified import UnifiedConfig,UnifiedDataset
started=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
assert verify_prepared_content(data/'manifest.json',expected=expected)==expected
config=UnifiedConfig(purpose='full_data',device='cpu',epochs=60,epoch_samples=8192)
train=UnifiedDataset(data/'manifest.json','train',config);val=UnifiedDataset(data/'manifest.json','val',config)
assert len(train)==2529 and len(val)==770
choices=[]
for name in ['drions','idrid_disc']:
 choices.append(next(i for i,rs in enumerate(train.groups) if any(r['dataset']==name for r in rs)))
for shape in [[1934,1956],[2000,2992]]:
 choices.append(next(i for i,rs in enumerate(train.groups) if any(r['dataset']=='gamma' and r['original_size']==shape for r in rs)))
choices.append(next(i for i,rs in enumerate(train.groups) if any(r['dataset']=='biomisa' and r['metadata'].get('cup_missing') for r in rs)))
choices.append(next(i for i,rs in enumerate(train.groups) if any(r['dataset']=='biomisa' and not r['metadata'].get('cup_missing') for r in rs)))
samples=[]
for i in choices:
 image,target,valid,trace=train.sample(i,return_trace=True)
 assert torch.isfinite(image).all() and torch.isfinite(target).all()
 assert image.shape[0]==3 and target.shape==valid.shape and target.shape[0]==5
 assert target.shape[1:]==image.shape[1:]
 assert not valid[:3].any()
 assert not (valid[3]&valid[4]&(target[4]>target[3])).any()
 samples.append({'photo_id':train.photo_ids[i],'trace':trace,'shape':list(image.shape),'valid_channels':[bool(v.any()) for v in valid]})
parent=json.loads((Path.home()/'code/DeepRetinaSegmentation_for_Leipzig/data/unified25/manifest.json').read_text());combined=json.loads((data/'manifest.json').read_text());ids={r['id']:r for r in combined}
assert all(ids[r['id']]==r for r in parent)
assert [r for r in combined if r['split']=='val']==[r for r in parent if r['split']=='val']
report={'status':'PASS','consumer_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'consumer_execution_sha256':h.hexdigest(),'consumer_module':str(__import__('deepretina.unified',fromlist=['x']).__file__),'canonical_fingerprint':expected,'train_photos':len(train),'val_photos':len(val),'train_annotation_variants':len(train.rows),'val_annotation_variants':len(val.rows),'train_only_mechanical_samples':samples,'parent_rows_identical':len(parent),'validation_rows_identical':sum(r['split']=='val' for r in parent),'model_forward_performed':False,'training_or_optimizer_performed':False,'test_model_evaluation_performed':False,'test_samples_opened_by_consumer':False,'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'seconds':time.monotonic()-started}
(run/'frozen3157-reader-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

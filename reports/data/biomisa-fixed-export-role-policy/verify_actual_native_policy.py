from pathlib import Path
import hashlib,json,time
import numpy as np
from deepretina.sources import Source
from deepretina.adapters.biomisa import biomisa,FIXED_POLICY,native_binary
from deepretina.adapters.additional import resources
from scripts.propose_corpus_v2 import pixel_sha256
v1=json.loads(Path('/var/home/tom/code/DeepRetinaSegmentation_hpo/data/unified25/manifest.json').read_text());identities={r['photo_id']:r['split'] for r in v1}
rows=[];started=time.monotonic()
with Source(Path('/var/home/tom/ownCloud/fundus_databases/BIOMISA.zip')) as s:
 def emit(stem,split,image,target,valid=None,source_id=None,metadata=None,**kwargs):
  assert split=='train';assert valid[0].all();assert not (target[1].astype(bool)&~target[0].astype(bool)).any()
  refs=metadata['annotation_resources'];named={r['source_filename_role']:r for r in refs}
  outer,_=native_binary(s,named['Cup']['path']);np.testing.assert_array_equal(target[0],outer)
  if 'Disc' in named:
   inner,_=native_binary(s,named['Disc']['path']);np.testing.assert_array_equal(target[1],inner);assert valid[1].all()
  else:assert not valid[1].any() and not target[1].any()
  photo='sha256:'+pixel_sha256(image)
  if photo in identities:raise ValueError('v1 photo alias requires heldout review, no silent train use')
  raw=s.bytes(metadata['source_image'])
  rows.append({'id':stem,'source_id':source_id,'photo_id':photo,'shape':list(image.shape[:2]),'positive_pixels':[int(x.sum()) for x in target],'valid_channels':[bool(v.any()) for v in valid],'source_image':metadata['source_image'],'source_image_sha256':hashlib.sha256(raw).hexdigest(),'annotations':refs,'policy':metadata['annotation_policy'],'physical_calibration':metadata['physical_calibration']})
 biomisa(s,emit,label_policy=FIXED_POLICY)
report={'policy':FIXED_POLICY,'actual_train_candidates':len(rows),'both_channels':sum(all(r['valid_channels']) for r in rows),'disc_only':sum(r['valid_channels']==[True,False] for r in rows),'unique_photos':len({r['photo_id'] for r in rows}),'source_groups':len({r['source_id'] for r in rows}),'seconds':time.monotonic()-started,'pixel_values_and_geometry_preserved':True,'all_mapped_cups_inside_discs':True,'v1_exact_photo_overlaps':0,'v1_masks_or_test_annotations_opened':False,'source_defaults_changed':False,'registered_for_default_preparation':False,'review_before_training_required':True,'records':rows}
Path('/tmp/biomisa-fixed-policy-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)

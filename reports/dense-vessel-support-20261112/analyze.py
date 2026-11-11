"""Supplementary vessel-support analysis of saved counts; no tensor/GT decoding."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
MANIFEST='17a2277c39cb37b7eee1b2063cfd13f2cf4adf4ebdf86cf449a1f51367c5a767'
CANONICAL='00d9ef19d262eef74f0ca1350fec260c687dff5cc3a53f8911cbe3403e821e52'
INPUTS={
 'original182':'reports/frozen-original-val-nearest/validation.json',
 'TPE5_LAST60_none_050':'reports/tpe72-completed-fresh-val/fixed_terminal60-fresh-counts.json',
 'TPE5_BEST58_rot4_035':'reports/first-TPE-BEST58-rot4-threshold-20261111/threshold-0.35.json',
 'raw384_BEST59_none_050':'reports/curated-raw384-typed-fresh-VAL-20261111/raw384-BEST59-runtime-redacted.json',
}

def digest(raw):return hashlib.sha256(raw).hexdigest()
def checked_counts(c):
 a=[c[k]for k in ['tp','fp','fn','valid_pixels']]
 if any(type(v)is not int or v<0 for v in a)or sum(a[:3])>a[3]:raise ValueError('Invalid integer confusion counts')
 return np.asarray(a,dtype=np.int64)
def scores(c):
 tp,fp,fn,valid=map(int,c);return {'tp':tp,'fp':fp,'fn':fn,'valid_pixels':valid,'GT_positive_pixels':tp+fn,'GT_negative_pixels':valid-tp-fn,'dice':2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None}
def support(ids,lookup,metadata):
 return {'photos':len(ids),'known_split_groups':len({metadata[p][0]['split_group_id']for p in ids}),'sources':sorted({lookup[p]['channel_datasets']['vessel']for p in ids})}
def validate_pair(reference,candidate):
 if set(reference)!=set(candidate):raise ValueError('Photo membership differs')
 for p,left in reference.items():
  right=candidate[p]
  if left['record_ids']!=right['record_ids']or left['channel_datasets']!=right['channel_datasets']or left['dataset']!=right['dataset']:raise ValueError('Record/source mismatch')
  a,b=[checked_counts(x['channels']['vessel'])for x in [left,right]]
  if a[3]!=b[3]or a[0]+a[2]!=b[0]+b[2]:raise ValueError('GT support differs')

def bootstrap(ids,reference,candidate,metadata):
 # Same source-stratified reviewed-group design as comparison.py, restricted to
 # actual vessel-support photos in THIS supplementary domain, not all612groups.
 groups=defaultdict(list)
 for p in ids:groups[metadata[p][0]['split_group_id']].append(p)
 g=sorted(groups);strata=defaultdict(list)
 vals=np.zeros((2,len(g),3),dtype=np.int64)
 for i,k in enumerate(g):
  signature=tuple(sorted({reference[p]['dataset']for p in groups[k]}));strata[signature].append(i)
  for m,lookup in enumerate([reference,candidate]):vals[m,i]=sum((checked_counts(lookup[p]['channels']['vessel'])[:3]for p in groups[k]),np.zeros(3,dtype=np.int64))
 rng=np.random.default_rng(42);differences=[]
 for _ in range(2000):
  draw=np.concatenate([rng.choice(strata[k],len(strata[k]),replace=True)for k in sorted(strata)])
  c=vals[:,draw].sum(axis=1);dice=2*c[:,0]/(2*c[:,0]+c[:,1]+c[:,2]);differences.append(float(dice[1]-dice[0]))
 pooled=vals.sum(axis=1);point=2*pooled[:,0]/(2*pooled[:,0]+pooled[:,1]+pooled[:,2])
 return {'reference':float(point[0]),'candidate':float(point[1]),'candidate_minus_reference':float(point[1]-point[0]),'paired_percentile_95ci':np.quantile(differences,[.025,.975]).tolist(),'draws':2000,'seed':42,'unit':'known split_group_id in actual vessel domain; not proven independent patients','groups':len(g),'stratification':'dataset signature among included vessel-support photos of whole retained group','strata':{str(k):len(v)for k,v in sorted(strata.items())},'scope':'contrast of these explicitly different checkpoint roles/inference pipelines; not isolated training/capacity effect'}

def main(manifest_path):
 mb=manifest_path.read_bytes();assert digest(mb)==MANIFEST;manifest=json.loads(mb);rows=manifest['records']if isinstance(manifest,dict)else manifest
 metadata=defaultdict(list)
 for x in rows:
  if x['split']=='val'and x['task']in {'av','vessels','disc_cup'}and x.get('usable_for_training',True)and not x.get('exclude_from_training',False):metadata[x['photo_id']].append(x)
 assert len(metadata)==770
 reports={};lookups={};pins={};contexts={}
 for name,relative in INPUTS.items():
  b=(ROOT/relative).read_bytes();d=json.loads(b);m=d.get('metrics',d)
  assert (d.get('data',{}).get('manifest_sha256')or m.get('manifest_sha256'))==MANIFEST
  assert (d.get('data',{}).get('prepared_content_sha256')or m.get('prepared_content_sha256'))==CANONICAL
  lookup={x['photo_id']:x for x in m['per_photo']};assert len(lookup)==len(m['per_photo'])==770 and set(lookup)==set(metadata)
  for p,e in lookup.items():assert set(e['record_ids'])=={x['id']for x in metadata[p]}
  summed=sum((checked_counts(e['channels']['vessel'])for e in lookup.values()),np.zeros(4,dtype=np.int64));assert np.array_equal(summed,checked_counts(m['channels']['vessel']))
  pins[name]={'path':relative,'sha256':digest(b),'checkpoint_sha256':d.get('checkpoint',{}).get('sha256',m.get('checkpoint_sha256')),'kind':d.get('kind',d.get('family'))}
  contexts[name]={k:d[k]for k in ['checkpoint','configuration','source','inference_protocol','actual_inference_protocol','tta_protocol','threshold','policy_spec','reviewed_policy_sha256','evaluated_at_utc']if k in d}
  if 'provenance'in d:contexts[name]['derived_count_provenance']=d['provenance']
  reports[name]=m;lookups[name]=lookup
 assert pins['original182']['sha256']=='182ad346b7bda554ae58e76acbc3f17d99228a30bd19cfa335c89534e9f66836'
 base=lookups['original182']
 for l in lookups.values():validate_pair(base,l)
 vessel=sorted(p for p in base if base[p]['channels']['vessel']['valid_pixels']>0)
 sparse=[p for p in vessel if base[p]['channel_datasets']['vessel']=='avrdb'];dense=[p for p in vessel if p not in set(sparse)]
 assert sparse
 for lookup in lookups.values():
  assert all((c:=checked_counts(lookup[p]['channels']['vessel']))[3]==c[0]+c[2]and c[1]==0 for p in sparse)
  assert all((c:=checked_counts(lookup[p]['channels']['vessel']))[3]>c[0]+c[2] for p in dense)
 sparse_metadata_sources=sorted({x['dataset']for v in metadata.values()for x in v if any(word in json.dumps(x.get('metadata',{})).lower()for word in ['sparse','positive-only','unmarked'])})
 assert sparse_metadata_sources==['avrdb']
 result={'kind':'supplementary_dense_vessel_saved_count_analysis_NOT_benchmark_redefinition','manifest_sha256':MANIFEST,'canonical_sha256':CANONICAL,'input_pins':pins,'original_contexts_preserved':contexts,'all770_exact_record_source_and_GTsupport_checks':'PASS','metadata_sparse_source_scan':sparse_metadata_sources,'excluded_from_denseOnly':['avrdb'],'policy':'exclude source AVRDB only from vessel-supported photo domain; leave all original inputs/gates/objectives unchanged','domains':{}}
 for domain,ids in [('includeAll',vessel),('denseOnly',dense),('excluded_positiveOnly_AVRDB',sparse)]:
  dd={'support':support(ids,base,metadata),'models':{},'paired_contrasts':{}}
  for name,l in lookups.items():
   dd['models'][name]={'pooled':scores(sum((checked_counts(l[p]['channels']['vessel'])for p in ids),np.zeros(4,dtype=np.int64))),'by_source':{}}
   for source in dd['support']['sources']:
    selected=[p for p in ids if l[p]['channel_datasets']['vessel']==source]
    dd['models'][name]['by_source'][source]={'support':support(selected,base,metadata),**scores(sum((checked_counts(l[p]['channels']['vessel'])for p in selected),np.zeros(4,dtype=np.int64)))}
  if domain!='excluded_positiveOnly_AVRDB':
   for name,l in lookups.items():
    if name=='original182':continue
    dd['paired_contrasts'][name+'__minus_original182']=bootstrap(ids,base,l,metadata)
   for name in ['TPE5_BEST58_rot4_035','raw384_BEST59_none_050']:
    dd['paired_contrasts'][name+'__minus_TPE5_LAST60_none_050']=bootstrap(ids,lookups['TPE5_LAST60_none_050'],lookups[name],metadata)
  result['domains'][domain]=dd
 summary=json.loads((ROOT/'reports/tpe72-completed-fresh-val/completion-and-fresh-validation-summary.json').read_bytes())
 result['TPE5_control_LAST60_actual_fresh_runtime_context']=summary['fresh_reports']['fixed_terminal60']
 result['qualifications']=['AVRDB FP0/precision1 are structural positive-only support, not dense vessel specificity/precision.','denseOnly means remaining benchmark label domains include negatives; unknown annotation completeness is not certified by counts.','Original historical exposure and preprocessing/backend differences persist.','BEST58rot4.35 was VAL-selected; role/TTA/threshold differ from LAST60none.5. These paired intervals compare specified pipelines on identical GT, not causal optimization effect.','No simultaneous multiple-comparison adjustment, clinical/noninferiority/equivalence claim or TEST.','No original report or current pool worst-six objective altered.']
 (HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:{'support':d['support'],'pooled':{n:v['pooled']for n,v in d['models'].items()}}for k,d in result['domains'].items()},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();main(a.manifest)

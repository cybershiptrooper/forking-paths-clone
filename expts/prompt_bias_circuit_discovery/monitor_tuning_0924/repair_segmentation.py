"""Retry empty public-text segmentations with raw responses retained; never replace a question."""
import os
os.environ['MONITOR_SEGMENT_MAX_TOKENS']='16000'
import json,re,time
from pathlib import Path
from transformers import AutoTokenizer
from .cohort import ROOT,save
from ..monitor_pilot_0924 import prepare as old
from ..monitor_pilot_0924.run import write_json
from ..openrouter_client import get_client,load_cache,_cache_key,_flush

def main():
 ledger=json.loads((ROOT/'preparation.json').read_text());assert ledger.get('status')=='preparation_failures'
 records={r['sample_id']:r for r in json.loads((ROOT/'cohort.json').read_text())['records']};client=get_client().with_options(max_retries=0,timeout=180.);cache=load_cache(old.ROOT/'segmenter_cache.json')
 tok=AutoTokenizer.from_pretrained(old.MODEL,revision=old.REVISION,local_files_only=True)
 repair=dict(policy='Up to three identical additional requests for each unresolved public factual text; keep the first valid exact-reconstruction segmentation. Never replace or remove a selected question. This is input preparation, not outcome-dependent tuning.',samples=[])
 for sid in ledger['unresolved_samples']:
  r=records[sid];p=old.prefix(r);facts=[r['question']] if r['task']==5 else [m.group(1) for m in re.finditer(r'"""(.*?)"""',p,re.S)]
  for text in facts:
   msgs=[dict(role='user',content=old.LLM_SEG_PROMPT.format(text=text))];key=_cache_key(old.SEGMENTER,msgs,dict(max_tokens=16000,temperature=0.,extra_body=None));rawdir=ROOT/'segment_repair';rawdir.mkdir(exist_ok=True)
   for attempt in range(3):
    try:
     resp=client.chat.completions.create(model=old.SEGMENTER,messages=msgs,max_tokens=16000,temperature=0.)
     raw=resp.model_dump(mode='json');write_json(rawdir/f'{sid}_{old.digest(text)[:12]}_{attempt}.json',dict(request=dict(model=old.SEGMENTER,messages=msgs,max_tokens=16000,temperature=0.),response=raw))
     content=resp.choices[0].message.content or '';match=re.search(r'\[.*\]',content,re.S);units=json.loads(match.group())
     assert isinstance(units,list) and all(isinstance(x,str) for x in units)
     norm=lambda s:re.sub(r'\s+','',s)
     assert norm(''.join(units))==norm(text)
     pos=0
     for u in units:
      if not u.strip():continue
      at=text.find(u.strip(),pos);assert at>=pos;pos=at+len(u.strip())
     cache[key]=content;_flush(cache,str(old.ROOT/'segmenter_cache.json'));break
    except Exception as exc:
     print('REPAIR ATTEMPT',sid,attempt+1,repr(exc),flush=True)
     if attempt<2:time.sleep(2**attempt)
  try:
   item=old.prepare_record(r,tok,client,cache);item['segmentation_output_limit']=16000;item['frozen_sha256']=old.digest(item);save(ROOT/'inputs'/f'{sid}.json',item);repair['samples'].append(dict(sample_id=sid,status='repaired'));print('REPAIRED',sid,flush=True)
  except Exception as exc:repair['samples'].append(dict(sample_id=sid,status='failed',error=repr(exc)))
 write_json(ROOT/'segmentation_repair.json',repair)
 ledger['unresolved_samples']=[sid for sid in ledger['unresolved_samples'] if not (ROOT/'inputs'/f'{sid}.json').exists()];ledger['status']='complete' if not ledger['unresolved_samples'] else 'preparation_failures';write_json(ROOT/'preparation.json',ledger)
if __name__=='__main__':main()

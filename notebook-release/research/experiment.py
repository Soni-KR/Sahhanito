"""Reproducible development challenge set + optional local encoder comparison.

Run from the project root. Assistant-authored fixtures, not clinical ground truth.
No classifier fitting, test-set access or score/threshold optimization occurs.
"""
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent

# Each input has one primary clinical concept, or none. This comparison probes
# concept retrieval only: it cannot establish NER span/context quality.
GROUPS = {
 'chest': ['I have chest pain','3andi wja3 fi sadri','صدري يوجعني','3andi douleur fi sadri','ma3andich wja3 fi sadri','ماعنديش وجيعة في صدري','j’ai une douleur thoracique','3andi wji3a fi sadri','عندي وجع في صدري'],
 'breath': ['I have difficulty breathing','ma najamch netnaffes','ما نجمش نتنفس','3andi s3ouba fel tanaffos','نفسي مقطوع','j’ai une difficulté à respirer','I am short of breath','3andi sou3ouba fel tanaffos'],
 'headache': ['I have a headache','rasi youja3ni barcha','راسي يوجعني برشة','3andi mal à la tête','wja3 fi rasi','راسي يوجعني','I have no headache'],
 'dizziness': ['I am dizzy','3andi doukha','عندي دوخة','taw 3andi doukha','j’ai des étourdissements'],
 'fever': ['I have a fever','3andi s5ana','عندي سخانة','3andi fièvre','I have no fever'],
 'diabetes': ['I have diabetes','3andi sokkor','عندي سكري','j’ai un diabète'],
 'none': ['I need a routine prescription refill','Ey','Ma na3rafch','bonjour','3andi 7aja ghriba','Can you explain healthy eating?'],
}
PROTOTYPES={
 'chest':'chest pain, pressure or tightness in the chest',
 'breath':'shortness of breath, difficulty breathing, unable to breathe',
 'headache':'headache, pain in the head',
 'dizziness':'dizziness, feeling dizzy, lightheadedness',
 'fever':'fever, high body temperature',
 'diabetes':'diabetes, high blood sugar disease',
 'cough':'cough, coughing',
 'vomiting':'vomiting, throwing up',
 'abdominal_pain':'abdominal pain, pain in the stomach',
 'rash':'rash, itchy skin eruption',
}
STRESS=[
 ('chest','n7ess b wej3a fi sedri'),('chest','n7ess b dhaght fi sedri'),
 ('chest','je me sens oppressé dans la poitrine'),('breath','نفسي موش لاقيه'),
 ('breath','3andi nafas 9sir'),('breath','انا لا استطيع التنفس'),
 ('headache','rassi youja3ni barcha'),('dizziness','دوخة شديدة'),
 ('diabetes','3andi sokor'),('none','n7eb na3ref kifech nakel s7i7'),
]

def concepts(info):
    found=set()
    for f in info['facts']:
        if f['kind'] not in ('symptom','condition'):continue
        name=f['name']
        if 'breath_severe' in f.get('concept_aliases',[]) or name=='cannot_breathe':name='breath'
        if name in PROTOTYPES:found.add(name)
    return found

def run(transformer=True):
    from triage_pipeline import _legacy_extract_info,extract_info
    import ast,importlib.util,inspect
    snapshot=ROOT/'archive/multilingual-2026-10-10'
    old_source=(snapshot/'triage_pipeline.py').read_text(encoding='utf-8')
    fn=next(n for n in ast.parse(old_source).body if isinstance(n,ast.FunctionDef) and n.name=='_legacy_extract_info')
    assert ast.get_source_segment(old_source,fn).strip()==inspect.getsource(_legacy_extract_info).strip()
    spec=importlib.util.spec_from_file_location('_hackcure_previous_state',snapshot/'clinical_state.py')
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
    def previous_extractor(text):return previous.enrich(text,_legacy_extract_info(text))
    rows=[{'text':text,'gold':name,'partition':'development'} for name,examples in GROUPS.items() for text in examples]
    rows += [{'text':text,'gold':name,'partition':'additional_stress'} for name,text in STRESS]
    result={'reference_provenance':'assistant-authored synthetic development challenge; native speaker/clinician review pending',
            'scope':'single primary concept retrieval (not full NER, assertion, span or clinical validation)',
            'records':len(rows),'approaches':{},'rows':rows}
    for name,extractor in [('existing_english',previous_extractor),('aligned_clinical_phrases',extract_info)]:
        latencies=[];correct=0
        for row in rows:
            start=time.perf_counter();found=concepts(extractor(row['text']));latencies.append((time.perf_counter()-start)*1000)
            expected=set() if row['gold']=='none' else {row['gold']}
            correct+=found==expected;row[name]=sorted(found)
        result['approaches'][name]={'exact_concept_set_matches':correct,'exact_concept_set_rate':correct/len(rows),
                                   'mean_extraction_ms':sum(latencies)/len(latencies),'runtime_model_download_mb':0}
    if transformer:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from huggingface_hub import model_info,hf_hub_download
        identifier='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        revision=model_info(identifier).sha
        start=time.perf_counter()
        # The ordinary Transformers import hit a Windows Application Control
        # rejection for regex._regex. Use the publisher's ONNX export with the
        # already working tokenizer, without changing security settings.
        download=lambda name:hf_hub_download(identifier,name,revision=revision,cache_dir=str(OUT/'research_cache'))
        model_path=download('onnx/model_quint8_avx2.onnx')
        tokenizer=Tokenizer.from_file(download('tokenizer.json'))
        tokenizer.enable_truncation(max_length=128);tokenizer.enable_padding(pad_id=1,pad_token='<pad>')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(model_path,options,providers=['CPUExecutionProvider'])
        def encode(texts):
            encodings=tokenizer.encode_batch(texts)
            values={'input_ids':np.array([e.ids for e in encodings],dtype=np.int64),
                    'attention_mask':np.array([e.attention_mask for e in encodings],dtype=np.int64),
                    'token_type_ids':np.array([e.type_ids for e in encodings],dtype=np.int64)}
            output=session.run(None,{i.name:values[i.name] for i in session.get_inputs()})[0]
            mask=values['attention_mask'][...,None];pooled=(output*mask).sum(axis=1)/mask.sum(axis=1)
            return pooled/np.linalg.norm(pooled,axis=1,keepdims=True)
        load_seconds=time.perf_counter()-start
        keys=list(PROTOTYPES);emb=encode(list(PROTOTYPES.values()))
        # Fixed exploratory cutoff, not selected against this fixture set.
        cutoff=.45;correct=0;latencies=[]
        for row in rows:
            start=time.perf_counter();v=encode([row['text']])[0];similarities=emb@v
            j=int(similarities.argmax());found={keys[j]} if float(similarities[j])>=cutoff else set()
            latencies.append((time.perf_counter()-start)*1000)
            expected=set() if row['gold']=='none' else {row['gold']};correct+=found==expected
            row['multilingual_encoder']=sorted(found);row['encoder_similarity']=float(similarities[j])
        result['approaches']['multilingual_encoder']={'model_id':identifier,'revision':revision,'cutoff':cutoff,
            'exact_concept_set_matches':correct,'exact_concept_set_rate':correct/len(rows),'mean_extraction_ms':sum(latencies)/len(latencies),
            'load_seconds_including_download':load_seconds,'runtime':'publisher ONNX quint8 avx2 export + tokenizers, CPU, 2 threads',
            'research_cache_mb':sum(p.stat().st_size for p in (OUT/'research_cache').rglob('*') if p.is_file())/1e6,
            'limitation':'Whole-sentence nearest-concept retrieval does not locate evidence spans or distinguish assertion/actor/time. Not a fine-tuned clinical NER model.'}
    for approach in result['approaches']:
        result['approaches'][approach]['by_partition']={}
        for partition in ('development','additional_stress'):
            subset=[row for row in rows if row['partition']==partition]
            correct=sum(row[approach]==([] if row['gold']=='none' else [row['gold']]) for row in subset)
            result['approaches'][approach]['by_partition'][partition]={'correct':correct,'records':len(subset),'rate':correct/len(subset)}
    result['stress_caveat']='Additional synthetic spelling/idiom stress added after initial implementation; not used to expand the vocabulary or tune the encoder. Still assistant-authored, not independent native-speaker data.'
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    return result

if __name__=='__main__':run('--no-transformer' not in sys.argv)

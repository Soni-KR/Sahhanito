"""Full-fact comparison on synthetic fixtures; no fitting or clinical gold claims."""
import copy
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent

def fixture(text, entities, split='development', critical=False):
    facts = []
    for kind, name, span, status, temporal, owner, attrs in entities:
        start = text.index(span)
        facts.append(dict(kind=kind, name=name, start=start, end=start+len(span),
                          status=status, temporality=temporal, experiencer=owner, **attrs))
    return dict(text=text, facts=facts, split=split, critical=critical)

def fact(name, span, status='confirmed', temporal='current', owner='patient', kind='symptom', **attrs):
    return kind, name, span, status, temporal, owner, attrs

# Written before the implementation/experiment. These are assistant-authored
# synthetic references, not independently annotated native Tunisian patient data.
DEVELOPMENT = [
 fixture("I've had chest pain for twenty minutes", [fact('chest','chest pain', duration_seconds=1200)]),
 fixture('9albi yodhrob bel zerba', [fact('palpitations','9albi yodhrob bel zerba')]),
 fixture('3andi 7aja ghriba fi batni', []),
 fixture('I am not sure whether I have chest pain, but I have a cough.', [fact('chest','chest pain','uncertain'),fact('cough','cough')]),
 fixture('My father has asthma. I have dizziness now.', [fact('asthma','asthma',owner='other:father',kind='condition'),fact('dizziness','dizziness')]),
 fixture('My headache resolved two hours ago.', [fact('headache','headache','negated','resolved',resolution_seconds=7200)]),
 fixture('My chest hurts when I walk.', [fact('chest','chest hurts',temporal='episodic')]),
 fixture('Je ressens une oppression thoracique maintenant.', [fact('chest','oppression thoracique')]),
 fixture('ما عنديش كحة اما توا عندي دوخة', [fact('cough','كحة','negated'),fact('dizziness','دوخة')]),
 fixture('I take amoxicillin 250 mg and have no rash.', [fact('amoxicillin','amoxicillin',kind='medication',medication_role='taking',dose_value=250),fact('rash','rash','negated')]),
 fixture('If I developed a seizure, what should I do?', [fact('seizure','seizure','hypothetical','hypothetical')]),
 fixture('I feel as though the room is spinning.', [fact('dizziness','the room is spinning')]),
]
FINAL = [
 fixture('عندي وجيعة في صدري من 15 دقيقة', [fact('chest','وجيعة في صدري',duration_seconds=900)]),
 fixture('ma3andich wji3a fi sadri ama taw 3andi doukha', [fact('chest','wji3a fi sadri','negated'),fact('dizziness','doukha')]),
 fixture('ما نجمش نتنفس', [fact('breath','ما نجمش نتنفس')],critical=True),
 fixture('ommi 3andha rbou; taw ena 3andi s5ana', [fact('asthma','rbou',owner='other:mother',kind='condition'),fact('fever','s5ana')]),
 fixture('My dad had a stroke last year; I now have a headache.', [fact('stroke','stroke',temporal='historical',owner='other:father',kind='condition'),fact('headache','headache')]),
 fixture('Could this be pneumonia? I have no fever.', [fact('pneumonia','pneumonia','uncertain',kind='condition'),fact('fever','fever','negated')]),
 fixture('My blurred vision started gradually yesterday.', [fact('blurred_vision','blurred vision',onset='gradual')]),
 fixture('I do not have nausea or vomiting.', [fact('nausea','nausea','negated'),fact('vomiting','vomiting','negated')]),
 fixture('3andi douleur fi sadri et une toux maintenant', [fact('chest','douleur fi sadri'),fact('cough','toux')]),
 fixture('I stopped warfarin yesterday; I feel dizzy now.', [fact('warfarin','warfarin',kind='medication',medication_role='discontinued'),fact('dizziness','dizzy')]),
 fixture('I cannot get enough air into my lungs.', [fact('breath','cannot get enough air')],critical=True),
 fixture('راسي يوجعني برشة', [fact('headache','راسي يوجعني',severity='intense')]),
 fixture('If my mother gets chest pain, should she call a doctor?', [fact('chest','chest pain','hypothetical','hypothetical','other:mother')]),
 fixture('I am allergic to penicillin.', [fact('penicillin','penicillin',kind='medication',medication_role='allergy')]),
 fixture('3andi 7aja fi krichi ma na3rafch chnowa', []),
 fixture('Hello, can you help?', []),
 fixture('I had weakness this morning, but it has gone away now.', [fact('weakness','weakness','negated','resolved')]),
 fixture('I have trouble catching my breath right now.', [fact('breath','trouble catching my breath')],critical=True),
 fixture('3andi diabetes ama I feel dizzy now', [fact('diabetes','diabetes',kind='condition'),fact('dizziness','dizzy')]),
 fixture('Mon père a de la fièvre. Je prends du paracétamol 500 mg.', [fact('fever','fièvre',owner='other:father'),fact('acetaminophen','paracétamol',kind='medication',medication_role='taking',dose_value=500)]),
 fixture('My oxygen saturation is 91%.', [fact('oxygen_saturation','oxygen saturation is 91%',kind='vital',value=91)]),
 fixture('I still have a cough that began thirty minutes ago.', [fact('cough','cough',onset_seconds=1800)]),
 fixture('سخانة؟ مش متأكد', [fact('fever','سخانة','uncertain')]),
 fixture('There is no change in my headache.', [fact('headache','headache')]),
]

def write_reference():
    path=OUT/'hybrid_reference.jsonl'
    rows=DEVELOPMENT+[dict(x,split='final_synthetic_challenge') for x in FINAL]
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def attrs(f):
    d=dict(f)
    for field, source, key in [('severity','severity','value'),('onset','onset','value'),
                             ('duration_seconds','duration','seconds'),('resolution_seconds','resolution_time','seconds_ago'),
                             ('onset_seconds','onset','seconds_ago'),('dose_value','dose','value')]:
        d[field]=(f.get(source) or {}).get(key) if isinstance(f.get(source),dict) else f.get(field)
    return d

def measure(rows, extractor):
    tp=fp=fn=full_tp=canonical=matched=unsupported=critical_tp=critical_n=0
    checks={};times=[];outputs=[]
    kinds={'symptom','condition','medication','vital','demographic'}
    for row in rows:
        began=time.perf_counter();info=extractor(row['text']);elapsed=(time.perf_counter()-began)*1000
        times.append(info.get('_measured_extraction_ms',elapsed))
        predicted=[f for f in info['facts'] if f['kind'] in kinds]
        gold=copy.deepcopy(row['facts']);used=set();exact=0
        for g in gold:
            pairs=[(i,p) for i,p in enumerate(predicted) if i not in used and p['kind']==g['kind'] and min(p['end'],g['end'])>max(p['start'],g['start'])]
            if not pairs:continue
            i,p=max(pairs,key=lambda v:int(v[1]['name']==g['name'])+int(v[1]['start']==g['start'] and v[1]['end']==g['end']))
            used.add(i);matched+=1;canonical+=p['name']==g['name']
            exact+=p['start']==g['start'] and p['end']==g['end']
            pp=attrs(p);correct=p['name']==g['name']
            for key in g.keys()-{'kind','name','start','end'}:
                checks.setdefault(key,[0,0]);checks[key][1]+=1;checks[key][0]+=pp.get(key)==g[key];correct &= pp.get(key)==g[key]
            full_tp+=correct
            if row['critical'] and p['name']==g['name'] and p['status']==g['status'] and p['temporality']==g['temporality'] and p['experiencer']==g['experiencer']:critical_tp+=1
        tp+=exact;fp+=len(predicted)-exact;fn+=len(gold)-exact;unsupported+=len(predicted)-len(used)
        if row['critical']:critical_n+=len(gold)
        outputs.append(dict(text=row['text'],expected=gold,predicted=predicted,semantic=info.get('semantic_metadata')))
    def prf(t,p,n):
        precision=t/(t+p) if t+p else 1.;recall=t/(t+n) if t+n else 1.
        return dict(precision=precision if t+p else None,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,tp=t,fp=p,fn=n)
    totalp=tp+fp;totalg=tp+fn
    return dict(exact_span_entities=prf(tp,fp,fn),complete_facts_overlap=prf(full_tp,totalp-full_tp,totalg-full_tp),
                canonical_accuracy_on_overlap=canonical/matched if matched else None,
                incorrect_or_unsupported_full_fact_rate=(totalp-full_tp)/totalp if totalp else None,
                attributes={k:dict(correct=v[0],matched=v[1],accuracy=v[0]/v[1]) for k,v in checks.items()},
                unsupported_fact_rate=unsupported/totalp if totalp else 0.,
                critical_symptom_recall=critical_tp/critical_n if critical_n else None,
                latency_ms=dict(median=statistics.median(times),maximum=max(times),mean=statistics.mean(times)),outputs=outputs)

def run(split):
    from triage_pipeline import extract_info,_english_extract_info
    from multilingual_nlp import extract,SemanticExtractor,merge_semantic
    semantic=SemanticExtractor.from_cache(OUT/'research_cache')
    rows=DEVELOPMENT if split=='development' else [dict(x,split='final_synthetic_challenge') for x in FINAL]
    # Phase-1 baseline is recovered from the verified GitHub ZIP, never from the
    # mutable export directory. Only source is extracted; no duplicate model.
    import subprocess
    import zipfile
    baseline=ROOT/'triage_artifacts/handoff/baseline_source';baseline.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'triage_artifacts/handoff/phase1.zip') as archive:
        for item in archive.namelist():
            if '/ai/' not in item or not item.endswith(('.py','.json')):continue
            relative=item.split('/ai/',1)[1]
            if relative.endswith('english_feature_reference.json') or relative.endswith('compatibility.json'):continue
            target=(baseline/relative).resolve()
            assert target.is_relative_to(baseline.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(item))
    code="import sys,json,time;from triage_pipeline import extract_info\nrows=[]\nfor t in json.loads(sys.stdin.read()):\n started=time.perf_counter();r=extract_info(t);r['_measured_extraction_ms']=(time.perf_counter()-started)*1000;rows.append(r)\nprint(json.dumps(rows))"
    frozen=json.loads(subprocess.check_output([sys.executable,'-c',code],input=json.dumps([r['text'] for r in rows]).encode(),cwd=baseline))
    previous={r['text']:f for r,f in zip(rows,frozen)}
    def transformer_only(text):
        info=dict(facts=[],patient_experiencer='patient',observations=[])
        return merge_semantic(info,text,semantic,inherited='patient',candidate_only=False,aligned=False)
    def hybrid(text):return extract(text,_english_extract_info,semantic=semantic)
    report=dict(provenance='Assistant-authored synthetic references; no native-speaker/clinician review; no independent clinical gold.',
                split=split,reference_sha256=write_reference(),model=semantic.metadata,approaches={})
    minilm=SpanEncoder()
    def embedding_facts(text):return merge_semantic(dict(facts=[],observations=[],patient_experiencer='patient'),text,minilm,candidate_only=False,aligned=False)
    for name,fn in [('phase1',lambda t:previous[t]),('improved_dictionary',extract_info),('gliner_fp32_native',transformer_only),('minilm_span_context',embedding_facts),('gliner_fp32_conservative_hybrid',hybrid)]:
        report['approaches'][name]=measure(rows,fn)
    (OUT/('hybrid_'+split+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='outputs'} for k,v in report['approaches'].items()},indent=2))

class SpanEncoder:
    """Research-only MiniLM span retrieval, evaluated through identical context.

    Explicitly not a trained NER head. Compare complete-fact outputs, not the
    earlier whole-query word retrieval accuracy. Fixed .65 similarity/.1 margin
    from development; these are not calibrated probabilities.
    """
    def __init__(self):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from multilingual_nlp import SEMANTIC_LABELS
        self.np=np;self.labels=list(SEMANTIC_LABELS);self.concepts=SEMANTIC_LABELS
        revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
        folder=OUT/'research_cache/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots'/revision
        self.tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'));self.tokenizer.enable_truncation(max_length=128);self.tokenizer.enable_padding()
        opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(folder/'onnx/model_quint8_avx2.onnx'),opt,providers=['CPUExecutionProvider'])
        self.prototypes=self.encode(self.labels)
        self.metadata=dict(model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',revision=revision,
                           threshold=.65,margin=.1,task='Span retrieval followed by identical clinical context; no trained NER head',score_is_calibrated=False)
    def encode(self,texts):
        np=self.np;encoded=self.tokenizer.encode_batch(texts)
        feeds=dict(input_ids=np.array([x.ids for x in encoded],dtype=np.int64),attention_mask=np.array([x.attention_mask for x in encoded],dtype=np.int64),
                   token_type_ids=np.array([x.type_ids for x in encoded],dtype=np.int64))
        feeds={k:v for k,v in feeds.items() if k in {x.name for x in self.session.get_inputs()}}
        hidden=self.session.run(None,feeds)[0];mask=feeds['attention_mask'][:,:,None]
        mean=(hidden*mask).sum(1)/mask.sum(1).clip(min=1);return mean/self.np.linalg.norm(mean,axis=1,keepdims=True).clip(min=1e-9)
    def propose(self,text):
        import re
        started=time.perf_counter();words=list(re.finditer(r'\w+(?:[-_]\w+)*|\S',text))[:64]
        spans=[(words[a].start(),words[b].end()) for a in range(len(words)) for b in range(a,min(a+5,len(words)))]
        if not spans:self.last_metadata={};return []
        vectors=self.encode([text[a:b] for a,b in spans]);similarity=vectors@self.prototypes.T
        order=self.np.argsort(similarity,axis=1);chosen=[]
        for i in self.np.argsort(-similarity.max(1)):
            best=int(order[i,-1]);score=float(similarity[i,best]);runner=float(similarity[i,order[i,-2]])
            if score<.65 or score-runner<.1:continue
            a,b=spans[i]
            if any(a<p['end'] and p['start']<b for p in chosen):continue
            kind,name=self.concepts[self.labels[best]];chosen.append(dict(kind=kind,name=name,start=a,end=b,text=text[a:b],score=score))
        self.last_metadata=dict(inference_ms=(time.perf_counter()-started)*1000,truncated=len(words)==64)
        return chosen

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='freeze':print(write_reference())
    else:run(sys.argv[1] if len(sys.argv)>1 else 'development')

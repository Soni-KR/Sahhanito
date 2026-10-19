"""Auditable Tunisian/French clinical phrase normalization with exact alignment.

This is a finite, unreviewed dialect vocabulary, not general translation. Unknown
text is retained. Arabic orthographic folding never changes evidence offsets.
"""
import copy
import re
import unicodedata

VERSION = 'tn_clinical_hybrid_3'
# Longest matching span wins; numbers inside Arabizi words are never transliterated.
PHRASES = [
    # Specific breathing complaints, not metaphorical "choking" or anxiety alone.
    (r'(?:i\s*)?(?:cannot|can.t)\s*get\s*enough\s*air(?:\s*into\s*my\s*lungs)?|(?:i\s*)?(?:have\s*)?trouble\s*catching\s*my\s*breath', 'difficulty breathing'),
    (r'(?:ma\s*n(?:aj[ae]m|ajem|ajjem)ch|manaj[ae]mch)\s*na[5kh]ou\s*(?:nef[ae]s|nf[ae]s)|ما\s*نجمش\s*ناخو\s*نفس', 'cannot breathe'),
    (r'(?:ma\s*)?n(?:aj[ae]m|ajjem)ch\s*n(?:et|at)naf[f]?[ae]s|مانجمش\s*نتنفس', 'cannot breathe'),
    (r'n(?:et|at)naf[f]?[ae]s\s*(?:b(?:e|i)?s3ouba|b\s*s3ouba)|نتنفس\s*(?:بصعوبة|بالصعوبة|بالصعيب)', 'difficulty breathing'),
    (r'(?:nefsi|nfasi|nafas[iy]?)\s*(?:9sir|ma9tou3|m9at3|na9es)|(?:نفسي|نفاسي)\s*(?:قصير|مقطوع|ناقص)', 'difficulty breathing'),
    (r'(?:el\s*)?(?:nef[ae]s|nf[ae]s)\s*ma\s*(?:y[eai]?kaf[iy]ch|y[eai]?kf[iy]ch)|النفس\s*ما\s*(?:يكفيش|يكفينيش)', 'difficulty breathing'),
    (r'(?:i\s*)?(?:can|am able to)\s*breathe\s*normally|n(?:et|at)naf[f]?[ae]s\s*3adi|نتنفس\s*عادي|je\s*respire\s*normalement', 'I have no difficulty breathing'),
    (r'tlahh[ae]th|نتلهث|نلهث', 'gasping'),
    (r'(?:ma\s*naj[ae]mch\s*nkammel\s*(?:jomla|klem[iy])\s*(?:mel|men)\s*(?:tanaffos|nef[ae]s))|ما\s*نجمش\s*نكمل\s*(?:جملة|كلامي)\s*من\s*(?:التنفس|النفس)', 'unable to get words out'),
    (r'ki\s*nemchi|كي\s*نمشي', 'when I walk'),
    (r'(?:kan|ken)\s*ki|كان\s*كي', 'only when'),
    (r'ma3adech|ما\s*عادش|ماعادش', 'no longer'),
    (r'batni|بطني|ventre', 'abdomen'),
    (r'chest\s*hurts', 'chest pain'),
    (r'(?:the\s*)?room\s*is\s*spinning', 'dizziness'),
    (r'oppression\s*thoracique', 'chest pressure'),
    (r'\bena\b|\bانا\b|\bje\b', 'I'),
    (r'\bparacetamol\b', 'paracetamol'),
    (r'وعندي', 'and I have'),
    (r'رقبتي\s*يابسة|ra9beti\s*yabsa', 'stiff neck'),
    (r'متخلط|met5[ae]l[ae]t|confus', 'confused'),
    (r'(?:وجيعة|وجع)\s*في\s*راسي|wji3a\s*fi\s*rasi', 'headache'),
    (r'بقلبي\s*يضرب\s*بالزربة|9albi\s*yodhrob\s*bel\s*zerba', 'palpitations'),
    (r'وجهي\s*مايل|wejhi\s*mayel', 'face drooping'),
    (r'تنميل|tn[ae]mil|engourdissement', 'numbness'),
    (r'ضعف|dh3of|faiblesse', 'weakness'),
    (r'طفح\s*جلدي|طفح|eruption\s*cutanee', 'rash'),
    (r'حك[ةه]|7akka|demangeaisons?', 'itching'),
    (r'ما\s*نجمش\s*نبلع|ma\s*najamch\s*n[ae]bl[ae]3', 'difficulty swallowing'),
    (r'ما\s*نجمش\s*نشوف|ma\s*najamch\s*nchouf', 'cannot see'),
    (r'ما\s*نجمش\s*نحكي|ma\s*najamch\s*n[ae]7ki', 'difficulty speaking'),
    (r'حلق[هي]\s*منفوخ|7al9i\s*manfou5', 'throat swelling'),
    (r'فمي\s*منفوخ|fommi\s*manfou5', 'mouth swelling'),
    (r'دم\s*في\s*البول|dam\s*fel\s*boul', 'blood in urine'),
    (r'جلطة|jalta|accident\s*vasculaire\s*cerebral', 'stroke'),
    (r'صرع|sar3|epilepsie', 'epilepsy'),
    (r'سرطان|sartan', 'cancer'),
    (r'حامل|7amel|enceinte', 'pregnant'),
    (r'قلق|9ala9|anxiete', 'anxiety'),
    (r'هلع|hala3|attaque\s*de\s*panique', 'panic attack'),
    (r'التهاب\s*السحايا|meningite', 'meningitis'),
    (r'التهاب\s*الرية|pneumonie', 'pneumonia'),
    (r'صداع\s*نصفي|ch9i9a', 'migraine'),
    (r'اسبرين|aspirine', 'aspirin'),
    (r'ايبوبروفين|ibuprofene', 'ibuprofen'),
    (r'اموكسيسيلين|amoxicilline', 'amoxicillin'),
    (r'ميتفورمين|metformine', 'metformin'),
    (r'اوميبرازول', 'omeprazole'),
    (r'فنتولين', 'Ventolin'),
    (r'اتورفاستاتين', 'atorvastatin'),
    (r'وارفارين', 'warfarin'),
    (r'ليفوثيروكسين', 'levothyroxine'),
    (r'نعرفش|ma\s*na3rafch|مانعرفش', 'not sure whether'),
    (r'ma\s*(?:y)?naj[ae]mch\s*(?:y|n)etnaff[ae]s|ما\s*[ين]جمش\s*[ين]تنفس', 'cannot breathe'),
    (r'(?:ma\s*)?n[ae]j[ae]mch\s*netnaff[ae]s', 'cannot breathe'),
    (r'(?:w[jg]a3|wji3a|douleur|وجيعة|وجع)\s*(?:fi|fel|في)\s*(?:sadr[iy]|صدري|الصدر)', 'chest pain'),
    (r'(?:sadr[iy]|صدري)\s*(?:you?j[a3]+ni|يوجعني)|douleur\s*(?:thoracique|a la poitrine)', 'chest pain'),
    (r'(?:rasi|راسي)\s*(?:you?j[a3]+ni|يوجعني)|w[jg]a3\s*(?:fi|fel)\s*rasi|mal\s*(?:a|à)\s*la\s*tete', 'headache'),
    (r'(?:s3ouba|sou3ouba|صعوبة)\s*(?:fi|fel|في)\s*(?:tanaff[ou]+s|التنفس)|difficulte\s*(?:a|à)\s*respirer', 'difficulty breathing'),
    (r'nefsi\s*(?:ma9tou3|m9at3)|نفسي\s*مقطوع|ضيق\s*(?:في\s*)?التنفس', 'difficulty breathing'),
    (r'وجه[هي]\s*مايل|wejhi\s*mayel', 'face drooping'),
    (r'يدي\s*اليسار\s*ضعيفة|yedi\s*lisar\s*d[h3]3ifa', 'left arm weakness'),
    (r'كلامي\s*متخلط|k[ae]l[ae]mi\s*met5[ae]l[ae]t|parole\s*pateuse', 'slurred speech'),
    (r'نزيف\s*قوي|nzif\s*9wi|saignement\s*abondant', 'heavy bleeding'),
    (r'فاقد\s*الوعي|f[ae]9[ae]d\s*el\s*wa3y|inconscient', 'unconscious'),
    (r'dou?kha|دوخة|etourdissements?', 'dizziness'),
    (r's[ou]k[ko]+r|سكري|السكري|diabete', 'diabetes'),
    (r'rbou|ربو|asthme', 'asthma'),
    (r'da8t|ضغط\s*الدم|hypertension\s*arterielle', 'hypertension'),
    (r's5ana|skhana|سخانة|حمى|fievre', 'fever'),
    (r'k[oou]+7a|كحة|toux', 'cough'),
    (r'nr[ao]jja3|nt9ayya|نتقيأ|vomissements?', 'vomiting'),
    (r'is[h7]al|اسهال|diarrhee', 'diarrhea'),
    (r'(?:wja3|وجع)\s*(?:fi|في)\s*(?:batni|بطني)|mal\s*au\s*ventre', 'abdominal pain'),
    (r'7asasiya\s*(?:men|l)|حساسية\s*(?:من|ل)|allergique\s*(?:a|à)', 'allergic to'),
    (r'na5ou|ناخو|je prends', 'I take'),
    (r'doliprane|بانادول|باراسيتامول', 'paracetamol'),
    (r'انسولين|insuline', 'insulin'),
    (r'بنسلين|penicilline', 'penicillin'),
    (r'ma\s*3andi?ch|ma3andi?ch|ماعنديش|معنديش|ما\s*عنديش|je n.ai pas(?: de)?', 'I have no'),
    (r'3and[iy]|عندي|j.ai', 'I have'),
    (r'3and[ou]+|عندو|il a', 'has'),
    (r'3andha|عندها|elle a', 'has'),
    (r'b[ou]+ya|بابا|بويا|mon pere', 'my father'),
    (r'ommi|[ou]mmi|امي|maman|ma mere', 'my mother'),
    (r'خويا|5ouya|mon frere', 'my brother'),
    (r'ختي|o5ti|ma soeur', 'my sister'),
    (r'mouch\s*met2[ae]ked|moch\s*met2[ae]ked|مش\s*متاكد|موش\s*متاكد|je ne suis pas sur', 'not sure whether'),
    (r'bal[ae]k|يمكن|peut.etre', 'maybe'),
    (r'ken\s*nwalli|اذا|if', 'if'),
    (r'3am\s*(?:elli\s*)?fet|العام\s*الفات|l.annee derniere', 'last year'),
    (r'9bal|قبل|auparavant', 'previously'),
    (r'taw[a]?|توا|الان|maintenant', 'now'),
    (r'faj2[ae]|فجاة|soudain(?:ement)?', 'suddenly'),
    (r'bchwaya\s*bchwaya|بشوية\s*بشوية|progressivement', 'gradually'),
    (r'b[ae]rch[ae]|برشة|forte?|intense', 'intense'),
    (r'chwaya|شوية|leger', 'mild'),
    (r'wfet|وفات|mchet|مشات|disparu', 'resolved'),
    (r'ama|لكن|اما|mais', 'but'),
    (r'men|من|depuis', 'for'),
    (r'sw[ae]y[ae]3|s[ae]3[ae]t?|ساعات|ساعة|heures?', 'hours'),
    (r'd9aye9|d9i9a|دقايق|دقيقة', 'minutes'),
    (r'ay[ae]m|nhar|ايام|يوم|jours?', 'days'),
    (r'j[ou]m3[ae]t?|اسابيع|اسبوع|semaines?', 'weeks'),
    (r'ch[h7]our|شهور|شهر|mois', 'months'),
    (r'3am|سنين|سنة|ans', 'years'),
    (r'3omri|عمري|j.ai l.age de', 'age'),
    (r'9addech\s*andek\s*mel\s*wji3a|قداش\s*عندك\s*ملي\s*بدا\s*وجع', 'how long have you had pain'),
    (r'saturation(?: en oxygene)?|نسبة\s*الاكسجين', 'SpO2'),
    (r'نبض|pouls', 'heart rate'),
    (r'tension', 'blood pressure'),
    (r'حرارة|temperature', 'temperature'),
    (r'سكر\s*الدم|glycemie', 'glucose'),
    (r'و', 'and'),
]
PATTERNS = [(re.compile(r'(?<!\w)(?:'+p+r')(?!\w)', re.I), value) for p,value in PHRASES]

def fold(text):
    chars=[];mapping=[]
    for i,ch in enumerate(text):
        if '\u064b'<=ch<='\u065f' or ch=='ـ':continue
        ch={'أ':'ا','إ':'ا','آ':'ا','ى':'ي','ؤ':'و','ئ':'ي','ة':'ة','٠':'0','١':'1','٢':'2','٣':'3','٤':'4','٥':'5','٦':'6','٧':'7','٨':'8','٩':'9'}.get(ch,ch)
        for c in unicodedata.normalize('NFD',ch.lower()):
            if not unicodedata.combining(c):chars.append(c);mapping.append((i,i+1))
    return ''.join(chars),mapping

def language_hint(text):
    arabic=bool(re.search(r'[\u0600-\u06ff]',text));latin=bool(re.search(r'[a-zA-Z]',text))
    if arabic:return 'mixed' if latin else 'tn_arabic'
    folded,_=fold(text)
    return 'tn_arabizi' if re.search(r'\b(?:3andi|ma3andich|rasi|barcha|ey|s3ouba|taw)\b|[a-z]\d[a-z]',folded) else 'en_or_fr'

def align(text):
    """Return a clinical pivot and one original span for each pivot character."""
    normalized,index=fold(text);matches=[]
    for pattern,value in PATTERNS:
        for m in pattern.finditer(normalized):matches.append((m.start(),m.end(),value))
    matches.sort(key=lambda m:(m[0],-(m[1]-m[0])));selected=[];end=-1
    for m in matches:
        if m[0]>=end:selected.append(m);end=m[1]
    pivot=[];alignment=[];changes=[];cursor=0
    # Unchanged letters use original spelling, so English extraction stays exact.
    for a,b,value in selected:
        for j in range(cursor,a):pivot.append(text[index[j][0]:index[j][1]]);alignment.append(index[j])
        source=(index[a][0],index[b-1][1]);pivot.extend(value);alignment.extend([source]*len(value))
        changes.append(dict(start=source[0],end=source[1],text=text[source[0]:source[1]],normalized=value,provenance=VERSION))
        cursor=b
    for j in range(cursor,len(index)):pivot.append(text[index[j][0]:index[j][1]]);alignment.append(index[j])
    return ''.join(pivot),alignment,changes

def remap_span(obj,text,alignment):
    if 'start' not in obj or 'end' not in obj:return
    a,b=obj['start'],obj['end']
    if b>a and a<len(alignment):a,b=alignment[a][0],alignment[min(b,len(alignment))-1][1]
    else:a=b=alignment[min(a,len(alignment)-1)][0] if alignment else 0
    obj.update(start=a,end=b,text=text[a:b])

def remap_fact(f,text,alignment,language,changes):
    remap_span(f,text,alignment)
    if 'context_start' in f:
        obj={'start':f['context_start'],'end':f['context_end']};remap_span(obj,text,alignment)
        f.update(context_start=obj['start'],context_end=obj['end'])
    for key in ('severity','duration','onset','dose','resolution_time','event_time'):
        if f.get(key):remap_span(f[key],text,alignment)
    for ev in f.get('evidence',[]):remap_span(ev,text,alignment)
    f.update(language=language,script='arabic' if re.search(r'[\u0600-\u06ff]',f['text']) else 'latin',
             extraction_provenance=VERSION if changes else 'existing_english_context',normalized_concept=f['kind']+'.'+f['name'])
    # Translation dictionaries have not been reviewed by Tunisian clinicians.
    if changes:f['extraction_confidence']=min(.85,f['extraction_confidence']);f['confidence_basis']+='; unreviewed_dialect_phrase_alignment'

def extract(text,english_extractor,inherited='patient',turn_id=None,semantic=None):
    pivot,alignment,changes=align(text)
    info=english_extractor(pivot,inherited,turn_id);info=copy.deepcopy(info)
    lang=language_hint(text)
    visited=set()
    for key in ('facts','observations','extraction_history'):
        for f in info.get(key,[]):
            if id(f) not in visited:remap_fact(f,text,alignment,lang,changes);visited.add(id(f))
    from clinical_state import facts_to_info
    result=facts_to_info(info['facts'],info['patient_experiencer'])
    result.update(observations=info.get('observations',[]),extraction_history=info.get('extraction_history',[]),
                  original_text=text,language_hint=lang,normalization=changes,nlp_version=VERSION)
    # Unknown text is retained for clarification, never assigned a diagnosis.
    meaningful=any(f['kind'] in ('symptom','condition','medication','vital') for f in info['facts'])
    result['uninterpreted_input']={'text':text,'reason':'No recognized clinical complaint; location/context alone is insufficient'} if not meaningful else None
    result['normalization_scope']='Finite recognized clinical phrases; remaining text retained, not assumed understood.'
    if semantic is not None:
        result=merge_semantic(result,text,semantic,inherited,turn_id)
    return result

def answer_text(text,field=None):
    """Canonicalize finite short answers; keep original answer separately in state."""
    folded,_=fold(text.strip())
    if re.fullmatch(r'(?:ma na3rafch|مانعرفش|ما نعرفش|مش متاكد|موش متاكد|je ne sais pas|i am not sure|i.m not sure|i cannot tell|i can.t tell)',folded):return 'Not sure'
    if re.fullmatch(r'(?:le|la|لا|non)(?:\s*[,،]?\s*(?:taw|tawa|توا|maintenant))?',folded):return 'No'
    if field and field.startswith('symptom.') and len(field.split('.'))==2:
        if re.fullmatch(r'(?:not at all|pas du tout|لا ابدا)',folded):return 'No'
        if field=='symptom.breath_functional_impairment' and re.fullmatch(r'(?:i can speak normally|i can finish a sentence|naj[ae]m na7ki 3adi|نجم نحكي عادي)',folded):return 'No'
    m=re.match(r'^(ey|اي|ايه|نعم|oui)(?=\W|$)',folded)
    if m:return 'Yes'+align(text[len(m.group()):])[0]
    return align(text)[0]

def answer_alignment(text,field=None):
    pivot,mapping,changes=align(text);canonical=answer_text(text,field)
    if canonical==pivot:return canonical,mapping,changes
    folded,_=fold(text.strip());m=re.match(r'^(ey|اي|ايه|نعم|oui)(?=\W|$)',folded)
    if m:
        offset=len(text)-len(text.lstrip());end=offset+len(m.group())
        _,tail_map,_=align(text[end:])
        return canonical,[(offset,end)]*3+[(a+end,b+end) for a,b in tail_map],changes
    return canonical,[(0,len(text))]*len(canonical),changes


# Optional zero-shot span proposals. No model downloads occur during a patient
# request. Dictionary assertions remain authoritative; unfamiliar model-only
# interpretations are explicitly uncertain and require a linked confirmation.
SEMANTIC_MODEL='onnx-community/gliner_multi-v2.1'
SEMANTIC_REVISION='6ddaeb9413b0e71ad8457da1aab378a165b24058'
SEMANTIC_FILES={'onnx/model.onnx':'57af7f3d8d72a3ca4e5619c247344b4b5ee362a5ebcf307c0634d67b65e721c0',
                'tokenizer.json':'914bd3c8fb7b525af9e23b60d0ec7b1248ddb2b99014efd9c02ebeb022f8cab7'}
SEMANTIC_LABELS={
    'chest pain':('symptom','chest'), 'difficulty breathing':('symptom','breath'),
    'headache':('symptom','headache'), 'dizziness':('symptom','dizziness'),
    'fever':('symptom','fever'), 'cough':('symptom','cough'),
    'vomiting':('symptom','vomiting'), 'nausea':('symptom','nausea'),
    'abdominal pain':('symptom','abdominal_pain'), 'palpitations':('symptom','palpitations'),
    'confusion':('symptom','confusion'), 'weakness':('symptom','weakness'),
    'numbness':('symptom','numbness'), 'rash':('symptom','rash'),
    'difficulty speaking':('symptom','speech'), 'seizure':('symptom','seizure'),
    'diabetes':('condition','diabetes'), 'asthma':('condition','asthma'),
    'hypertension':('condition','hypertension'), 'pneumonia':('condition','pneumonia'),
    'stroke':('condition','stroke'), 'meningitis':('condition','meningitis'),
}

class SemanticExtractor:
    """Pinned GLiNER span model, CPU ONNX; bounded inputs and no generative LLM.

    Prompt, first-subtoken word masks and width-12 spans follow upstream GLiNER.
    Label inventory is intentionally finite. Scores are uncalibrated NER scores.
    """
    def __init__(self,model_path,tokenizer_path):
        import time
        import onnxruntime as ort
        from tokenizers import Tokenizer
        started=time.perf_counter()
        self.tokenizer=Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=384)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model_path),options,providers=['CPUExecutionProvider'])
        from threading import RLock
        self.lock=RLock()
        from pathlib import Path
        self.metadata=dict(model=SEMANTIC_MODEL,revision=SEMANTIC_REVISION,upstream='urchade/gliner_multi-v2.1',
            artifact=Path(model_path).name,license='Apache-2.0 upstream',runtime='CPU FP32 ONNX' if Path(model_path).name=='model.onnx' else 'CPU int8 ONNX',load_seconds=time.perf_counter()-started,
            medical_finetuning=False,tunisian_validation=False,threshold=.5,score_is_calibrated=False,
            upstream_training='General NER; upstream reports synthetic Pile-Mistral teacher data')

    @classmethod
    def from_cache(cls,cache_dir):
        from pathlib import Path
        folder=Path(cache_dir)/('models--'+SEMANTIC_MODEL.replace('/','--'))/'snapshots'/SEMANTIC_REVISION
        model=folder/'onnx/model.onnx';tokenizer=folder/'tokenizer.json'
        if not model.is_file() or not tokenizer.is_file():
            raise FileNotFoundError('Optional semantic model not installed. Run multilingual_nlp.py --download-semantic explicitly.')
        for name,expected in SEMANTIC_FILES.items():
            if artifact_hash(folder/name)!=expected:raise ValueError('Semantic artifact checksum mismatch: '+name)
        return cls(model,tokenizer)

    def propose(self,text):
        import numpy as np
        import time
        started=time.perf_counter()
        words=list(re.finditer(r'\w+(?:[-_]\w+)*|\S',text))
        if not words:return []
        labels=list(SEMANTIC_LABELS);prompt=[]
        for label in labels:prompt.extend(['<<ENT>>',label])
        prompt.append('<<SEP>>');skip=len(prompt)
        encoded=self.tokenizer.encode(prompt+[m.group() for m in words],is_pretokenized=True)
        mask=[];previous=None
        for wid in encoded.word_ids:
            mask.append(wid-skip+1 if wid is not None and wid>=skip and wid!=previous else 0)
            previous=wid
        count=max(mask)
        if not count:return []
        spans=[(start,start+width) for start in range(count) for width in range(12)]
        feeds=dict(input_ids=np.array([encoded.ids],dtype=np.int64),attention_mask=np.array([encoded.attention_mask],dtype=np.int64),
            words_mask=np.array([mask],dtype=np.int64),text_lengths=np.array([[count]],dtype=np.int64),
            span_idx=np.array([spans],dtype=np.int64),span_mask=np.array([[b<count for a,b in spans]],dtype=bool))
        logits=self.session.run(['logits'],feeds)[0][0]
        scores=1/(1+np.exp(-np.clip(logits,-50,50)))
        found=[]
        for start,width,label in np.argwhere(scores>=.5):
            end=start+width
            if end>=count:continue
            label=labels[label];kind,name=SEMANTIC_LABELS[label]
            a,b=words[start].start(),words[end].end()
            found.append(dict(kind=kind,name=name,start=a,end=b,text=text[a:b],score=float(scores[start,width,labels.index(label)])))
        # Greedy non-overlap follows upstream flat-NER decoding. Retain rejected
        # overlaps in merge metadata rather than silently counting duplicates.
        chosen=[]
        for f in sorted(found,key=lambda f:-f['score']):
            if any(f['start']<p['end'] and p['start']<f['end'] for p in chosen):continue
            chosen.append(f)
        self.last_metadata=dict(inference_ms=(time.perf_counter()-started)*1000,
                                truncated=count<len(words),input_words=len(words),processed_words=count)
        return sorted(chosen,key=lambda f:f['start'])

def merge_semantic(info,text,semantic,inherited='patient',turn_id=None,candidate_only=True,aligned=True):
    """Merge model spans through the same clinical context, without rule inputs.

    Default model-only additions remain uncertain; they cannot supersede an
    existing literal interpretation. Negated/historical/hypothetical context
    remains explicit. Model confidence never establishes a clinical assertion.
    """
    from clinical_state import context,evidence,facts_to_info
    pivot,mapping,changes=align(text) if aligned else (text,[(i,i+1) for i in range(len(text))],[])
    result=copy.deepcopy(info);facts=result['facts'];rejected=[]
    from contextlib import nullcontext
    with getattr(semantic,'lock',nullcontext()):
        proposals=semantic.propose(pivot)
        request_metadata=copy.deepcopy(semantic.last_metadata)
    for proposal in proposals:
        f=copy.deepcopy(proposal);score=f.pop('score')
        ctx=context(pivot,f['start'],f['end'],f['kind'],inherited);ctx.pop('cues')
        f.update(ctx,concept=f['name'],concept_aliases=[f['name']],evidence=[evidence(pivot,f['start'],f['end'],turn_id=turn_id)],
                 extraction_confidence=score,confidence_basis='uncalibrated_general_NER_score; model_only_requires_patient_confirmation',
                 semantic_candidate=True,contextual_status=ctx['status'])
        for attr in ['severity','duration','onset','resolution_time','event_time']:
            if f.get(attr):f['evidence'].append(evidence(pivot,f[attr]['start'],f[attr]['end'],turn_id=turn_id,kind=attr))
        f['evidence'].append(evidence(pivot,f['context_start'],f['context_end'],turn_id=turn_id,kind='context'))
        remap_fact(f,text,mapping,language_hint(text),changes)
        f['extraction_provenance']=SEMANTIC_MODEL+'@'+SEMANTIC_REVISION
        overlap=[p for p in facts if min(f['end'],p['end'])>max(f['start'],p['start']) and p['kind'] in ('symptom','condition','medication','vital')]
        if overlap:
            rejected.append(dict(candidate=f,reason='Existing literal fact owns this span',conflicting=any(p['name']!=f['name'] for p in overlap)));continue
        if candidate_only and f['status']=='confirmed':f['status']='uncertain'
        facts.append(f);result.setdefault('observations',[]).append(copy.deepcopy(f))
    projected=facts_to_info(facts,result.get('patient_experiencer',inherited))
    result.update({k:v for k,v in projected.items() if k!='facts'})
    result['semantic_metadata']=dict(**semantic.metadata,**request_metadata,
                                   candidate_assertions_require_confirmation=candidate_only,rejected=rejected)
    return result

def download_semantic(cache_dir):
    """Explicit setup only, pinned immutable revision; one cache, no Git weights."""
    from huggingface_hub import hf_hub_download
    paths={name:hf_hub_download(SEMANTIC_MODEL,name,revision=SEMANTIC_REVISION,cache_dir=str(cache_dir)) for name in SEMANTIC_FILES}
    for name,path in paths.items():
        if artifact_hash(path)!=SEMANTIC_FILES[name]:raise ValueError('Downloaded semantic checksum mismatch: '+name)
    return paths

def artifact_hash(path):
    import hashlib
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()

if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--download-semantic',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=Path(__file__).resolve().parent/'semantic_cache')
    args=parser.parse_args()
    if args.download_semantic:
        for name in download_semantic(args.cache_dir):print('Installed '+name+' at pinned revision '+SEMANTIC_REVISION)

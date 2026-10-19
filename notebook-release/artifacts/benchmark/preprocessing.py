"""English question -> contextual clinical facts -> sparse features -> binary ML."""
import hashlib
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from labeling_binary.base import Labeler as MentionExtractor
from labeling_binary.context import normalize,clauses,classify

ROOT=Path(__file__).resolve().parent
ARTIFACTS=ROOT/'triage_artifacts'
CONFIG=ROOT/'labeling_binary/rules.v3.json'
NLP=MentionExtractor(CONFIG)
SEED=42
AGE=re.compile(r"\b(?:I'm |I am |I.m |my |a |an |is |aged )?(\d{1,3})\s*[- ]?years?[- ]old\b",re.I)
DURATION=re.compile(r'\b(?:for (?:the past |over |about )?|over the past |past |last )(\d+(?:\.\d+)?|one|two|three|four|five|six|several|few|many|a)\s+(minutes?|hours?|days?|weeks?|months?|years?)\b',re.I)
EXTRA=re.compile(r'\b(?:panic attacks?|anxiety|hypertension|HIV|asthma|COPD|dementia|osteoporosis|pregnancy|pregnant|migraine|kidney disease|immunosuppress\w*|chest|abdomen|tooth|teeth|jaw|left arm|right arm|neck|throat|tongue|head|back|knee)\b',re.I)
SEVERITY=re.compile(r'\b(?:mild|moderate|severe|extreme|unbearable|excruciating|intense)\b',re.I)

def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def case_id(text):return hashlib.sha256(re.sub(r'\s+',' ',normalize(text)).casefold().strip().encode()).hexdigest()
def extraction_signature():
    paths=[Path(__file__),CONFIG,ROOT/'labeling_binary/base.py',ROOT/'labeling_binary/context.py']
    return hashlib.sha256(''.join(file_hash(p) for p in paths).encode()).hexdigest()

def extract_info(question):
    """Pure text boundary: no labels, answers, rules fired, or dataset metadata."""
    if not isinstance(question,str):raise TypeError('Pass only the patient question string.')
    if not question.strip():raise ValueError('The patient question cannot be empty.')
    text=normalize(question)
    mentions=NLP.mentions(question)  # mentions(), never Labeler.label()
    spans=clauses(text)
    for match in EXTRA.finditer(text):
        span=next(((a,b) for a,b in spans if a<=match.start()<b),(0,len(text)))
        context=classify(text,match.start(),match.end(),'clinical_term',span)
        mentions.append(dict(concept='term_'+re.sub(r'\W+','_',match.group().lower()),text=question[match.start():match.end()],start=match.start(),end=match.end(),**context))
    for m in mentions:
        local=text[max(m['context_start'],m['start']-45):min(m['context_end'],m['end']+35)]
        m['severity_words']=sorted(set(x.group().lower() for x in SEVERITY.finditer(local)))
        assert question[m['start']:m['end']]==m['text']
    match=AGE.search(text[:180]);age=int(match.group(1)) if match and 0<int(match.group(1))<=120 else None
    first=text[:180]
    sex='female' if re.search(r'\b(?:female|woman|girl|mother|daughter|wife)\b',first,re.I) else 'male' if re.search(r'\b(?:male|man|boy|father|son|husband)\b',first,re.I) else 'not_reported'
    durations=[dict(value=m.group(1),unit=m.group(2).lower(),text=question[m.start():m.end()],start=m.start(),end=m.end()) for m in DURATION.finditer(text)]
    vitals={};vital_evidence=[]
    patterns={'oxygen_saturation':r'\b(?:oxygen saturation|SpO2|O2 sat(?:uration)?)\s*(?:was|is|of|at|:|=)?\s*(\d{2,3}(?:\.\d+)?)\s*%',
              'heart_rate':r'\b(?:heart rate|pulse)\s*(?:was|is|of|at|:|=)?\s*(\d{2,3})\b',
              'blood_pressure':r'\b(?:blood pressure|BP)\s*(?:was|is|of|at|:|=)?\s*(\d{2,3})\s*/\s*(\d{2,3})',
              'temperature':r'\b(?:temperature|fever)(?:\s+of|\s+is|\s+was|\s+up to|\s*:)?\s*(\d{2,3}(?:\.\d+)?)\s*[°\ufffd ]*([CF])\b'}
    for name,pattern in patterns.items():
        m=re.search(pattern,text,re.I)
        if not m:continue
        if name=='blood_pressure':vitals.update(systolic=float(m.group(1)),diastolic=float(m.group(2)))
        elif name=='temperature':
            value=float(m.group(1));vitals['temperature_c']=(value-32)*5/9 if m.group(2).upper()=='F' else value
        else:vitals[name]=float(m.group(1))
        vital_evidence.append(dict(field=name,text=question[m.start():m.end()],start=m.start(),end=m.end()))
    active=[m for m in mentions if m['assertion']=='present' and m['temporality']=='current']
    missing=[]
    if age is None:missing.append('age')
    if sex=='not_reported':missing.append('reported_sex')
    if not durations and not any(m['acute'] or m['recent'] for m in active):missing.append('onset_or_duration')
    if not any(m['severity_words'] for m in active):missing.append('symptom_severity')
    if not vitals:missing.append('vital_signs')
    return dict(age=age,reported_sex=sex,vitals=vitals,vital_evidence=vital_evidence,durations=durations,mentions=mentions,missing_information=missing)

def to_features(info):
    """Only extracted information enters ML; no raw question or rule prediction."""
    age=info['age'];features={'reported_sex':info['reported_sex'],'age_missing':float(age is None),'age_scaled':0.0 if age is None else age/100,
       'age_group':'not_reported' if age is None else 'child' if age<18 else 'older_adult' if age>=65 else 'adult'}
    for m in info['mentions']:
        concept=m['concept'];status=m['assertion'];temporal=m['temporality']
        features[f'mention.{status}.{concept}']=1.0
        features[f'time.{temporal}.{concept}']=1.0
        if status=='present' and temporal=='current':
            features[f'current.{concept}']=1.0
            if m['acute']:features[f'acute.{concept}']=1.0
            if m['recent']:features[f'recent.{concept}']=1.0
            for s in m['severity_words']:features[f'severity.{s}.{concept}']=1.0
        if temporal=='episodic':features['episodic_presentation']=1.0
        if m['experiencer'].startswith('other:'):features['third_person_report']=1.0
    for d in info['durations']:features['duration_unit.'+d['unit'].rstrip('s')]=1.0
    scales={'oxygen_saturation':100,'heart_rate':200,'systolic':200,'diastolic':120,'temperature_c':40}
    for name,scale in scales.items():
        features[name+'_missing']=float(name not in info['vitals'])
        if name in info['vitals']:features[name+'_scaled']=info['vitals'][name]/scale
    for name in info['missing_information']:features['missing.'+name]=1.0
    return features

def prepare_data(csv_path=ROOT/'train.csv'):
    frame=pd.read_csv(csv_path,usecols=['question_en','urgency'],keep_default_na=False)
    frame['emergency_label']=frame.urgency.map({'non_emergency':0,'emergency':1})
    if frame.emergency_label.isna().any():raise ValueError('Expected binary urgency values only.')
    frame['source_record_index']=np.arange(len(frame));frame['case_id']=frame.question_en.map(case_id)
    assert frame.groupby('case_id').emergency_label.nunique().max()==1
    frame.emergency_label=frame.emergency_label.astype('int8')
    ARTIFACTS.mkdir(exist_ok=True);signature=extraction_signature();source_hash=file_hash(csv_path)
    cache=ARTIFACTS/'nlp_features.parquet';manifest_path=ARTIFACTS/'feature_cache.json'
    if cache.exists() and manifest_path.exists() and json.loads(manifest_path.read_text())=={'source_hash':source_hash,'nlp_signature':signature}:
        data=pd.read_parquet(cache);assert data.case_id.tolist()==frame.case_id.tolist()
        frame['features']=data.features_json.map(json.loads);print('Loaded cached NLP features.',flush=True)
    else:
        memo={};records=[];features=[]
        for row in frame.itertuples():
            if row.question_en not in memo:
                info=extract_info(row.question_en);memo[row.question_en]=(info,to_features(info))
            info,feature=memo[row.question_en];features.append(feature)
            records.append(dict(source_record_index=row.source_record_index,case_id=row.case_id,features_json=json.dumps(feature),extracted_info_json=json.dumps(info,ensure_ascii=False)))
            if (row.source_record_index+1)%5000==0:print(f'NLP extracted {row.source_record_index+1}/{len(frame)}',flush=True)
        frame['features']=features;pd.DataFrame(records).to_parquet(cache,index=False)
        manifest_path.write_text(json.dumps({'source_hash':source_hash,'nlp_signature':signature}),encoding='utf-8')
    # Fit/evaluate once per normalized question. Every duplicate inherits its group's split.
    unique=frame.drop_duplicates('case_id').copy()
    train,remainder=train_test_split(unique,test_size=.30,stratify=unique.emergency_label,random_state=SEED)
    validation,test=train_test_split(remainder,test_size=.50,stratify=remainder.emergency_label,random_state=SEED)
    groups=[set(part.case_id) for part in [train,validation,test]]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    mapping={g:name for name,part in [('train',train),('validation',validation),('test',test)] for g in part.case_id}
    frame['split']=frame.case_id.map(mapping)
    frame[['source_record_index','case_id','emergency_label','split']].to_csv(ARTIFACTS/'data_splits.csv',index=False)
    return frame,train,validation,test

def evaluate(model,frame):
    x=frame.features.tolist();y=frame.emergency_label.to_numpy();prediction=model.predict(x);score=model.predict_proba(x)[:,1]
    return dict(classification_report=classification_report(y,prediction,labels=[0,1],target_names=['non_emergency','emergency'],output_dict=True,zero_division=0),
        confusion_matrix=confusion_matrix(y,prediction,labels=[0,1]).tolist(),roc_auc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)))

def train_models(train,validation):
    candidates={'logistic_regression':LogisticRegression(C=1,max_iter=1000,class_weight='balanced',solver='liblinear',random_state=SEED),
       'extra_trees':ExtraTreesClassifier(n_estimators=160,max_depth=24,min_samples_leaf=2,class_weight='balanced',n_jobs=2,random_state=SEED)}
    models={};rows=[]
    for name,classifier in candidates.items():
        model=Pipeline([('vectorizer',DictVectorizer(sparse=True)),('classifier',classifier)])
        model.fit(train.features.tolist(),train.emergency_label)
        scores=evaluate(model,validation);report=scores['classification_report'];models[name]=model
        rows.append(dict(model=name,macro_f1=report['macro avg']['f1-score'],emergency_precision=report['emergency']['precision'],emergency_recall=report['emergency']['recall'],accuracy=report['accuracy']))
    comparison=pd.DataFrame(rows).sort_values(['macro_f1','emergency_recall'],ascending=False).reset_index(drop=True)
    return models,comparison

def save_model(model,metrics):
    bundle=dict(model=model,feature_signature=extraction_signature(),label_mapping={0:'non_emergency',1:'emergency'},metrics=metrics)
    joblib.dump(bundle,ARTIFACTS/'triage_model.joblib',compress=3)
    (ARTIFACTS/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return bundle

def predict_case(question,bundle=None):
    if bundle is None:bundle=joblib.load(ARTIFACTS/'triage_model.joblib')
    if bundle['feature_signature']!=extraction_signature():raise ValueError('NLP code changed; regenerate features and retrain before inference.')
    info=extract_info(question);features=to_features(info);model=bundle['model']
    label=int(model.predict([features])[0]);score=float(model.predict_proba([features])[0,1])
    return dict(emergency_label=label,prediction=bundle['label_mapping'][label],model_score_emergency=score,score_is_clinically_calibrated=False,
        extracted_info=info,adaptive_questions_enabled=False)

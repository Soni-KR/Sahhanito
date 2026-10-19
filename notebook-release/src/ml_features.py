"""Versioned ML projection; rich clinical facts are never rewritten for label agreement."""
from clinical_state import FACT_KINDS
FEATURE_SCHEMA = 'clinical_projection_v3'

def shared_features(info):
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

def detailed_features(info):
    """Explicit fact whitelist: decision-policy/safety-engine outputs are ignored."""
    features={};target=info.get('patient_experiencer','patient')
    for f in info.get('facts',[]):
        if f.get('kind') not in FACT_KINDS:continue
        if f['experiencer']!=target:continue
        key=f['kind']+'.'+f['name'];features['fact.'+f['status']+'.'+key]=1.
        features['fact_time.'+f['temporality']+'.'+key]=1.
        if f.get('onset'):features['onset.'+f['onset']['value']+'.'+key]=1.
        if f['kind']=='medication':features['medication_role.'+f.get('medication_role','mentioned')+'.'+f['name']]=1.
        d=f.get('duration')
        if d and d.get('seconds') is not None:features['duration_log.'+key]=min(20.,__import__('math').log1p(d['seconds']))/20
        if f['kind']=='vital' and f['status']=='confirmed' and f['temporality']=='current' and isinstance(f.get('value'),(int,float)):
            if f['name']=='respiratory_rate':features['respiratory_rate_scaled']=f['value']/50
            if f['name']=='glucose':features['glucose_mgdl_scaled']=(f['value']*18 if f['unit'].lower()=='mmol/l' else f['value'])/500
    return features

def build_features(info,profile='repaired'):
    if profile not in ('shared_only','repaired'):raise ValueError('Unknown feature projection.')
    features=shared_features(info)
    features.update(detailed_features(info))
    if profile=='shared_only':return features
    observations=info.get('observations',info.get('facts',[]))
    for fact in observations:
        if fact.get('kind') not in FACT_KINDS:continue
        # Observed topic is not a current clinical assertion. It is useful even
        # when the requester and the patient are different people.
        if fact['kind']=='context':
            for concept in fact.get('concept_aliases',[fact['concept']]):
                features['topic.'+concept]=1.
                features['topic_status.'+fact['status']+'.'+concept]=1.
        if fact['experiencer'].startswith('other:'):features['third_person_report']=1.
        # Preserve occurrence context lost by latest-state deduplication, without
        # promoting an old/negated/uncertain report to an active patient symptom.
        for concept in fact.get('concept_aliases',[fact['concept']]):
            key='observed.'+fact['status']+'.'+fact['temporality']+'.'+concept
            features[key]=1.
    return features

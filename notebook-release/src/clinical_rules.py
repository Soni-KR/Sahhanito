"""Independent, unvalidated clinical screening prototype. No ML inputs/outputs.

These are selected guidance-derived screens, not a complete implementation of
WHO IITT and not a remote diagnosis. Absence of a flag never establishes safety.
"""
import copy

VERSION = 'clinical_screens_1'
WHO = 'https://www.who.int/tools/triage'
WHO_ADULT = 'https://cdn.who.int/media/docs/default-source/integrated-health-services-(ihs)/csy/iitt/iitt_adult.pdf?sfvrsn=b2a91431_1'
CHEST = 'https://www.nhs.uk/symptoms/chest-pain/'
BREATH = 'https://www.nhs.uk/symptoms/shortness-of-breath/'
STROKE = 'https://www.nhs.uk/conditions/stroke/symptoms/'
SAMU = 'https://santetunisie.rns.tn/fr/sante-en-tunisie/medecine-d-urgence/les-services-d-aide-medicale-urgente-samu'

RULES = {
    'chest_dyspnea': ('critical', CHEST, 'Current chest pain with shortness of breath.'),
    'severe_breathing': ('critical', BREATH, 'Severe breathing difficulty or explicit inability to breathe.'),
    'stroke_signs': ('critical', STROKE, 'Sudden focal weakness, face droop or speech difficulty; recent resolved signs also matter.'),
    'unresponsive': ('critical', WHO_ADULT, 'Currently unconscious or unresponsive, not a recovered faint.'),
    'heavy_bleeding': ('critical', WHO_ADULT, 'Current heavy/uncontrolled bleeding.'),
    'neurologic_infection_cluster': ('critical', WHO_ADULT, 'Selected subset, age >=12: two of altered mental state, stiff neck, fever or headache. Hypothermia is not included in this prototype cluster.'),
    'extreme_heart_rate': ('critical', WHO_ADULT, 'Age >=12: current heart rate <50 or >150 beats/minute.'),
    'high_risk_vitals': ('urgent_review', WHO_ADULT, 'Age >=12: HR <60/>130; RR <10/>30; temperature <36/>39 C; SpO2 <92%. Prompt clinician review, not automatically a red criterion.'),
}

def evaluate_rules(state):
    target=state.get('patient_experiencer','patient')
    facts=[f for f in state.get('facts',[]) if f.get('experiencer')==target]
    def find(names,kind='symptom'):
        return [f for f in facts if f['kind']==kind and (f['name'] in names or set(f.get('concept_aliases',[]))&set(names))]
    def present(names):
        matches=find(names)
        active=[f for f in matches if f['status']=='confirmed' and f['temporality']=='current']
        if active:return True,active,[]
        negatives=[f for f in matches if f['status']=='negated' and f['temporality']=='current']
        if negatives:return False,negatives,[]
        return None,matches,['symptom.'+names[0]+'.current_status']
    def value(name,kind='vital'):
        matches=find([name],kind)
        f=next((f for f in reversed(matches) if f['status']=='confirmed' and f['temporality']=='current'),None)
        return (f.get('value'),[f],[]) if f else (None,matches,[kind+'.'+name])
    def conjunction(*parts):
        status=False if any(p[0] is False for p in parts) else True if all(p[0] is True for p in parts) else None
        return status,[f for p in parts for f in p[1]],sorted(set(x for p in parts for x in p[2]))
    def any_of(parts):
        positives=[p for p in parts if p[0] is True]
        if positives:return True,[f for p in positives for f in p[1]],[]
        return (False if parts and all(p[0] is False for p in parts) else None,
                [f for p in parts for f in p[1]],sorted(set(x for p in parts for x in p[2])))
    results=[]
    def emit(rule,part):
        truth,support,missing=part;priority,source,reason=RULES[rule]
        results.append(dict(rule_id=rule,status='triggered' if truth is True else 'not_triggered' if truth is False else 'insufficient_information',
                            priority=priority,justification=reason,supporting_facts=copy.deepcopy(support),
                            evidence=[copy.deepcopy(e) for f in support for e in f.get('evidence',[])],
                            missing_required_information=missing if truth is None else [],source=source,
                            validation_status='unvalidated_prototype',patient_experiencer=target))
    emit('chest_dyspnea',conjunction(present(['chest']),present(['breath','breath_severe'])))
    breathing=present(['breath','breath_severe']);severe=[]
    for f in breathing[1]:
        if f['status']!='confirmed' or f['temporality']!='current':continue
        # "barcha/intense" is an intensity expression, not evidence that words
        # cannot be spoken. Only explicit severe/functional evidence is used here.
        word=(f.get('severity') or {}).get('value')
        if f['name']=='cannot_breathe' or 'breath_severe' in f.get('concept_aliases',[]) or word in ('severe','extreme','unbearable','excruciating'):
            severe.append(f)
    for f in find(['breath_functional_impairment']):
        if f['status']=='confirmed' and f['temporality']=='current' and (breathing[0] is True or f['text'].lower() in ('gasping','choking')):severe.append(f)
    emit('severe_breathing',(True,severe,[]) if severe else (False,breathing[1],[]) if breathing[0] is False else (None,breathing[1],['symptom.breath.severity_or_functional_impairment']))
    focal=find(['one_side','face_droop','face_drooping','speech'])
    focal.extend(f for f in find(['weakness','numbness']) if any(x in f.get('text','').lower() for x in ('arm','one side','left','right'))
                 or any(a['kind']=='anatomy' and a['name'] in ('left_arm','right_arm') and a['experiencer']==target
                        and a['status']=='confirmed' and a['temporality']=='current'
                        and a['context_start']==f['context_start'] and a['context_end']==f['context_end']
                        and {e.get('turn_id') for e in a['evidence']} & {e.get('turn_id') for e in f['evidence']} for a in facts))
    stroke=[];unclear=[]
    for f in focal:
        recent_resolution=(f.get('resolution_time') or {}).get('seconds_ago')
        recent_event=(f.get('event_time') or {}).get('seconds_ago')
        sudden=(f.get('onset') or {}).get('value')=='sudden'
        onset_ago=(f.get('onset') or {}).get('seconds_ago')
        if (f['status']=='confirmed' and f['temporality']=='current' and (sudden or f['acute'] or onset_ago is not None and onset_ago<=86400)) or (
            f['temporality']=='resolved' and (recent_resolution is not None and recent_resolution<=86400 or (f.get('resolution_time') or {}).get('within_last_24_hours'))) or (
            f['temporality']=='historical' and f['status']=='confirmed' and (recent_event is not None and recent_event<=86400 or (f.get('event_time') or {}).get('within_last_24_hours'))):stroke.append(f)
        elif f['temporality'] not in ('historical','hypothetical') and f['status']!='negated':unclear.append(f)
    emit('stroke_signs',(True,stroke,[]) if stroke else (None,unclear,['symptom.focal_neurology','symptom.focal_neurology.onset_or_event_time']))
    emit('unresponsive',present(['unresponsive','unconscious']))
    emit('heavy_bleeding',present(['bleed_severe','heavy_bleeding','uncontrolled_bleeding']))
    age,age_f,age_missing=value('age','demographic')
    adult=(None,age_f,age_missing) if age is None else (age>=12,age_f,[])
    cluster=[present([name]) for name in ('confusion','neck_stiffness','fever','headache')]
    confirmed=[p for p in cluster if p[0] is True]
    count_part=(True,[f for p in confirmed for f in p[1]],[]) if len(confirmed)>=2 else (
        False if sum(p[0] is not False for p in cluster)<2 else None,
        [f for p in cluster for f in p[1]],sorted(set(x for p in cluster for x in p[2])))
    emit('neurologic_infection_cluster',conjunction(adult,count_part))
    hr,hr_f,hr_missing=value('heart_rate')
    extreme=(None,hr_f,hr_missing) if hr is None else (hr<50 or hr>150,hr_f,[])
    emit('extreme_heart_rate',conjunction(adult,extreme))
    risk=[]
    for name,low,high in [('heart_rate',60,130),('respiratory_rate',10,30),('temperature',36,39),('oxygen_saturation',92,None)]:
        v,fs,missing=value(name)
        risk.append((None,fs,missing) if v is None else (v<low or high is not None and v>high,fs,[]))
    emit('high_risk_vitals',conjunction(adult,any_of(risk)))
    triggered=[r for r in results if r['status']=='triggered']
    critical=any(r['priority']=='critical' for r in triggered)
    return dict(rule_version=VERSION,critical_flags_detected=critical,triggered_rules=triggered,rule_results=results,
                missing_information=sorted(set(x for r in results for x in r['missing_required_information'])),
                no_trigger_does_not_establish_safety=True,validation_status='unvalidated_prototype',
                emergency_guidance=None if not critical else dict(
                    en='Seek emergency help now. In Tunisia call SAMU 190. Do not drive yourself; follow the emergency operator’s instructions.',
                    tn_arabizi='Otlob mosa3da mosta3jla taw. Fi Tounes, 3ayet lel SAMU 190. Ma tsou9ch wa7dek; ettaba3 ta3limet el operateur.',
                    tn_arabic='اطلب مساعدة استعجالية توا. في تونس عيط للسامو 190. ما تسوقش وحدك واتبع تعليمات فريق الإسعاف.',
                    local_number_source=SAMU,guidance_source=STROKE))

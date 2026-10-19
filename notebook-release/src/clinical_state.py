"""Deterministic clinical context and conversation state; no decision rules or LLMs.

Confidence values are explainable extraction heuristics, not medical probabilities.
Offsets refer to the named source (answer or question) in the identified turn.
"""
import copy
import re
from labeling_binary.context import normalize

VERSION = '3.0'
FACT_KINDS = {'symptom','condition','medication','vital','demographic','anatomy','context'}
LEXICON = {
    'symptom': {
        'breath': r'shortness of breath|short of breath|cannot breathe|can.t breathe|unable to breathe|dyspn[o]?ea|(?-i:SOB)|breathless(?:ness)?|difficulty breathing',
        'breath_functional_impairment': r'gasping|choking|(?:unable|not able) to get words out|cannot get words out',
        'chest': r'chest (?:pain|pressure|tightness)|angina',
        'headache': r'headaches?|cephalgia', 'confusion': r'confusion|confused|disoriented|altered mental status',
        'fever': r'fever|febrile|pyrexia', 'cough': r'cough(?:ing)?', 'nausea': r'nausea|nauseous',
        'vomiting': r'vomit(?:ing|ed)?|throwing up', 'diarrhea': r'diarrh(?:oea|ea)',
        'dizziness': r'dizz(?:y|iness)|vertigo', 'weakness': r'weakness|weak', 'numbness': r'numbness|numb',
        'palpitations': r'palpitations|racing heart', 'rash': r'rash|hives|urticaria', 'wheeze': r'wheez(?:e|ing)',
        'seizure': r'seizures?|convulsions?', 'fatigue': r'fatigue|exhaustion', 'syncope': r'syncope|fainted|passed out',
        'dysuria': r'dysuria|burning (?:when|with) urination', 'hematuria': r'hematuria|blood in (?:my |the )?urine',
        'hemoptysis': r'hemoptysis|coughing up blood', 'edema': r'edema|oedema|swelling',
        'vision_loss': r'vision loss|loss of vision|cannot see', 'neck_stiffness': r'neck stiffness|stiff neck',
        'melena': r'melena|black tarry stools?', 'abdominal_pain': r'abdominal pain|stomach ache|belly pain',
        'photophobia': r'photophobia|light sensitivity|sensitivity to light',
        'speech': r'slurred speech|difficulty speaking|speech difficulty',
        'dysphagia': r'dysphagia|difficulty swallowing|trouble swallowing',
    },
    'condition': {
        'asthma': r'asthma', 'copd': r'COPD|chronic obstructive pulmonary disease',
        'diabetes': r'diabetes|diabetic', 'hypertension': r'hypertension|high blood pressure',
        'stroke': r'stroke|CVA|cerebrovascular accident', 'heart_attack': r'heart attack|myocardial infarction',
        'pneumonia': r'pneumonia', 'meningitis': r'meningitis', 'sepsis': r'sepsis|septicemia',
        'anaphylaxis': r'anaphylaxis', 'migraine': r'migraines?', 'epilepsy': r'epilepsy', 'cancer': r'cancer',
        'kidney_disease': r'kidney disease|renal failure|CKD', 'heart_failure': r'heart failure|CHF',
        'atrial_fibrillation': r'atrial fibrillation|AFib', 'pregnancy': r'pregnancy|pregnant',
        'dementia': r'dementia', 'panic_attack': r'panic attacks?', 'anxiety': r'anxiety',
        'hiv': r'HIV', 'dvt': r'DVT|deep vein thrombosis', 'pulmonary_embolism': r'pulmonary embolism',
    },
    'medication': {
        'acetaminophen': r'acetaminophen|paracetamol|Tylenol', 'ibuprofen': r'ibuprofen|Advil|Nurofen',
        'aspirin': r'aspirin', 'naproxen': r'naproxen', 'amoxicillin': r'amoxicillin', 'penicillin': r'penicillin',
        'azithromycin': r'azithromycin', 'insulin': r'insulin', 'metformin': r'metformin',
        'warfarin': r'warfarin|Coumadin', 'apixaban': r'apixaban|Eliquis', 'rivaroxaban': r'rivaroxaban|Xarelto',
        'heparin': r'heparin', 'clopidogrel': r'clopidogrel|Plavix', 'prednisone': r'prednisone|prednisolone',
        'albuterol': r'albuterol|salbutamol|Ventolin', 'epinephrine': r'epinephrine|adrenaline|EpiPen',
        'nitroglycerin': r'nitroglycerin|GTN', 'lisinopril': r'lisinopril', 'amlodipine': r'amlodipine',
        'atorvastatin': r'atorvastatin', 'omeprazole': r'omeprazole', 'sertraline': r'sertraline',
        'fluoxetine': r'fluoxetine', 'lorazepam': r'lorazepam', 'diazepam': r'diazepam',
        'morphine': r'morphine', 'codeine': r'codeine', 'tramadol': r'tramadol', 'levothyroxine': r'levothyroxine',
        'doxycycline': r'doxycycline', 'amitriptyline': r'amitriptyline',
    },
}
PATTERNS = [(kind, name, re.compile(r'\b(?:'+pattern+r')\b', re.I))
            for kind, values in LEXICON.items() for name, pattern in values.items()]
BOUNDARY = re.compile(r'(?<!\d)\.(?!\d)|[;!?\n]|\b(?:but|however|although|except)\b', re.I)
ACTOR = re.compile(r"\b(?:my (?:\d+[- ]years?[- ]old )?(?:mother|father|son|daughter|child|baby|husband|wife|brother|sister|friend|partner)|I|he|she|they|the patient)\b", re.I)
NEGATION = re.compile(r"\b(?:no(?: evidence of)?|without|denies|denied|deny|never|not|don't|doesn't|do not|does not|negative for|free of)\b", re.I)
UNCERTAIN = re.compile(r'\b(?:maybe|possibly|possible|suspected|suspect|not sure|unsure|might|may have|cannot rule out|can.t rule out|rule out|could be|worried about|afraid of|fear(?: of)?)\b', re.I)
HYPOTHETICAL = re.compile(r'\b(?:if|what if|could .{0,25} cause|risk of|prevent|would|what are (?:the )?symptoms|article says|read that)\b', re.I)
SEVERITY = re.compile(r'\b(?:mild|moderate|severe|extreme|unbearable|excruciating|intense|worst)\b|\b(\d{1,2})\s*/\s*10\b', re.I)
NUMBER = r'\d+(?:\.\d+)?|(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|one|two|three|four|five|six|seven|eight|nine|ten|a|an|few|several'
DURATION = re.compile(r'\b('+NUMBER+r')\s*[- ]?\s*(minutes?|hours?|days?|weeks?|months?|years?)\b', re.I)
WORDS = dict(zip(['one','two','three','four','five','six','seven','eight','nine','ten','a','an'], [1,2,3,4,5,6,7,8,9,10,1,1]))
WORDS.update(dict(zip(['eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen'],range(11,20))))
for _word,_number in zip(['twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'],range(20,100,10)):
    WORDS[_word]=_number
    for _unit in ['one','two','three','four','five','six','seven','eight','nine']:
        for _separator in [' ','-']:WORDS[_word+_separator+_unit]=_number+WORDS[_unit]
UNITS = {'minute': 60, 'hour': 3600, 'day': 86400, 'week': 604800, 'month': 2629800, 'year': 31557600}
VITALS = {
    'oxygen_saturation': (r'\b(?:oxygen saturation|oxygen level|SpO2|O2 sat(?:uration)?|sats?)\s*(?:was|is|of|at|:|=)?\s*(\d{1,3}(?:\.\d+)?)\s*%?', '%', (0,100)),
    'heart_rate': (r'\b(?:heart rate|pulse|HR)\s*(?:was|is|of|at|:|=)?\s*(\d{1,3})\s*(?:bpm)?', 'bpm', (10,350)),
    'respiratory_rate': (r'\b(?:respiratory rate|respirations|RR)\s*(?:was|is|of|at|:|=)?\s*(\d{1,3})', '/min', (1,100)),
    'blood_pressure': (r'\b(?:blood pressure|BP)\s*(?:was|is|of|at|:|=)?\s*(\d{2,3})\s*/\s*(\d{2,3})', 'mmHg', None),
    'temperature': (r'\b(?:temperature|temp|fever)\s*(?:was|is|of|at|:|=)?\s*(\d{2,3}(?:\.\d+)?)\s*[° ]*([CF])\b', 'C', (25,45)),
    'glucose': (r'\b(?:blood glucose|blood sugar|glucose)\s*(?:was|is|of|at|:|=)?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mmol/L)\b', None, (0,1500)),
}

def evidence(text, start, end, source='answer', turn_id=None, kind='entity'):
    return dict(text=text[start:end], start=start, end=end, source=source, turn_id=turn_id, kind=kind)

def actor(text, position, inherited='patient'):
    result=inherited
    for m in ACTOR.finditer(text):
        if m.start()>position:break
        value=m.group().lower()
        if value=='i': result='patient'
        elif value.startswith('my '): result='other:'+value.split()[-1]
        elif value=='the patient': result='patient'
        # Pronouns inherit the explicitly named patient; unresolved pronouns stay unknown.
        elif result=='patient' and inherited=='patient': result='unknown'
    return result

def context(text, start, end, kind, inherited='patient'):
    """Bounded ConText-style scopes, with pseudo-negation and affirmative resets."""
    a=0; b=len(text)
    for m in BOUNDARY.finditer(text):
        if m.end()<=start:a=m.end()
        elif m.start()>=end:b=m.start();break
    actors=list(ACTOR.finditer(text[a:start]))
    if actors:
        before=text[a:a+actors[-1].start()]
        cue=re.search(r'(?:(?:not sure|unsure)\s+(?:if|whether)|if|whether|maybe|possibly|afraid(?: that)?|worried(?: that)?)\s*$',before,re.I)
        a+=cue.start() if cue else actors[-1].start()
    # Local clauses stop at a new subject after the entity, too.
    next_actor=ACTOR.search(text,end,b)
    activity_clause=next_actor and re.search(r'\bwhen\s*$',text[end:next_actor.start()],re.I) and re.match(r'I\s+(?:walk|run|exercise)\b',text[next_actor.start():],re.I)
    if next_actor and not activity_clause:b=next_actor.start()
    prefix=text[a:start];suffix=text[end:b];local=text[a:b]
    reset=list(re.finditer(r'\b(?:and (?:have|has|feel|feels|developed)|now (?:have|has))\b',prefix,re.I))
    if reset:prefix=prefix[reset[-1].start():]
    # A comma followed by a fresh affirmative verb ends a negation list.
    reset=list(re.finditer(r',\s*(?:have|has|taking|take|feel|feels|just|only|instead)\b',prefix,re.I))
    if reset:prefix=prefix[reset[-1].start():]
    prefix=' '.join(prefix.split()[-16:])
    pseudo=bool(re.search(r'\b(?:not only|not necessarily|cannot rule out|can.t rule out|not sure|no change in|no relief from)\b',prefix,re.I))
    neg=bool(NEGATION.search(prefix)) and not pseudo
    neg=neg or bool(re.match(r'\s*(?:is |are )?(?:absent|not present|denied|negative)\b',suffix,re.I))
    uncertain=bool(UNCERTAIN.search(prefix) or re.search(r'\b(?:not necessarily|do I have|could I have|could this be|is this|afraid I|worried I|worried that|afraid that)\b',prefix,re.I))
    hypothetical=bool(HYPOTHETICAL.search(prefix) or re.search(r'^\s*(?:could|can|would|does)\b.*\b(?:cause|lead to|result in)\b',local,re.I))
    # Resolution describes a symptom ending; stopping a medication is a separate
    # role and does not make another symptom in the sentence resolved.
    resolved=kind!='medication' and bool(re.search(r'\b(?:no longer|no more|recovered from)\b',prefix,re.I) or
        re.match(r'\s*(?:(?:is|are|has|have)\s+)?(?:now\s+)?(?:resolved|gone(?: away)?|went away|stopped|not anymore|anymore|no longer)\b',suffix,re.I))
    # History-of is a forward cue. It must not reach backward into a current
    # symptom, medication, demographic or measurement preceding the diagnosis.
    historical=bool(re.search(r'\b(?:history of|family history|used to|previously|as a child)\b',prefix,re.I) or
                    re.search(r'\b(?:last year|last month|years? ago|months? ago|in (?:19|20)\d{2})\b',prefix+' '+' '.join(suffix.split()[:8]),re.I))
    ongoing=bool(re.search(r"\b(?:still|currently|now|ongoing|have had|has had|have been|has been|I've had|I've been)\b",local,re.I))
    past=bool(re.search(r'\b(?:had|was|were|experienced|suffered)\b',prefix,re.I)) and not ongoing
    status='negated' if neg or (resolved and kind=='symptom') else 'uncertain' if uncertain else 'hypothetical' if hypothetical else 'confirmed'
    temporal='resolved' if resolved else 'historical' if (historical or past) and not ongoing else 'episodic' if re.search(r'\b(?:sometimes|intermittent|recurring|recurrent|comes and goes|whenever|during (?:panic )?attacks|when I (?:walk|run|exercise))\b',local,re.I) and not ongoing else 'current'
    if hypothetical:temporal='hypothetical'
    acute=bool(re.search(r'\b(?:sudden(?:ly)?|new|just|started|began|today)\b',local,re.I)) and temporal=='current'
    recent=bool(re.search(r'\b(?:now|today|tonight|this morning|yesterday|last night|hours? ago|minutes? ago)\b',local,re.I)) and temporal not in ('historical','hypothetical')
    severities=list(SEVERITY.finditer(local));severity=None
    if severities:
        m=min(severities,key=lambda m:abs(a+m.start()-start))
        valid=not m.group(1) or 0<=int(m.group(1))<=10
        severity=dict(text=m.group(),value=int(m.group(1)) if m.group(1) else m.group().lower(),start=a+m.start(),end=a+m.end(),status='confirmed' if valid else 'uncertain')
    durations=[]
    for m in DURATION.finditer(local):
        raw=m.group(1).lower();value=WORDS.get(raw)
        if value is None:
            try:value=float(raw)
            except ValueError:pass
        unit=m.group(2).lower().rstrip('s')
        # Ages are not symptom durations.
        tail=local[m.end():m.end()+6]
        if unit=='year' and re.match(r'[- ]?old',tail):continue
        durations.append(dict(value=value,reported_value=raw,unit=unit,seconds=None if value is None else value*UNITS[unit],start=a+m.start(),end=a+m.end(),text=m.group()))
    onsets=list(re.finditer(r'\b(?:sudden(?:ly)?|gradual(?:ly)?|started|began|new|since (?:yesterday|today|last night|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b',local,re.I))
    specific=[m for m in onsets if re.match(r'sudden|gradual|since ',m.group(),re.I)]
    onset=min(specific or onsets,key=lambda m:abs(a+m.start()-start)) if onsets else None
    duration=min(durations,key=lambda d:abs(d['start']-end)) if durations else None
    resolution_time=None;event_time=None
    if kind=='symptom' and temporal=='historical':
        dated=[d for d in durations if re.match(r'\s*ago\b',text[d['end']:b],re.I)]
        if dated:
            d=min(dated,key=lambda d:abs(d['start']-end));event_time=dict(seconds_ago=d['seconds'],start=d['start'],end=d['end']+len(re.match(r'\s*ago',text[d['end']:b],re.I).group()),text='')
            event_time['text']=text[event_time['start']:event_time['end']]
            if duration is d:duration=None
    if resolved:
        marker=re.search(r'\b(?:resolved|gone(?: away)?|went away|stopped)\b',local,re.I)
        if marker:
            after=[d for d in durations if d['start']>=a+marker.end() and re.match(r'\s*ago\b',text[d['end']:b],re.I)]
            if after:
                d=after[0];resolution_time=dict(seconds_ago=d['seconds'],start=d['start'],end=d['end']+len(re.match(r'\s*ago',text[d['end']:b],re.I).group()),text='')
                resolution_time['text']=text[resolution_time['start']:resolution_time['end']]
                if duration is d:duration=None
        if resolution_time is None:
            relative=re.search(r'\b(?:just now|now|today)\b',local,re.I)
            if relative:resolution_time=dict(seconds_ago=None,within_last_24_hours=True,start=a+relative.start(),end=a+relative.end(),text=relative.group())
    if kind=='symptom' and temporal=='historical' and event_time is None:
        relative=re.search(r'\btoday\b',local,re.I)
        if relative:event_time=dict(seconds_ago=None,within_last_24_hours=True,start=a+relative.start(),end=a+relative.end(),text=relative.group())
    onset_value=None if not onset else dict(value='relative_time' if onset.group().lower().startswith('since ') else 'sudden' if onset.group().lower().startswith('sudden') else 'gradual' if onset.group().lower().startswith('gradual') else onset.group().lower(),start=a+onset.start(),end=a+onset.end(),text=onset.group())
    # A dated start is distinct from the duration of a symptom or its resolution.
    if onset and not resolved:
        dated=[d for d in durations if d['start']>=a+onset.end() and re.match(r'\s*ago\b',text[d['end']:b],re.I)]
        if dated:
            d=dated[0];onset_value.update(seconds_ago=d['seconds'],start=a+onset.start(),end=d['end']+len(re.match(r'\s*ago',text[d['end']:b],re.I).group()))
            onset_value['text']=text[onset_value['start']:onset_value['end']]
    cues=[]
    for cue in [NEGATION,UNCERTAIN,HYPOTHETICAL]:
        for m in cue.finditer(text[a:b]):cues.append((a+m.start(),a+m.end()))
    subject=actor(text,start,inherited)
    if re.search(r'\bfamily history of\b',prefix,re.I):subject='family'
    if re.search(r'^\s*(?:what is|what are|how does|how do|can you explain)\b',prefix,re.I):
        status='hypothetical';temporal='hypothetical';subject='generic'
    return dict(status=status,temporality=temporal,experiencer=subject,acute=acute,recent=recent,
                severity=severity,duration=duration,resolution_time=resolution_time,event_time=event_time,
                onset=onset_value,
                context_start=a,context_end=b,cues=cues)

def enrich(text, legacy, inherited='patient', turn_id=None):
    normalized=normalize(text);facts=[];seen=set()
    candidates=[]
    lex_matches=[(kind,name,m.start(),m.end()) for kind,name,pattern in PATTERNS for m in pattern.finditer(normalized)]
    for m in legacy['mentions']:
        kind='context' if m['concept'] in {'routine','stable','persistent','planned_care','preventive_care','chronic_care','general_education','nonacute_duration'} else 'condition' if m['concept']=='disease_name' else 'anatomy' if m['concept'].startswith('term_') else 'symptom'
        name=m['concept'] if kind=='context' else re.sub(r'\W+','_',m['text'].lower()).strip('_')
        if m['concept']=='other_symptom' and any(a<=m['start'] and b>=m['end'] for _,_,a,b in lex_matches):continue
        if m['concept'] in ('severe_headache','thunderclap','abdomen_severe','breath_severe','one_side'):
            nested=[(k,n) for k,n,a,b in lex_matches if m['start']<=a and m['end']>=b and k=='symptom']
            if nested:kind,name=nested[0]
        candidates.append((kind,name,m['concept'],m['start'],m['end']))
    candidates.extend((kind,name,name,a,b) for kind,name,a,b in lex_matches)
    for kind,name,concept,start,end in candidates:
        # Explicit lexicon names take precedence over broad aliases at the same span.
        aliases=[(k,n) for k,n,a,b in lex_matches if a==start and b==end]
        if aliases:kind,name=aliases[0]
        if concept=='unresponsive' and re.match(r'\s+to .*?(?:antibiotic|treatment|therap|medication)',normalized[end:],re.I):
            kind,name,concept='context','treatment_nonresponse','treatment_nonresponse'
        if kind=='symptom' and re.fullmatch(r'allergic|allerg(?:y|ies)',normalized[start:end],re.I) and re.match(r'\s+to\b',normalized[end:],re.I):
            kind,name='context','medication_allergy_context'  # Allergy history alone is not an active allergic reaction.
        key=(kind,name,start,end)
        if key in seen:continue
        seen.add(key);ctx=context(normalized,start,end,kind,inherited)
        ev=[evidence(text,start,end,turn_id=turn_id)]
        for a,b in ctx.pop('cues'):ev.append(evidence(text,a,b,turn_id=turn_id,kind='context'))
        ev.append(evidence(text,ctx['context_start'],ctx['context_end'],turn_id=turn_id,kind='context'))
        for attr in ('severity','duration','onset','resolution_time','event_time'):
            if ctx[attr]:ev.append(evidence(text,ctx[attr]['start'],ctx[attr]['end'],turn_id=turn_id,kind=attr))
        confidence=.95 if aliases else .85
        if ctx['experiencer']=='unknown':confidence-=.2
        if ctx['status'] in ('uncertain','hypothetical'):confidence-=.1
        fact=dict(kind=kind,name=name,concept=concept,text=text[start:end],start=start,end=end,
                  evidence=ev,extraction_confidence=round(confidence,2),confidence_basis='literal_dictionary_and_bounded_context',**ctx)
        # Keep the ensemble's original shared clinical vocabulary, not its label/rule outputs.
        # The aliases inherit this fact's improved assertion, temporality and experiencer.
        old_aliases=[m['concept'] for m in legacy['mentions']
                    if (start<=m['start'] and end>=m['end'] or m['start']<=start and m['end']>=end)
                    and (kind!='anatomy' or m['concept'].startswith('term_'))]
        if concept=='treatment_nonresponse':old_aliases=[]
        fact['concept_aliases']=sorted(set([concept]+old_aliases))
        if kind=='medication':
            prefix=normalized[max(ctx['context_start'],start-50):start];suffix=normalized[end:min(ctx['context_end'],end+60)]
            fact['medication_role']='allergy' if re.search(r'allerg(?:y|ic) to',prefix,re.I) else 'discontinued' if re.search(r'stopped|no longer',prefix,re.I) else 'taking' if re.search(r'\b(?:taking|take|takes|on|using|use|started)\b',prefix,re.I) else 'mentioned'
            dose=re.search(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|mL|units?)\b',suffix,re.I)
            if dose:
                fact['dose']=dict(value=float(dose.group(1)),unit=dose.group(2),text=dose.group(),start=end+dose.start(),end=end+dose.end())
                ev.append(evidence(text,end+dose.start(),end+dose.end(),turn_id=turn_id,kind='dose'))
        facts.append(fact)
    explicit_vitals=[(m.end(),name) for name,(pattern,_,_) in VITALS.items() for m in re.finditer(pattern,normalized,re.I)]
    for name,(pattern,unit,bounds) in VITALS.items():
        matches=list(re.finditer(pattern,normalized,re.I))
        if name=='oxygen_saturation':
            for m in re.finditer(r'\b(?:it|the reading|the value|now|currently)\s*(?:is|was|at|:|=)?\s*(\d{1,3}(?:\.\d+)?)\s*%',normalized,re.I):
                previous=sorted((end,n) for end,n in explicit_vitals if end<=m.start() and m.start()-end<=160)
                if previous and previous[-1][1]==name:matches.append(m)
        for m in sorted(matches,key=lambda m:m.start()):
            ctx=context(normalized,m.start(),m.end(),'vital',inherited);ctx.pop('cues')
            # A disease's history in the same sentence is not a past measurement.
            nearby=normalized[max(ctx['context_start'],m.start()-25):min(ctx['context_end'],m.end()+30)]
            dated=bool(re.search(r'\b(?:last year|last month|years? ago|months? ago|in (?:19|20)\d{2})\b',nearby,re.I))
            measurement_past=bool(re.search(r'\b(?:was|were)\b',m.group(),re.I))
            if ctx['temporality']=='historical' and not dated and not measurement_past:ctx['temporality']='current'
            if measurement_past and not re.search(r'\b(?:now|currently|just now)\b',nearby,re.I):ctx['temporality']='historical'
            value=float(m.group(1));valid=True
            if name=='blood_pressure':value={'systolic':value,'diastolic':float(m.group(2))};valid=20<=value['diastolic']<=200 and 30<=value['systolic']<=300
            elif name=='temperature':value=(value-32)*5/9 if m.group(2).upper()=='F' else value
            elif name=='glucose':unit=m.group(2);valid=(0<=value<=80) if unit.lower()=='mmol/l' else (0<=value<=1500)
            if bounds and name!='glucose':valid=bounds[0]<=value<=bounds[1]
            if not valid:ctx['status']='uncertain'
            facts.append(dict(kind='vital',name=name,concept=name,value=value,unit=unit,text=text[m.start():m.end()],start=m.start(),end=m.end(),
                              evidence=[evidence(text,m.start(),m.end(),turn_id=turn_id),evidence(text,ctx['context_start'],ctx['context_end'],turn_id=turn_id,kind='context')],extraction_confidence=.95 if valid else .35,
                              confidence_basis='explicit_value_and_unit' if valid else 'value_outside_extraction_range',**ctx))
    age_pattern=re.compile(r"\b(?:aged?\s*|age\s*[:=]?\s*)(\d{1,3})\b|\b(\d{1,3})\s*[- ]?years?[- ]old\b",re.I)
    for m in age_pattern.finditer(normalized):
        value=int(m.group(1) or m.group(2));ctx=context(normalized,m.start(),m.end(),'demographic',inherited);ctx.pop('cues')
        prefix=normalized[ctx['context_start']:m.start()]
        past_age=bool(re.search(r'\b(?:was|were|used to be|died at|at the age of)\s*(?:a |an |about )?$',prefix,re.I))
        if ctx['temporality']=='historical' and not past_age:ctx['temporality']='current'
        if not 0<=value<=120:ctx['status']='uncertain'
        facts.append(dict(kind='demographic',name='age',concept='age',value=value,text=m.group(),start=m.start(),end=m.end(),evidence=[evidence(text,m.start(),m.end(),turn_id=turn_id)],extraction_confidence=.95 if 0<=value<=120 else .35,confidence_basis='explicit_age',**ctx))
    for m in re.finditer(r'\b(female|male|woman|man|girl|boy)\b',normalized,re.I):
        ctx=context(normalized,m.start(),m.end(),'demographic',inherited);ctx.pop('cues')
        prefix=normalized[ctx['context_start']:m.start()]
        if ctx['temporality']=='historical' and not re.search(r'\b(?:was|were|used to be)\b',prefix,re.I):ctx['temporality']='current'
        value='female' if m.group().lower() in ('female','woman','girl') else 'male'
        facts.append(dict(kind='demographic',name='reported_sex',concept='reported_sex',value=value,text=m.group(),start=m.start(),end=m.end(),evidence=[evidence(text,m.start(),m.end(),turn_id=turn_id)],extraction_confidence=.9,confidence_basis='explicit_reported_sex',**ctx))
    first=ACTOR.search(normalized);target=inherited
    if first and first.group().lower().startswith('my '):target='other:'+first.group().split()[-1].lower()
    active=next((f for f in facts if f['kind']=='symptom' and f['status']=='confirmed' and f['temporality']=='current'),None)
    if active and inherited=='patient':target=active['experiencer']
    latest={};history=[]
    for f in sorted(facts,key=lambda f:f['end']):
        key=(f['kind'],f['name'],f['experiencer']);old=latest.get(key)
        if old:
            if f['temporality']=='historical' and old['temporality']=='current':history.append(f);continue
            history.append(old)
        latest[key]=f
    result=facts_to_info(list(latest.values()),target,legacy);result['extraction_history']=history
    result['observations']=facts  # Exact occurrence contexts survive latest-state consolidation.
    return result

def facts_to_info(facts,target='patient',legacy=None):
    """Compatibility view for existing NLP consumers; values describe the target patient."""
    result=dict(schema_version=VERSION,facts=facts,patient_experiencer=target,mentions=[],vitals={},vital_evidence=[],durations=[],age=None,reported_sex='not_reported',extraction_confidence_is_calibrated=False)
    for f in facts:
        if f.get('kind') not in FACT_KINDS:continue
        if f['experiencer']!=target:continue
        status=f['status'];temporal=f['temporality']
        if f['kind'] in ('vital','demographic'):
            if status=='confirmed' and temporal=='current':
                if f['kind']=='demographic':result[f['name']]=f['value']
                elif f['name']=='blood_pressure':result['vitals'].update(f['value'])
                else:result['vitals'][{'temperature':'temperature_c'}.get(f['name'],f['name'])]=f['value']
            if f['kind']=='vital':result['vital_evidence'].extend(f['evidence'])
            continue
        assertion={'confirmed':'present','negated':'absent','uncertain':'uncertain','hypothetical':'hypothetical'}[status]
        if status=='confirmed' and temporal in ('historical','resolved'):assertion='historical'
        severity=f.get('severity');words=[]
        if severity and severity.get('status','confirmed')=='confirmed':
            v=severity['value'];words=['severe' if v>=7 else 'moderate' if v>=4 else 'mild'] if isinstance(v,(int,float)) else [v]
        for concept in f.get('concept_aliases',[f['concept']]):
            result['mentions'].append(dict(concept=concept,assertion=assertion,temporality=temporal,experiencer=f['experiencer'],acute=f['acute'],recent=f['recent'],severity_words=words,text=f['text'],start=f['start'],end=f['end'],context_start=f['context_start'],context_end=f['context_end'],context_flags=[],extraction_confidence=f['extraction_confidence'],evidence=f['evidence']))
        if f.get('duration'):result['durations'].append(f['duration'])
    result['missing_information']=[key for key,missing in [('age',result['age'] is None),('reported_sex',result['reported_sex']=='not_reported'),('onset_or_duration',not result['durations'] and not any(m['acute'] for m in result['mentions'])),('symptom_severity',not any(m['severity_words'] for m in result['mentions'])),('vital_signs',not result['vitals'])] if missing]
    result['field_status']={key:('missing' if key in result['missing_information'] else 'confirmed') for key in ['age','reported_sex','onset_or_duration','symptom_severity','vital_signs']}
    # Named value slots preserve reported uncertainty/negation instead of silently calling them missing.
    for f in facts:
        if f.get('kind') not in FACT_KINDS:continue
        if f['experiencer']==target:result['field_status'][f['kind']+'.'+f['name']]=f['status']
        if f['experiencer']==target and f['kind']=='demographic':result['field_status'][f['name']]=f['status']
        if f['experiencer']==target:
            for attr,status in f.get('attribute_status',{}).items():
                result['field_status'][f['kind']+'.'+f['name']+'.'+attr]=status
                summary='symptom_severity' if attr=='severity' else 'onset_or_duration'
                if summary in result['missing_information']:result['field_status'][summary]=status
    return result

def new_patient_state(patient_experiencer='patient',patient_id=None):
    from uuid import uuid4
    return dict(schema_version=VERSION,patient_id=patient_id or str(uuid4()),patient_experiencer=patient_experiencer,facts=[],observations=[],history=[],turns=[],unresolved_answers=[],revision=0)

def _update_patient_state_english(state, answer, question=None, field=None):
    """Merge a patient answer, optionally linked to its preceding question/explicit slot.

    No questions are generated. Bare yes/no without a single known target is retained
    as unresolved input, never converted into a clinical assertion.
    """
    from triage_pipeline import extract_info
    if not isinstance(answer,str) or not answer.strip():raise ValueError('A nonempty patient answer is required.')
    updated=copy.deepcopy(state) if state is not None else new_patient_state();turn_id=len(updated['turns'])
    target=updated['patient_experiencer'];info=extract_info(answer,inherited_experiencer=target,turn_id=turn_id)
    if not updated['turns']:target=info['patient_experiencer'];updated['patient_experiencer']=target
    incoming=info['facts'];prompt=extract_info(question,inherited_experiencer=target)['facts'] if question else []
    targets={(f['kind'],f['name']):f for f in prompt if f['kind'] in ('symptom','condition','medication')}
    slot_cues={'demographic.age':r'how old|\bage\b','demographic.reported_sex':r'\bsex\b|male or female',
        'vital.oxygen_saturation':r'oxygen saturation|SpO2|O2 sat|\bsats\b','vital.heart_rate':r'heart rate|\bpulse\b',
        'vital.respiratory_rate':r'respiratory rate','vital.blood_pressure':r'blood pressure|\bBP\b',
        'vital.temperature':r'\btemperature\b|\btemp\b','vital.glucose':r'blood sugar|\bglucose\b'}
    slots=[slot for slot,pattern in slot_cues.items() if question and re.search(pattern,question,re.I)]
    if not targets and len(slots)==1:
        kind,name=slots[0].split('.',1);targets={(kind,name):None}
    attribute='severity' if question and re.search(r'how severe|severity|pain (?:level|rating)|rate .*pain',question,re.I) else 'duration' if question and re.search(r'how long|duration',question,re.I) else 'onset' if question and re.search(r'\bwhen\b|onset|\bstart(?:ed)?\b|\bbegan\b',question,re.I) else None
    if field:
        pieces=field.split('.')
        if len(pieces) not in (2,3):raise ValueError('Use kind.name or kind.name.attribute for a clinical slot.')
        kind,name=pieces[:2]
        if len(pieces)==3:
            attribute=pieces[2]
            if attribute not in ('severity','duration','onset'):raise ValueError('Unsupported clinical attribute.')
        if kind not in ('symptom','condition','medication','vital','demographic'):raise ValueError('Unsupported clinical slot.')
        targets={(kind,name):next((f for f in prompt if f['kind']==kind and f['name']==name),None)}
    short=answer.strip().lower().strip('.! ')
    polar=bool(re.fullmatch(r"yes|yes,? (?:i do|she does|he does)|no|no,? (?:i don.t|she doesn.t|he doesn.t)|not sure|unsure|i don.t know",short))
    numeric_target=next(iter(targets)) if len(targets)==1 else None
    if not incoming and numeric_target and numeric_target[0] in ('vital','demographic'):
        kind,name=numeric_target;f=None
        if short in ('not sure','unsure',"i don't know"):
            f=dict(kind=kind,name=name,concept=name,value=None,unit=None,status='uncertain',temporality='current',experiencer=target,acute=False,recent=False,severity=None,duration=None,onset=None,context_start=0,context_end=len(answer))
        elif kind=='demographic':
            if name=='age' and re.fullmatch(r'\d{1,3}(?: years?(?: old)?)?',short):value=int(re.search(r'\d+',short).group());valid=0<=value<=120
            elif name=='reported_sex' and short in ('male','female'):value=short;valid=True
            else:value=None;valid=False
            if value is not None:f=dict(kind=kind,name=name,concept=name,value=value,status='confirmed' if valid else 'uncertain',temporality='current',experiencer=target,acute=False,recent=False,severity=None,duration=None,onset=None,context_start=0,context_end=len(answer))
        else:
            prefixes={'oxygen_saturation':'SpO2','heart_rate':'heart rate','respiratory_rate':'respiratory rate','blood_pressure':'BP','temperature':'temperature','glucose':'glucose'}
            parsed=extract_info(prefixes[name]+' '+answer,inherited_experiencer=target)['facts']
            f=next((x for x in parsed if x['kind']=='vital' and x['name']==name),None)
        if f:
            if question and re.search(r'\b(?:was|were|last year|previously|ago)\b',question,re.I) and not re.search(r'\b(?:now|currently)\b',question,re.I):f['temporality']='historical'
            f.update(text=answer,start=0,end=len(answer),context_start=0,context_end=len(answer),extraction_confidence=.9,confidence_basis='explicit_question_slot_and_answer')
            f['evidence']=[evidence(answer,0,len(answer),turn_id=turn_id,kind='answer')]
            if question:f['evidence'].append(evidence(question,0,len(question),source='question',turn_id=turn_id))
            incoming=[f]
    leading=re.match(r'\s*(yes|no)\b',answer,re.I)
    if attribute and polar and len(targets)==1 and numeric_target[0] in ('symptom','condition'):
        kind,name=numeric_target;template=targets[numeric_target]
        previous=next((f for f in reversed(updated['facts']) if (f['kind'],f['name'],f['experiencer'])==(kind,name,target)),None)
        if short in ('not sure','unsure',"i don't know"):
            f=copy.deepcopy(previous or template) if (previous or template) else dict(kind=kind,name=name,concept=name,acute=False,recent=False,severity=None,duration=None,onset=None)
            f.update(status=previous['status'] if previous else 'uncertain',temporality=previous['temporality'] if previous else 'current',experiencer=target,text=answer,start=0,end=len(answer),context_start=0,context_end=len(answer),extraction_confidence=.85,confidence_basis='attribute_uncertainty_from_linked_answer')
            f[attribute]=None;f.setdefault('attribute_status',{})[attribute]='uncertain'
            f['evidence']=copy.deepcopy(previous['evidence']) if previous else []
            f['evidence'].append(evidence(answer,0,len(answer),turn_id=turn_id,kind=attribute))
            if question:f['evidence'].append(evidence(question,0,len(question),source='question',turn_id=turn_id))
            incoming=[f]
        else:incoming=[] # A yes/no about an attribute cannot negate symptom presence.
    elif (polar or (leading and not any((x['kind'],x['name'])==numeric_target for x in incoming))) and not attribute and len(targets)==1 and numeric_target[0] in ('symptom','condition','medication'):
        (kind,name),template=next(iter(targets.items()))
        if kind not in ('symptom','condition','medication'):raise ValueError('Yes/no cannot set a numeric slot.')
        f=copy.deepcopy(template) if template else dict(kind=kind,name=name,concept=name,acute=False,recent=False,severity=None,duration=None,onset=None,context_start=0,context_end=len(answer))
        f.update(status='uncertain' if short in ('not sure','unsure',"i don't know") else 'negated' if short.startswith('no') else 'confirmed',temporality='current',experiencer=target,text=answer,start=0,end=len(answer),extraction_confidence=.9,confidence_basis='single_question_target_and_explicit_answer')
        f.pop('semantic_candidate',None)
        f['extraction_provenance']='linked_patient_confirmation'
        # A positive linked answer may also carry a severity expression.
        answer_ctx=context(normalize(answer),len(answer),len(answer),kind,target)
        if f['status']=='confirmed' and answer_ctx.get('severity'):f['severity']=answer_ctx['severity']
        f['evidence']=[evidence(answer,0,len(answer),turn_id=turn_id,kind='answer')]
        if f.get('severity') and answer_ctx.get('severity'):
            f['evidence'].append(evidence(answer,f['severity']['start'],f['severity']['end'],turn_id=turn_id,kind='severity'))
        if template:f['evidence'].append(evidence(question,template['start'],template['end'],source='question',turn_id=turn_id))
        elif question:f['evidence'].append(evidence(question,0,len(question),source='question',turn_id=turn_id))
        incoming=[f]+[x for x in incoming if (x['kind'],x['name'])!=(kind,name)]
    elif not incoming and not attribute and len(targets)==1 and re.fullmatch(r'\s*(?:only\s+)?when I (?:walk|run|exercise)[.! ]*',answer,re.I):
        (kind,name),template=next(iter(targets.items()))
        if kind=='symptom':
            previous=next((old for old in reversed(updated['facts']) if (old['kind'],old['name'],old['experiencer'])==(kind,name,target)),None)
            f=copy.deepcopy(previous or template) if previous or template else dict(kind=kind,name=name,concept=name,acute=False,recent=False,severity=None,duration=None,onset=None)
            f.update(status='confirmed',temporality='episodic',experiencer=target,text=answer,start=0,end=len(answer),context_start=0,context_end=len(answer),extraction_confidence=.8,confidence_basis='linked_symptom_and_explicit_activity_answer')
            f['evidence']=[evidence(answer,0,len(answer),turn_id=turn_id,kind='answer')]
            if question:f['evidence'].append(evidence(question,0,len(question),source='question',turn_id=turn_id))
            incoming=[f]
    elif not incoming and not polar and len(targets)==1:
        (kind,name),template=next(iter(targets.items()))
        previous=next((old for old in reversed(updated['facts']) if (old['kind'],old['name'],old['experiencer'])==(kind,name,target)),None)
        template=template or previous
        if template and kind in ('symptom','condition'):
            f=copy.deepcopy(template);ctx=context(normalize(answer),len(answer),len(answer),kind,target);ctx.pop('cues')
            if attribute=='onset' and ctx.get('duration') and ctx['duration'].get('seconds') is not None and re.search(r'\bago\b',answer,re.I):
                d=ctx['duration']
                ctx['onset']=dict(value=(ctx.get('onset') or {}).get('value','relative_time'),seconds_ago=d['seconds'],start=d['start'],end=d['end'],text=d['text'])
            if ctx['severity'] or ctx['duration'] or ctx['onset'] or re.search(r'\b(?:gone|resolved|no longer)\b',answer,re.I):
                previous=next((old for old in reversed(updated['facts']) if (old['kind'],old['name'],old['experiencer'])==(kind,name,target)),None)
                if previous:f=copy.deepcopy(previous)
                f.update(ctx)
                f.update(experiencer=target,text=answer,start=0,end=len(answer),extraction_confidence=.8,confidence_basis='explicit_question_target_and_attribute_answer')
                if re.search(r'\b(?:gone|resolved|no longer)\b',answer,re.I):f.update(status='negated',temporality='resolved')
                f['evidence']=copy.deepcopy(previous['evidence']) if previous else []
                f['evidence'].append(evidence(answer,0,len(answer),turn_id=turn_id,kind='answer'))
                if question:f['evidence'].append(evidence(question,0,len(question),source='question',turn_id=turn_id))
                for attr in ('severity','duration','onset','resolution_time','event_time'):
                    if ctx[attr]:f['evidence'].append(evidence(answer,ctx[attr]['start'],ctx[attr]['end'],turn_id=turn_id,kind=attr))
                if attribute and ctx.get(attribute):f.setdefault('attribute_status',{})[attribute]='confirmed'
                incoming=[f]
    if not incoming:
        updated['unresolved_answers'].append(dict(turn_id=turn_id,answer=answer,question=question,reason='No unambiguous clinical target/value; state not inferred.'))
    updated.setdefault('observations',[]).extend(copy.deepcopy(info.get('observations',[])))
    if not info.get('observations'):updated['observations'].extend(copy.deepcopy(incoming))
    updated['turns'].append(dict(turn_id=turn_id,answer=answer,question=question,field=field))
    updated['history'].extend(info.get('extraction_history',[]))
    for f in incoming:
        key=(f['kind'],f['name'],f['experiencer'])
        index=next((i for i,old in enumerate(updated['facts']) if (old['kind'],old['name'],old['experiencer'])==key),None)
        if index is not None:
            old=updated['facts'][index];updated['history'].append(copy.deepcopy(old))
            # Remote history cannot overwrite a current assertion about the same person.
            if f['temporality']=='historical' and old['temporality']=='current':updated['history'].append(f);continue
            if f['status']=='confirmed' and old['status']=='confirmed':
                inherited_onset=f.get('onset') is None and old.get('onset') is not None and f.get('attribute_status',{}).get('onset') not in ('uncertain','negated')
                for attr in ['severity','duration','onset']:
                    if f.get(attr) is None and old.get(attr) is not None and f.get('attribute_status',{}).get(attr) not in ('uncertain','negated'):
                        f[attr]=old[attr]
                        f['evidence'].extend(ev for ev in old['evidence'] if ev.get('kind')==attr)
                if inherited_onset and f['temporality']=='current':f['acute']=old['acute'];f['recent']=old['recent']
            updated['facts'][index]=f
        else:updated['facts'].append(f)
    updated['revision']+=1
    updated.update({k:v for k,v in facts_to_info(updated['facts'],target).items() if k not in ('facts','patient_experiencer','schema_version')})
    return updated

def update_patient_state(state, answer, question=None, field=None):
    """Reuse English state merging after aligned clinical normalization.

    Original messages and original-source evidence are restored before returning.
    Clinical facts from different experiencers keep different keys.
    """
    from multilingual_nlp import align,answer_alignment,remap_span,language_hint,VERSION as NLP_VERSION
    canonical,answer_map,changes=answer_alignment(answer,field)
    q_pivot,q_map,_=align(question) if question else (None,[],[])
    # An explicitly linked generic duration question can target the only current
    # symptom, but never guess among multiple symptoms or different people.
    if field is None and q_pivot and re.search(r'how long have you had pain',q_pivot,re.I):
        options=[f for f in (state or {}).get('facts',[]) if f['kind']=='symptom' and f['status']=='confirmed' and f['temporality']=='current' and f['experiencer']==state['patient_experiencer']]
        if len(options)==1:field='symptom.'+options[0]['name']+'.duration'
    updated=_update_patient_state_english(state,canonical,q_pivot,field)
    turn_id=len(updated['turns'])-1;visited=set()
    for f in updated['facts']+updated['history']+updated.get('observations',[]):
        if id(f) in visited:continue
        visited.add(id(f));evs=f.get('evidence',[])
        current=[e for e in evs if e.get('turn_id')==turn_id and e.get('source')=='answer']
        if current:
            remap_span(f,answer,answer_map)
            obj={'start':f['context_start'],'end':f['context_end']};remap_span(obj,answer,answer_map)
            f.update(context_start=obj['start'],context_end=obj['end'],language=language_hint(answer),
                     script='arabic' if re.search(r'[\u0600-\u06ff]',f['text']) else 'latin',
                     extraction_provenance=NLP_VERSION if changes or canonical!=answer else 'existing_english_context',normalized_concept=f['kind']+'.'+f['name'])
            if changes:f['extraction_confidence']=min(.85,f['extraction_confidence'])
            for attr in ('severity','duration','onset','dose','resolution_time','event_time'):
                if f.get(attr) and any(e.get('kind')==attr for e in current):remap_span(f[attr],answer,answer_map)
        for e in evs:
            if e.get('turn_id')!=turn_id:continue
            if e.get('source')=='answer':remap_span(e,answer,answer_map)
            elif e.get('source')=='question' and question:remap_span(e,question,q_map)
    recognized=any(e.get('turn_id')==turn_id and e.get('source')=='answer' for f in updated['facts'] for e in f['evidence'])
    updated['turns'][-1].update(answer=answer,question=question,language=language_hint(answer),normalization=changes,
                               understanding_status='recognized_facts_with_limited_vocabulary' if recognized else 'no_clinical_fact_recognized')
    primary=any(f['kind'] in ('symptom','condition','medication','vital') and f['experiencer']==updated['patient_experiencer']
                and any(e.get('turn_id')==turn_id and e.get('source')=='answer' for e in f['evidence']) for f in updated['facts'])
    routine=any(f['kind']=='context' and f['status']=='confirmed' and f['name'] in ('routine','planned_care','preventive_care','general_education','chronic_care')
                and any(e.get('turn_id')==turn_id and e.get('source')=='answer' for e in f['evidence']) for f in updated['facts'])
    # Explicit ambiguous complaints cannot be explained by a known condition or
    # a routine context from another turn. This recognizes ambiguity, not pain.
    from multilingual_nlp import fold
    folded,fold_map=fold(answer)
    def unresolved_span(a,b):
        return evidence(answer,fold_map[a][0],fold_map[b-1][1],turn_id=turn_id,kind='unresolved_complaint')
    ambiguity=r'\b(?:7aja\s*ghriba|something\s*(?:strange|weird|wrong)|makhnou9|m5nou9)\b|حاجة\s*غريبة|مخنوق'
    unresolved=[unresolved_span(m.start(),m.end()) for m in re.finditer(ambiguity,folded,re.I)]
    # Detect an explicit additional complaint without assuming all unfamiliar
    # words are symptoms. Recognized clinical facts in that clause resolve it.
    markers=r'\b(?:and\s+(?:also\s+)?I\s+(?:have|feel)|also\s+I\s+(?:have|feel)|w\s+zeda\s+3andi)\b|وزادة\s*عندي|اما\s*زادة\s*عندي'
    for m in re.finditer(markers,folded,re.I):
        end=m.end()+len(re.split(r'[.;!?،]',folded[m.end():],maxsplit=1)[0])
        raw_start=fold_map[m.end()-1][1];raw_end=fold_map[end-1][1]
        if not any(f['kind'] in ('symptom','condition','medication','vital') and any(e.get('turn_id')==turn_id and e.get('source')=='answer' and raw_start<=e['start']<raw_end for e in f['evidence']) for f in updated['facts']):
            unresolved.append(unresolved_span(m.start(),end))
    updated['turns'][-1]['unresolved_complaint_spans']=unresolved
    linked_understood=None
    if field:
        pieces=field.split('.')
        matching=[f for f in updated['facts'] if (f['kind'],f['name'])==tuple(pieces[:2])]
        linked_understood=any(
            any(e.get('turn_id')==turn_id and e.get('source')=='answer'
                and (len(pieces)==2 or e.get('kind') in (pieces[2],'answer')) for e in f['evidence'])
            and (len(pieces)==2 or f.get(pieces[2]) is not None or f.get('attribute_status',{}).get(pieces[2])=='uncertain')
            for f in matching)
    updated['turns'][-1]['linked_answer_understood']=linked_understood
    updated['nlp_requires_clarification']=bool(unresolved) or linked_understood is False or (field is None and not primary and not routine)
    for item in updated['unresolved_answers']:
        if item['turn_id']==turn_id:item.update(answer=answer,question=question)
    from clinical_state import facts_to_info
    updated.update({k:v for k,v in facts_to_info(updated['facts'],updated['patient_experiencer']).items() if k not in ('facts','schema_version','patient_experiencer')})
    return updated

def add_semantic_candidates(state,answer,semantic):
    """Append uncertain transformer candidates to this turn, never overwrite facts.

    Candidate observations are separate from confirmed facts; the ML adapter
    does not receive them until a linked patient answer confirms the concept.
    """
    from multilingual_nlp import merge_semantic
    updated=copy.deepcopy(state);turn=len(updated['turns'])-1;target=updated['patient_experiencer']
    local=[f for f in updated['facts'] if any(e.get('turn_id')==turn and e.get('source')=='answer' for e in f['evidence'])]
    info=merge_semantic(dict(facts=copy.deepcopy(local),observations=[],patient_experiencer=target),answer,semantic,target,turn)
    additions=[f for f in info['facts'] if f.get('semantic_candidate')]
    for f in additions:
        key=(f['kind'],f['name'],f['experiencer'])
        previous=next((p for p in updated['facts'] if (p['kind'],p['name'],p['experiencer'])==key),None)
        if previous:
            updated['history'].append(f);continue
        updated['facts'].append(f)
    updated['turns'][-1]['semantic_metadata']=info['semantic_metadata']
    # An unfamiliar new complaint cannot be scored using only old known facts.
    if additions:updated['nlp_requires_clarification']=True
    updated.update({k:v for k,v in facts_to_info(updated['facts'],target).items() if k not in ('facts','schema_version','patient_experiencer')})
    return updated

def clinical_features(info):
    """Compatibility alias; classifier features live separately from clinical state."""
    from ml_features import detailed_features
    return detailed_features(info)

"""Symptom-driven, priority-based missing-slot questions; no generative models.

Templates/translations are prototype drafts pending clinical/dialect review.
"""
from multilingual_nlp import language_hint
from clinical_rules import CHEST,BREATH,STROKE,WHO

VERSION = 'adaptive_templates_1'
# field, relevant complaints, priority, English / Tunisian Arabizi / Arabic
TEMPLATES = [
 ('symptom.speech.onset', ['speech'], 101, 'Did the speaking difficulty start suddenly or gradually? When?', 'S3ouba fel klem bdet faj2a walla bchwaya? Wa9tech?', 'صعوبة الكلام بدات فجأة ولا بشوية؟ وقتاش؟'),
 ('symptom.weakness.onset', ['weakness','numbness'], 101, 'Did the weakness start suddenly or gradually? When?', 'El dh3of bde faj2a walla bchwaya? Wa9tech?', 'الضعف بدا فجأة ولا بشوية؟ وقتاش؟'),
 ('symptom.breath', ['chest'], 100, 'Do you have difficulty breathing now?', '3andek s3ouba fel tanaffos taw?', 'عندك صعوبة في التنفس توا؟'),
 ('symptom.breath_functional_impairment', ['breath'], 99, 'Are you gasping or unable to get words out because of breathing difficulty?', 'Tlahheth walla ma tnajemch tkammel klemek mel tanaffos?', 'تلهث ولا ما تنجمش تكمل كلامك من صعوبة التنفس؟'),
 ('symptom.breath.severity', ['breath','cannot_breathe'], 99, 'How severe is the breathing difficulty? Can you speak normally?', '9addech s3ouba fel tanaffos? Tnajem ta7ki 3adi?', 'قداش صعوبة التنفس؟ تنجم تحكي عادي؟'),
 ('symptom.speech', ['headache','weakness','numbness'], 98, 'Do you have new difficulty speaking now?', '3andek s3ouba jdida fel klem taw?', 'عندك صعوبة جديدة في الكلام توا؟'),
 ('symptom.headache.onset', ['headache'], 95, 'Did the headache start suddenly or gradually? When?', 'Wja3 rasek bde faj2a walla bchwaya? Wa9tech?', 'وجع راسك بدا فجأة ولا بشوية؟ وقتاش؟'),
 ('symptom.confusion', ['headache','fever'], 94, 'Do you feel confused now?', 'T7ess rou7ek met5alet taw?', 'تحس روحك متخلط توا؟'),
 ('symptom.neck_stiffness', ['headache','fever'], 93, 'Do you have a stiff neck now?', '3andek ra9betek yabsa taw?', 'رقبتك يابسة توا؟'),
 ('symptom.chest.duration', ['chest'], 90, 'How long have you had chest pain?', '9addech andek mel wji3a fi sadrek?', 'قداش عندك ملي بدا وجع صدرك؟'),
 ('symptom.chest.onset', ['chest'], 89, 'Did the chest pain start suddenly or gradually? When?', 'Wja3 sadrek bde faj2a walla bchwaya? Wa9tech?', 'وجع صدرك بدا فجأة ولا بشوية؟ وقتاش؟'),
 ('symptom.chest.severity', ['chest'], 88, 'How severe is the chest pain?', '9addech wja3 sadrek 9wi?', 'قداش وجع صدرك قوي؟'),
 ('symptom.headache.duration', ['headache'], 87, 'How long have you had the headache?', '9addech andek mel wja3 fi rasek?', 'قداش عندك ملي بدا وجع راسك؟'),
 ('symptom.fever.duration', ['fever'], 86, 'How long have you had fever?', '9addech andek mel s5ana?', 'قداش عندك ملي بدات السخانة؟'),
 ('symptom.dizziness.onset', ['dizziness'], 85, 'Did the dizziness start suddenly or gradually? When?', 'El doukha bdet faj2a walla bchwaya? Wa9tech?', 'الدوخة بدات فجأة ولا بشوية؟ وقتاش؟'),
 ('symptom.abdominal_pain.duration', ['abdominal_pain'], 84, 'How long have you had abdominal pain?', '9addech andek mel wja3 fi batnek?', 'قداش عندك ملي بدا وجع بطنك؟'),
 ('vital.temperature', ['fever'], 75, 'What temperature did you measure, and in C or F? Say unsure if not measured.', '9addech el temperature bel C walla F? Ken ma 9estech, 9oul ma na3rafch.', 'قداش قست الحرارة، بالـ C ولا F؟ كان ما قستش قول مانعرفش.'),
 ('demographic.age', ['chest','breath','headache','fever','confusion','dizziness','abdominal_pain'], 70, 'How old is the patient?', '9addech 3omr el maridh?', 'قداش عمر المريض؟'),
]

def candidates(state):
    target=state.get('patient_experiencer','patient')
    current=[f for f in state.get('facts',[]) if f['experiencer']==target and f['kind']=='symptom' and f['status'] in ('confirmed','uncertain') and f['temporality']=='current']
    complaints={f['name'] for f in current}
    known={(f['kind'],f['name']):f for f in state.get('facts',[]) if f['experiencer']==target}
    asked={q['field'] for q in state.get('asked_questions',[]) if q.get('answered',True)};result=[]
    retry=state.get('clarification_target')
    if retry:
        result.append(dict(field=retry['field'],priority=111,templates=retry['templates'],
                           reason='Previous answer did not establish the requested fact; clarify without guessing.'))
    for f in current:
        field='symptom.'+f['name']
        if f['status']!='uncertain' or field in asked:continue
        label={'chest':'chest pain','breath':'difficulty breathing','speech':'difficulty speaking'}.get(f['name'],f['name'].replace('_',' '))
        result.append(dict(field=field,priority=110,reason='Confirm the uncertain symptom before asking its attributes.',
                           templates={'en':'Are you experiencing '+label+' right now?',
                                      'tn_arabizi':'El kelma "'+f['text']+'": t7ess biha taw? Fasser aktar.',
                                      'tn_arabic':'تقصد بعبارة «'+f['text']+'» إحساس عندك توا؟ فسر أكثر.'}))
    for field,relevant,priority,en,arabizi,arabic in TEMPLATES:
        if not complaints.intersection(relevant):continue
        pieces=field.split('.');f=known.get(tuple(pieces[:2]))
        missing=f is None or f['status']=='uncertain' or f['temporality']!='current'
        if len(pieces)==3:missing=f is not None and (not f.get(pieces[2]) or f.get('attribute_status',{}).get(pieces[2])=='uncertain')
        if not missing or field in asked:continue
        result.append(dict(field=field,priority=priority,reason='Clarify uncertainty' if f and f['status']=='uncertain' else 'Relevant missing clinical information',
                           templates={'en':en,'tn_arabizi':arabizi,'tn_arabic':arabic}))
    if state.get('nlp_requires_clarification',not state.get('facts')) and state.get('turns') and None not in asked:
        result.append(dict(field=None,priority=105,reason='Clinical complaint not understood; request clarification without inferring a diagnosis.',
                           templates={'en':'Please describe what is bothering you and where you feel it.',
                                      'tn_arabizi':'Fasser chnowa m9al9ek w win t7ess bih.',
                                      'tn_arabic':'فسر شنوة مقلقك ووين تحس بيه.'}))
    return sorted(result,key=lambda q:-q['priority'])

def select_next_question(state,rule_result,language=None):
    if rule_result['critical_flags_detected']:return None
    choices=candidates(state)
    if not choices:return None
    question=choices[0];hint=language or language_hint(state['turns'][-1]['answer'])
    if hint not in ('tn_arabizi','tn_arabic'):hint='tn_arabic' if hint=='mixed' else 'en'
    slot=question['field'] or ''
    sources=[CHEST,BREATH] if any(x in slot for x in ('chest','breath')) else [STROKE,WHO] if any(x in slot for x in ('headache','speech','confusion','neck_stiffness')) else [WHO]
    question.update(id=f"{state['patient_id']}:{state['revision']}:{question['field']}",text=question['templates'][hint],
                    language=hint,patient_experiencer=state['patient_experiencer'],
                    answer_options=[dict(value='yes',text={'en':'Yes','tn_arabizi':'Ey','tn_arabic':'إي'}[hint]),
                                    dict(value='no',text={'en':'No','tn_arabizi':'Le','tn_arabic':'لا'}[hint]),
                                    dict(value='unsure',text={'en':'Unsure','tn_arabizi':'Ma na3rafch','tn_arabic':'مانعرفش'}[hint])],
                    allow_free_text=True,template_version=VERSION,validation_status='unreviewed_prototype',
                    source_guidance=sources,
                    question_priority_is_clinically_validated=False)
    # Numeric/attribute questions need a value rather than a yes/no response.
    if question['field'] is None:
        question['answer_options']=[]
    elif len(question['field'].split('.'))==3 or not question['field'].startswith('symptom.'):
        question['answer_options']=[question['answer_options'][-1]]
    return question

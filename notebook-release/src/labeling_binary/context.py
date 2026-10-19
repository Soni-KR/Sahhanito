"""Character-preserving normalization and conservative, auditable context heuristics."""
import re

def normalize(text):
    # One input codepoint -> one output codepoint, so original evidence offsets survive.
    trans=str.maketrans({'\u2019':"'",'\u2018':"'",'\u201c':'"','\u201d':'"','\u2013':'-','\u2014':'-','\xa0':' '})
    return ''.join(' ' if c.isspace() else c for c in text.translate(trans))

BOUNDARY = re.compile(r'(?<!\d)\.(?!\d)|[!?;\n]+|\b(?:but|however|although|whereas)\b',re.I)
ACTOR = re.compile(r'\b(?:my (?:father|mother|son|daughter|child|baby|husband|wife|brother|sister|friend|partner|tongue|throat|lips|chest|body|arm|leg|face|skin)|i|he|she|they|we|the patient|my patient)\b',re.I)
HISTORY = re.compile(r'\b(?:last (?:year|month|week)|(?:\d+|one|two|three|five|ten|several|many) (?:years?|months?|weeks?|days?) ago|in (?:19|20)\d{2}|as a child|childhood|history of|previously|used to|in the past)\b',re.I)
CURRENT = re.compile(r"\b(?:now|currently|today|this morning|this evening|tonight|just|sudden(?:ly)?|still|ongoing|continue|continues|continuing|(?:i|we) (?:have|am|feel|cannot|can't)|(?:he|she|my \w+|the patient) (?:has|is)|(?:i'm|he's|she's)|experiencing|suffering)\b",re.I)
ACUTE = re.compile(r'\b(?:sudden(?:ly)?|new(?:ly)?|just|today|this morning|this evening|started (?:an? |\d+ )?(?:minutes?|hours?) ago)\b',re.I)
RECENT = re.compile(r'\b(?:now|today|this morning|this evening|just|(?:one|two|three) (?:minutes?|hours?) ago|last (?:hour|night))\b',re.I)
NEGATE = re.compile(r"\b(?:no|not|never|without|den(?:y|ies|ied)|don't|doesn't|didn't|do not|does not|did not|have not|has not|haven't|hasn't|no longer|free of)\b",re.I)
HYPOTHETICAL = re.compile(r"\b(?:if|what if|in case|risk of|worried (?:about|that)|afraid|fear|scared|prevent|could (?:i|it|this)|might|may (?:have|develop)|would|can (?:i|it|this)|possibility of)\b",re.I)
RESOLVED = re.compile(r"\b(?:resolved|went away|stopped|gone away|recovered|nothing now|no symptoms now|no longer|not anymore)\b",re.I)
UNCERTAIN = re.compile(r'\b(?:maybe|possibly|not sure|unsure|uncertain|seems like|might be)\b',re.I)

def clauses(text):
    spans=[]; start=0
    for m in BOUNDARY.finditer(text):
        if start<m.start(): spans.append((start,m.start()))
        start=m.end()
    if start<len(text): spans.append((start,len(text)))
    return spans

def actor_at(text, start, inherited='unknown'):
    actors=[m for m in ACTOR.finditer(text) if m.start()<=start]
    if not actors: return inherited
    value=actors[-1].group().lower()
    if value=='i': return 'patient'
    if value in ('my tongue','my throat','my lips','my chest','my body','my arm','my leg','my face','my skin'): return 'patient'
    if value in ('he','she','they'): return inherited if inherited.startswith('other:') else 'other:unspecified'
    if value=='we': return 'ambiguous_group'
    if value in ('the patient','my patient'): return 'other:patient'
    return 'other:'+value.removeprefix('my ')

def classify(text, start, end, concept, clause_span, previous_actor='unknown'):
    a,b=clause_span
    clause=text[a:b]; prefix=text[a:start]; suffix=text[end:b]
    # Negation/hypothesis scopes are finite lexical windows, not a syntactic parser.
    # Stop scope at a new explicit subject ("no pain and I have breathlessness").
    subject=list(ACTOR.finditer(prefix))
    local_prefix=prefix[subject[-1].end():] if subject else prefix
    if concept=='routine':
        # "no symptoms and need a refill" does not negate the requested refill.
        request=list(re.finditer(r'\b(?:need|request|want|seeking|schedule|book)\b',local_prefix,re.I))
        if request: local_prefix=local_prefix[request[-1].start():]
    local_prefix=' '.join(local_prefix.split()[-10:])
    near_suffix=' '.join(suffix.split()[:8])
    has_history=bool(HISTORY.search(clause))
    retrospective=bool(re.search(r'\b(?:after (?:experiencing|having|being found)|(?:found (?:me|him|her|them))|(?:was|were) (?:recently )?(?:found|experiencing|admitted|hospitali[sz]ed))\b',prefix,re.I))
    has_history=has_history or retrospective
    # Do not let an unrelated "I'm 40" turn a later past-tense event into current evidence.
    past=list(re.finditer(r'\b(?:had|suffered|experienced|developed)\b',local_prefix,re.I))
    ongoing=list(re.finditer(r'\b(?:have|has|am|is|experiencing|currently|still)\b',local_prefix,re.I))
    if past and (not ongoing or past[-1].start()>ongoing[-1].start()):
        # Present perfect "have had" needs clarification unless continued symptoms are explicit.
        has_history=True
    old_hours=any((int(m.group(1))>24 if m.group(2).lower().startswith('hour') else int(m.group(1))>1440)
                  for m in re.finditer(r'\b(\d+) (minutes?|hours?) ago\b',clause,re.I))
    has_history=has_history or old_hours
    explicit_current=bool(CURRENT.search(clause)) or bool(re.search(r'\b(?:overnight|woke up with)\b',clause,re.I))
    numeric_recent=any((int(m.group(1))<=24 if m.group(2).lower().startswith('hour') else int(m.group(1))<=1440)
                       for m in re.finditer(r'\b(\d+) (minutes?|hours?) ago\b',clause,re.I))
    remote_history=bool(HISTORY.search(clause)) or old_hours
    recent=(bool(RECENT.search(clause)) or numeric_recent) and not remote_history
    acute=bool(ACUTE.search(clause)) and not remote_history
    intrinsic_negative=concept in {'breath_severe','unresponsive','imminent_harm','stable','persistent','bleed_severe'}
    # Internal negation that expresses impairment remains inside the concept span.
    absent=bool(NEGATE.search(local_prefix))
    absent=absent and not re.search(r'\bnot only\b',local_prefix,re.I)
    absent=absent or bool(re.match(r'\s*(?:is |are |was |were )?(?:absent|not present|denied)\b',suffix,re.I))
    hypo=bool(HYPOTHETICAL.search(local_prefix))
    # "I fear a heart attack because I have chest pain" resets at the last subject.
    uncertain=bool(UNCERTAIN.search(local_prefix))
    educational=bool(re.search(r'\b(?:i read|article says|for example|what are the symptoms|what are symptoms|example sentence)\b',prefix,re.I))
    treatment_nonresponse=concept=='unresponsive' and bool(re.match(r'\s+to (?:outpatient |the |my |usual )*(?:therap\w*|treatment\w*|medication\w*|antibiotics?)\b',suffix,re.I))
    uncertain=uncertain or educational or treatment_nonresponse
    resolved=bool(RESOLVED.search(near_suffix))
    if absent: assertion='absent'
    elif hypo: assertion='hypothetical'
    elif has_history and not re.search(r'\b(?:still|continues?|ongoing|currently)\b',clause,re.I): assertion='historical'
    elif uncertain: assertion='uncertain'
    elif resolved: assertion='historical'
    else: assertion='present'
    temporal='historical' if assertion=='historical' else ('current' if explicit_current else 'unknown')
    if recent and assertion=='historical': temporal='recent_24h' # Only stroke rules may use resolved recent signs.
    if has_history and re.search(r'\b(?:still|continues?|ongoing|currently)\b',clause,re.I): temporal='current'
    if re.search(r'\b(?:when(?:ever)?|sometimes|occasionally|intermittent(?:ly)?|comes and goes|after (?:heavy lifting|exercise)|during exercise|on exertion|while sleeping|at night|episodes of|surges of)\b',clause,re.I) and not re.search(r'\b(?:right now|currently|still|today)\b',clause,re.I):
        temporal='unknown'
    actor=actor_at(text,start,previous_actor)
    flags=[]
    if absent: flags.append('negation_scope')
    if hypo: flags.append('hypothetical_scope')
    if has_history or resolved: flags.append('historical_scope')
    if educational: flags.append('reported_or_educational_speech')
    if treatment_nonresponse: flags.append('treatment_response_not_consciousness')
    if temporal=='unknown': flags.append('missing_temporality')
    if actor in ('unknown','ambiguous_group'): flags.append('missing_or_ambiguous_experiencer')
    return dict(assertion=assertion,temporality=temporal,experiencer=actor,
                acute=acute,recent=recent,context_start=a,context_end=b,context_flags=flags)

_legacy_classify=classify
def classify(text,start,end,concept,clause_span,previous_actor='unknown'):
    result=_legacy_classify(text,start,end,concept,clause_span,previous_actor)
    a,b=clause_span
    prefix=text[a:start]; suffix=text[end:b]; clause=text[a:b]
    # Reset negation at an affirmative reporting verb, including coordinated lists.
    local=re.split(r"\b(?:have|has|having|experiencing|developed|feel|feels|suffering from)\b",prefix,flags=re.I)[-1]
    local=' '.join(local.split()[-12:])
    absent=bool(NEGATE.search(local)) and not bool(re.search(r'\bnot only\b',local,re.I))
    absent=absent or bool(re.search(r"\b(?:do not|does not|don't|doesn't|never|not) (?:have|has|feel|experience|experiencing)\b[^,;.!?]*$",prefix,re.I))
    absent=absent or bool(re.match(r'\s*(?:is |are )?(?:absent|not present|denied)\b',suffix,re.I))
    hypothetical=bool(re.search(r'\b(?:if|what if|in case|risk of|prevent|could I|might|would|may develop)\b',local,re.I))
    educational=bool(re.search(r'\b(?:what are (?:the )?symptoms|example sentence|article says|I read that)\b',prefix,re.I))
    # Present perfect and ongoing duration are reports, not resolved history.
    ongoing=bool(re.search(r"\b(?:have|has|I've|he's|she's) (?:been |had )|\b(?:experiencing|suffering|still|ongoing|currently|persistent|continues?|for the past)\b",clause,re.I))
    historical=bool(re.search(r'\b(?:history of|used to|as a child|previously|in the past|last year|last month|years? ago|months? ago)\b',local,re.I))
    past=bool(re.search(r'\b(?:had|experienced|suffered)\b',local,re.I))
    resolved=bool(RESOLVED.search(' '.join(suffix.split()[:8])))
    retrospective=bool(re.search(r'\b(?:was|were) (?:recently )?(?:found|admitted|hospitali[sz]ed)\b|\bafter (?:experiencing|having)\b',prefix,re.I))
    subsequent_improvement=bool(re.search(r'\b(?:improved|recovered|resolved|discharged)\b',text[end:end+500],re.I)) and bool(re.search(r'\b(?:developed|had|experienced)\b',prefix,re.I)) and not ongoing
    family=bool(re.search(r'\b(?:family history|family has a history|father had|mother had)\b',prefix,re.I))
    nonresponse=concept=='unresponsive' and bool(re.match(r'\s+to .*?(?:treatment|therap|medication|antibiotic)',suffix,re.I))
    if absent: assertion='absent'
    elif hypothetical or educational: assertion='hypothetical'
    elif family or resolved or subsequent_improvement or (retrospective and not re.search(r'\b(?:still|currently|now)\b',prefix,re.I)) or (historical and not ongoing) or (past and not ongoing and not RECENT.search(clause)): assertion='historical'
    elif nonresponse or UNCERTAIN.search(local): assertion='uncertain'
    else: assertion='present'
    result['assertion']=assertion
    result['temporality']='current' if assertion in ('present','absent','uncertain') else 'historical' if assertion=='historical' else 'unknown'
    if assertion=='historical' and RECENT.search(clause) and not historical: result['temporality']='recent_24h'
    result['acute']=bool(ACUTE.search(clause) or re.search(r'\b(?:yesterday|last night|past \d+ hours|started|began|developed)\b',clause,re.I)) and not historical
    result['recent']=bool(RECENT.search(clause) or re.search(r'\b(?:yesterday|last night|just swallowed|just took)\b',clause,re.I))
    # Resolve he/she to the last explicitly named relative; default report subject is
    # the question's patient, while hypothetical/family mentions remain excluded.
    relatives=list(re.finditer(r'\bmy (?:\d+[- ]year[- ]old )?(father|mother|son|daughter|child|baby|husband|wife|brother|sister|friend|partner)\b',text[:start],re.I))
    if result['experiencer']=='other:unspecified' and relatives: result['experiencer']='other:'+relatives[-1].group(1).lower()
    if result['experiencer']=='unknown': result['experiencer']='patient'
    result['context_flags']=[flag for flag,condition in [('negation_scope',absent),('hypothetical_scope',hypothetical or educational),('historical_scope',assertion=='historical'),('treatment_response_not_consciousness',nonresponse)] if condition]
    if concept=='breath_severe' and re.match(r'\s+properly\b',suffix,re.I):
        result['assertion']='uncertain'; result['context_flags'].append('breathing_severity_not_established')
    recurrent_panic=bool(re.search(r'\bpanic attacks?\b',text,re.I)) and bool(re.search(r'\b(?:past year|past \d+ months|for \d+ years|recurring|recurrent|before exams|social events)\b',text,re.I))
    immediate=bool(re.search(r'\b(?:right now|currently (?:cannot|can.t|unable)|today I|gasping|blue lips|unconscious|sudden(?:ly)?|new chest pain|severe difficulty breathing|severe chest pain)\b',text,re.I))
    if concept in {'breath','breath_severe','chest','other_symptom'} and recurrent_panic and not immediate:
        # Recurrent attacks are not evidence that the symptom is occurring now.
        result['temporality']='episodic'; result['context_flags'].append('recurrent_panic_episode_not_current')
    return result

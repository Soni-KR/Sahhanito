"""No metadata enters this module: public label() accepts question text only."""
import hashlib
import itertools
import json
import re
from pathlib import Path
from .context import normalize, clauses, classify, actor_at

CONFIG_PATH=Path(__file__).with_name('rules.v1.json')

class Labeler:
    def __init__(self, config_path=CONFIG_PATH):
        raw=Path(config_path).read_bytes(); self.config=json.loads(raw)
        self.config_sha=hashlib.sha256(raw).hexdigest()
        self.patterns={k:re.compile(v,re.I) for k,v in self.config['concepts'].items()}

    def mentions(self, original):
        text=normalize(original); spans=clauses(text); result=[]
        for concept,pattern in self.patterns.items():
            for match in pattern.finditer(text):
                span=next(((a,b) for a,b in spans if a<=match.start()<b), (0,len(text)))
                context=classify(text,match.start(),match.end(),concept,span)
                result.append(dict(concept=concept,text=original[match.start():match.end()],
                    start=match.start(),end=match.end(),**context))
        return sorted(result,key=lambda x:(x['start'],x['end'],x['concept']))

    def eligible(self, mention, rule):
        current=mention['assertion']=='present' and mention['temporality']=='current'
        recent=(rule.get('allow_recent_resolved') and mention['assertion']=='historical'
                and mention['temporality']=='recent_24h')
        return ((current or recent) and mention['experiencer'] not in ('unknown','ambiguous_group')
                and (not rule.get('require_acute') or mention['acute'])
                and (not rule.get('require_recent') or mention['recent']))

    def label(self, question):
        if question is not None and not isinstance(question,str):
            raise TypeError('Only the patient question string is accepted.')
        original=question or ''
        mentions=self.mentions(original)
        fired=[]; used=[]; categories=[]; flags=[]
        if '\ufffd' in original or '\u0092' in original or '\u00e2\u20ac' in original:
            flags.append('source_encoding_artifact')
        for rule in self.config['rules']:
            groups=[[m for m in mentions if m['concept'] in options and self.eligible(m,rule)] for options in rule['all_of']]
            if any(not g for g in groups): continue
            for combo in itertools.product(*groups):
                if len({m['experiencer'] for m in combo})!=1: continue
                if max(m['end'] for m in combo)-min(m['start'] for m in combo)>self.config['max_combination_chars']: continue
                # Cross-clause combinations must not combine two distinct symptom episodes.
                if len(combo)>1 and any(m['temporality']!='current' for m in combo): continue
                fired.append(rule['id']); categories.append(rule['category']); used.extend(combo); break

        # Contradictions are not reconciled by silently choosing the stronger interpretation.
        conflicts=[]
        for m in mentions:
            if m['concept'] in {'other_symptom','disease_name','chronic_care','planned_care','preventive_care','general_education','nonacute_duration'}: continue
            if m['assertion']=='present' and m['temporality']=='current':
                for n in mentions:
                    if (n['concept']==m['concept'] and n['assertion']=='absent'
                            and n['experiencer']==m['experiencer'] and n['temporality']=='current'):
                        conflicts.extend([m,n]); break
        stable=[m for m in mentions if m['concept']=='stable' and m['assertion']=='present' and m['temporality']=='current']
        routine=[m for m in mentions if m['concept']=='routine' and m['assertion']=='present']
        relevant=[m for m in mentions if m['concept'] not in {'stable','routine','disease_name','persistent','allergen','high_speed','extreme_pain'}]
        active=[m for m in relevant if m['assertion'] in ('present','uncertain') and m['temporality']!='historical']
        unresolved=[m for m in relevant if m['assertion']=='hypothetical' or m['temporality']=='unknown']
        asymptomatic=[m for m in stable if re.search(r'no .*symptoms|asymptomatic|symptom[- ]free',normalize(m['text']),re.I)]
        if fired and asymptomatic and any(m['temporality']=='current' for m in used):
            conflicts.extend(asymptomatic+used)
        if conflicts:
            label=-1; reason='Contradictory current symptom evidence; manual clarification required.'
            used=conflicts; fired=[]; categories=[]; flags.append('contradictory_evidence')
        elif fired:
            label=1; reason='Explicit question evidence satisfies contextual red-flag rule(s): '+', '.join(fired)+'.'
        else:
            text=normalize(original)
            # Broad veto for unmodeled acute complaints: an engineering safeguard, not a diagnosis.
            acute_unmodeled=bool(re.search(r'\b(?:severe|sudden|unbearable|worsening|new symptoms|emergency|urgent|overdose|suicid\w*|harm myself|abnormal|too high|too low|asymptomatic except)\b',text,re.I))
            known_actor=stable and stable[0]['experiencer'] not in ('unknown','ambiguous_group')
            same_actor=known_actor and any(m['experiencer']==stable[0]['experiencer'] for m in routine)
            if same_actor and not active and not unresolved and not acute_unmodeled:
                label=0; fired=[self.config['negative_rule']['id']]; used=[stable[0],next(m for m in routine if m['experiencer']==stable[0]['experiencer'])]
                categories=['Routine/stable presentation']; reason='Explicit stable or symptom-free status and routine purpose; no active recognized symptom or unresolved warning mention. This does not establish medical safety.'
            else:
                label=-1; used=mentions
                if not original.strip(): reason='Missing patient question.'
                elif not mentions: reason='No supported context-aware decision; symptoms or clinical state insufficiently specified.'
                elif unresolved or any(m['temporality']=='unknown' for m in relevant): reason='Missing or ambiguous onset, assertion, severity, or experiencer; no supported complete rule.'
                elif active: reason='Reported symptoms do not establish a supported emergency rule or explicit routine/stable presentation.'
                else: reason='Only historical, absent, hypothetical, disease-name or incomplete evidence; no justified binary label.'
        if any(m['context_flags'] for m in relevant): flags.append('possible_context_error')
        if any(m['temporality']=='unknown' for m in relevant): flags.append('missing_critical_information')
        if any(m['experiencer'].startswith('other:') for m in mentions): flags.append('third_person_report')
        if any(m['concept'] in {'disease_name','poison_exposure','imminent_harm','head_injury','faint'} for m in mentions): flags.append('high_risk_requires_verification')
        # Evidence spans always quote the actual original question and include offsets/context.
        unique={(m['start'],m['end'],m['concept']):m for m in used}
        evidence=list(unique.values())
        return dict(emergency_label=label,emergency_category='; '.join(dict.fromkeys(categories)) or 'Uncertain',
                    triggered_rules=list(dict.fromkeys(fired)),evidence_spans=evidence,label_reason=reason,
                    label_source='rule_based_weak_label',needs_human_review=True,
                    labeling_version=self.config['version'],context_mentions=mentions,qc_flags=sorted(set(flags)))

def label_record(record,labeler):
    """Allowlist boundary: all metadata is ignored by construction."""
    return labeler.label(record.get('question_en'))

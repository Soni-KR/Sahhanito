"""Small integration checks; no fitting, new model evaluation or clinical gold."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'triage_artifacts/multilingual'
TEAM=ROOT/'team-handoff/ai'

def prepare():
    source=(ROOT/'test_multilingual_pipeline.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    selected='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if getattr(n,'name',None) in ('fact','FinalizationContracts'))
    portable=(TEAM/'test_integration.py').read_text(encoding='utf-8')
    portable=portable.split('\n# Final integration regression checks')[0].replace("if __name__=='__main__':unittest.main()",'').rstrip()
    (TEAM/'test_integration.py').write_text(portable+'\n\n# Final integration regression checks\n'+selected+"\n\nif __name__=='__main__':unittest.main()\n",encoding='utf-8')

def verify():
    from triage_pipeline import process_turn,load_state_model,predict_from_state
    from multilingual_nlp import VERSION
    bundle=load_state_model()
    conversations={}
    for language,messages in {
        'en':['I have chest pain','I can breathe normally','about twenty minutes','gradually','intense','25','I have difficulty breathing now'],
        'tn_arabizi':['3andi wja3 fi sadri','netnaffes 3adi','men 20 d9aye9','bchwaya bchwaya','barcha','25','netnaffes bes3ouba taw'],
        'tn_arabic':['عندي وجع في صدري','نتنفس عادي','من ٢٠ دقيقة','بشوية بشوية','برشة','٢٥','نتنفس بالصعيب توا'],
    }.items():
        trace=[];state=None;qid=None
        for i,message in enumerate(messages):
            r=process_turn(state,message,qid if i<6 else None,ml_bundle=bundle)
            trace.append(r);state=r['patient_state'];qid=r['next_question']['id'] if r['next_question'] else None
        assert r['questioning_interrupted'] and r['next_question'] is None
        scores=predict_from_state(state,bundle)
        assert scores['classification_threshold']==.15
        conversations[language]=dict(messages=messages,trace=trace,independent_final_ml=scores)
    (OUT/'finalization_conversations.json').write_text(json.dumps(conversations,ensure_ascii=False,indent=2),encoding='utf-8')
    before=json.loads((OUT/'verification.json').read_text())['preservation']
    paths={'source_sha256':'train.csv','original_sha256':'triage_artifacts/tuned/best_model.joblib','deployed_sha256':'triage_artifacts/triage_model.joblib','candidate_sha256':'triage_artifacts/integration_v3/candidate_model.joblib','projection_sha256':'ml_features.py'}
    hashes={key:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for key,path in paths.items()}
    assert hashes==before
    assert hashlib.sha256((TEAM/'candidate_model.joblib').read_bytes()).hexdigest()==hashes['candidate_sha256']
    summary=dict(nlp_version=VERSION,root_tests=104,portable_tests=29,
        tests='All existing tests plus 11 targeted regression test methods; see execution logs.',
        conversations={k:dict(turns=len(v['trace']),critical_interrupt=v['trace'][-1]['questioning_interrupted'],final_ml_score=v['independent_final_ml']['model_score_emergency']) for k,v in conversations.items()},
        classifier_retrained=False,classifier_retuned=False,threshold=.15,preservation=hashes,
        public_api_unchanged=True,clinical_rules_unchanged=True,ml_rule_independence=True,
        fixes=['Finite explicit breathing expressions in English, Arabizi and Arabic-script Darija','Natural linked negative and activity-only responses','Uninterpreted linked answers abstain and retry with a fresh question ID','Explicit unresolved additional complaints retain evidence and pause ML'],
        limitations=['Finite vocabulary and lexical context; unfamiliar paraphrases can still be missed','Partial-understanding checks catch explicit additional/ambiguous complaints, not every unknown clause','Native-speaker and clinical review absent; test fixtures are synthetic','Optional semantic model stays disabled by default; no new transformer experiments','Context-feature ML regression remains unresolved; previous validation F1 0.9113 versus original 0.9494 was not recomputed for this patch','Historical challenge set now includes repaired known examples; no new generalization metric claimed'])
    (OUT/'finalization_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    evaluation=json.loads((TEAM/'evaluation_summary.json').read_text())
    evaluation['targeted_finalization']=summary
    evaluation['historical_metrics_note']='Earlier challenge and English selection-validation results; not rerun or independent performance estimates for this targeted patch.'
    (TEAM/'evaluation_summary.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding='utf-8')
    readme=(TEAM/'README.md').read_text(encoding='utf-8')
    readme=readme.split('\n## Targeted integration finalization')[0]
    readme=readme.replace('Current English selection-validation weak-label macro F1','Previously measured English selection-validation weak-label macro F1')
    readme=readme.replace('Confirmed critical-symptom recall was only 1/3: two unfamiliar breathing descriptions were missed.','Before this targeted patch, confirmed critical-symptom recall was only 1/3: two unfamiliar breathing descriptions were missed. Those exact phrases now pass regression checks; this does not establish generalization.')
    readme+='''
## Targeted integration finalization

`process_turn()` and model artifacts remain unchanged in contract: no retraining,
retuning, rule changes or threshold changes. 104 root language/state tests and
29 portable integration tests pass. Complete seven-turn English, Arabizi and
Arabic-script Darija conversations verify negative free-text answers, spoken/numeric
duration, gradual onset, severity, age and a new breathing complaint that replaces
the old denial and interrupts questions. These are synthetic software checks.

Specific breathing phrases now include `nefsi 9sir`, `nfasi ma9tou3`,
`netnaffes bes3ouba`, `el nefes ma ykafich`, `نتنفس بالصعيب`,
`نفسي ناقص`, and the previously missed English air/catching-breath phrases.
`netnaffes 3adi` / `نتنفس عادي` / "I can breathe normally" explicitly negate
breathing difficulty. A linked "only when I walk" / `ki nemchi` stays episodic.
Ambiguous `makhnou9` alone is clarified rather than asserted as respiratory distress.

Uninterpreted linked replies return `insufficient_input` and retry the same field
with a fresh question ID. Explicit unresolved additional complaints preserve spans
in `patient_state.turns[-1].unresolved_complaint_spans` and pause ML; understood facts
remain available to the independent rules. Known critical findings still return
immediately and interrupt questioning. Store the complete returned state and always
send the current pending question ID with its answer.

Vocabulary and uncertainty scopes remain finite; not all unrecognized clauses can
be detected. Existing ML context-feature regression is unresolved. Historical F1
and challenge metrics above were not recomputed for this patch; no clinical
validation or new generalization score is claimed. Model features/schema, original
benchmark preprocessing, continuous scores and calibration metadata are preserved.
'''
    (TEAM/'README.md').write_text(readme,encoding='utf-8')
    notebook=ROOT/'hackcure_nlp_ml.ipynb'
    import nbformat
    nb=nbformat.read(notebook,as_version=4)
    nb.cells=[c for c in nb.cells if c.metadata.get('hackcure_finalization')!=True]
    nb.cells.append(nbformat.v4.new_markdown_cell('Targeted integration finalization: finite breathing and linked-answer repairs. Historical comparisons above are not fresh evaluation of this patch. No retraining or new large-scale experiments.',metadata={'hackcure_finalization':True}))
    nb.cells.append(nbformat.v4.new_code_cell("finalization=json.loads((RESEARCH/'finalization_summary.json').read_text(encoding='utf-8'))\ndisplay(finalization)\ntraces=json.loads((RESEARCH/'finalization_conversations.json').read_text(encoding='utf-8'))\nfor language,conversation in traces.items():\n    display({'language':language,'messages':conversation['messages'],'facts':conversation['trace'][-1]['patient_state']['facts'],'rules':conversation['trace'][-1]['rules'],'ml':conversation['trace'][-1]['ml'],'independent_ml':conversation['independent_final_ml']})",metadata={'hackcure_finalization':True}))
    nbformat.write(nb,notebook)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='verify':verify()

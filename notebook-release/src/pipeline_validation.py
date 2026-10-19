"""NLP reference scoring and train/validation-only ML diagnostics; no deployment."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from triage_pipeline import ROOT, ARTIFACTS, extract_info, file_hash, extraction_signature


def frozen_preprocessing(version='original'):
    """Load byte-preserved preprocessing with its own dependency objects.

    Import paths alone are privately bound in the parsed source; preserved source
    bytes and clinical computations are unchanged. Live modules are not replaced.
    Signature proof is checked against the fitted bundle.
    """
    import types,sys,hashlib,ast
    from contextlib import contextmanager
    snapshot=ARTIFACTS/'benchmark';proof=json.loads((snapshot/'manifest.json').read_text())
    for name,sha in proof['snapshot_sha256'].items():assert file_hash(snapshot/name)==sha
    package=types.ModuleType('_hackcure_frozen');package.__path__=[str(snapshot)]
    sys.modules['_hackcure_frozen']=package
    def module(name,path,package_name='_hackcure_frozen',virtual_file=None):
        m=types.ModuleType(name);m.__file__=str(virtual_file or path);m.__package__=package_name;sys.modules[name]=m
        tree=ast.parse(path.read_bytes(),filename=str(path))
        bindings={'labeling_binary.context':'_hackcure_frozen.context','labeling_binary.base':'_hackcure_frozen.base',
                  'clinical_state':'_hackcure_frozen.state_v2','triage_pipeline':'_hackcure_frozen.pipeline_v2'}
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module in bindings:node.module=bindings[node.module]
        exec(compile(tree,str(path),'exec'),m.__dict__)
        return m
    context=module('_hackcure_frozen.context',snapshot/'context.py')
    base=module('_hackcure_frozen.base',snapshot/'base.py')
    @contextmanager
    def aliases(values):
        previous={k:sys.modules.get(k) for k in values}
        sys.modules.update(values)
        try:yield
        finally:
            for k,v in previous.items():
                if v is None:sys.modules.pop(k,None)
                else:sys.modules[k]=v
    imports={}  # Private import binding changes no clinical computation or live modules.
    if version=='original':
        with aliases(imports):pipeline=module('_hackcure_frozen.preprocessing',snapshot/'preprocessing.py',virtual_file=ROOT/'triage_pipeline.py')
        pipeline.CONFIG=snapshot/'rules.v3.json';pipeline.NLP=base.Labeler(pipeline.CONFIG)
        bundle=joblib.load(ARTIFACTS/'tuned/best_model.joblib')
        assert bundle['feature_signature']==proof['original_feature_signature']
        return pipeline,bundle,aliases,imports
    if version!='deployed_v2':raise ValueError('Unknown pinned preprocessing version.')
    saved=ARTIFACTS/'integration_v3'
    with aliases(imports):state=module('_hackcure_frozen.state_v2',saved/'before_clinical_state.py')
    with aliases(imports):pipeline=module('_hackcure_frozen.pipeline_v2',saved/'before_triage_pipeline.py',virtual_file=ROOT/'triage_pipeline.py')
    pipeline.CONFIG=snapshot/'rules.v3.json';pipeline.NLP=base.Labeler(pipeline.CONFIG)
    sig=hashlib.sha256(''.join(file_hash(p) for p in [saved/'before_triage_pipeline.py',saved/'before_clinical_state.py',snapshot/'rules.v3.json',snapshot/'base.py',snapshot/'context.py']).encode()).hexdigest()
    bundle=joblib.load(ARTIFACTS/'triage_model.joblib');assert sig==bundle['feature_signature']
    assert file_hash(ARTIFACTS/'triage_model.joblib')==proof['deployment_sha256_before']
    pipeline.extraction_signature=lambda:sig
    return pipeline,bundle,aliases,imports


def predict_original_benchmark(question):
    pipeline,bundle,_,_=frozen_preprocessing()
    features=pipeline.to_features(pipeline.extract_info(question))
    score=float(bundle['model'].predict_proba([features])[0,1]);label=int(bundle['model'].predict([features])[0])
    return dict(emergency_label=label,model_score_emergency=score,model_score_non_emergency=1-score,
                score_calibration='none',score_is_clinically_calibrated=False,ml_preprocessing_version='original_frozen')


def predict_preserved_deployment(question):
    pipeline,bundle,aliases,imports=frozen_preprocessing('deployed_v2')
    with aliases(imports):result=pipeline.predict_case(question,bundle)
    result['ml_preprocessing_version']='preserved_state_v2'
    result['ml_feature_schema']='preserved_state_v2'
    result['model_candidate_only']=False
    result['classification_threshold']=bundle['model'].threshold
    result['threshold_reference']='preserved_experimental_not_decision_policy'
    return result


def validate_multilingual_integration():
    """Reevaluate the existing v3 model at its fixed threshold, validation only.

    No fit, calibration fitting, threshold selection or test-question extraction.
    This checks English weak-label agreement after the clinical NLP changes.
    """
    from triage_pipeline import load_state_model,to_features
    from triage_tuning import score_diagnostics
    from threadpoolctl import threadpool_limits
    out=ARTIFACTS/'multilingual';bundle=load_state_model()
    proof=json.loads((out/'compatibility.json').read_text())
    assert file_hash(ROOT/'train.csv')==proof['source_sha256']
    selected=pd.read_csv(ARTIFACTS/'integration_v3/data_splits.csv')
    selected=selected.loc[selected.split.eq('validation_selection')].copy()
    old=pd.read_csv(ARTIFACTS/'integration_v3/validation_predictions.csv')
    assert set(selected.case_id)==set(old.case_id) and len(selected)==2546
    # Read only the question column and select allowed row IDs before extraction.
    questions=pd.read_csv(ROOT/'train.csv',usecols=['question_en'],keep_default_na=False)['question_en']
    selected['text']=questions.loc[selected.source_record_index].to_numpy();del questions
    features=[to_features(extract_info(text)) for text in selected.text]
    with threadpool_limits(limits=2):raw=bundle['model'].model.predict_proba(features)[:,1]
    scores=bundle['model'].calibrator.transform(raw);threshold=bundle['model'].threshold
    y=selected.emergency_label.to_numpy()
    prior=old.set_index('case_id').loc[selected.case_id,'score_emergency'].to_numpy()
    metrics=dict(records=len(selected),partition='validation_selection',classifier_retrained=False,threshold_tuned=False,
                 calibration_refitted=False,threshold=threshold,candidate_sha256=bundle['model_sha256'],
                 extraction_signature=extraction_signature(),reference='English weak annotations, already used in model development; not independent clinical accuracy',
                 before=measures(y,prior,threshold),after=measures(y,scores,threshold),
                 after_score_diagnostics=score_diagnostics(y,scores),
                 changed_predictions=int(np.sum((prior>=threshold)!=(scores>=threshold))),
                 mean_absolute_score_change=float(np.mean(abs(scores-prior))),
                 source_and_models_preserved=all(file_hash(p)==proof[k] for p,k in [
                     (ROOT/'train.csv','source_sha256'),(ARTIFACTS/'triage_model.joblib','deployed_sha256'),
                     (ARTIFACTS/'tuned/best_model.joblib','original_sha256'),(ARTIFACTS/'integration_v3/candidate_model.joblib','candidate_sha256')]))
    pd.DataFrame(dict(case_id=selected.case_id,emergency_label=y,old_score=prior,new_score=scores,new_raw_score=raw)).to_csv(out/'english_validation_predictions.csv',index=False)
    (out/'english_validation.json').write_text(json.dumps(metrics,indent=2))
    return metrics


def measures(y, score, threshold):
    pred=np.asarray(score)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(threshold=float(threshold),macro_f1=float(f1_score(y,pred,average='macro')),
                emergency_precision=float(precision_score(y,pred,zero_division=0)),
                emergency_recall=float(recall_score(y,pred,zero_division=0)),
                false_positives=int(fp),false_negatives=int(fn),true_positives=int(tp),true_negatives=int(tn),
                roc_auc=float(roc_auc_score(y,score)))


def run_regression_audit():
    """Freeze extractor/ensemble; diagnose only training and validation cases.

    Linear ablations isolate representations in a fixed lightweight probe, not
    ensemble performance. Masked features are distribution-shift diagnostics,
    not causal estimates. No classifier, threshold or calibration is deployed.
    """
    out=ARTIFACTS/'ml_regression';out.mkdir(exist_ok=True)
    old=joblib.load(ARTIFACTS/'tuned/best_model.joblib')
    new=joblib.load(ARTIFACTS/'state_v2/model.joblib')
    assert new['feature_signature']==extraction_signature()
    assert file_hash(ROOT/'train.csv')==new['metrics']['source_sha256']
    assignments=pd.read_csv(ARTIFACTS/'state_v2/data_splits.csv')
    # Filter identifiers before selecting questions/clinical caches; test rows
    # never enter a diagnostic frame, fit, prediction or review export.
    allowed=assignments[assignments.split.ne('test')].drop_duplicates('case_id')
    ids=set(allowed.case_id)
    old_cache=pd.read_parquet(ARTIFACTS/'nlp_features.parquet',filters=[('case_id','in',list(ids))])
    new_cache=pd.read_parquet(ARTIFACTS/'state_v2/nlp_features.parquet',filters=[('case_id','in',list(ids))])
    def attach(cache):
        f=allowed.merge(cache.drop_duplicates('case_id')[['case_id','features_json']],on='case_id',validate='one_to_one')
        f['features']=f.features_json.map(json.loads)
        return f
    a,b=attach(old_cache),attach(new_cache)
    assert a.case_id.tolist()==b.case_id.tolist() and not set(a.case_id)&set(assignments.loc[assignments.split.eq('test'),'case_id'])
    train=a.split.eq('train');val=a.split.eq('validation_selection')
    av,bv=a.loc[val],b.loc[val];y=av.emergency_label.to_numpy()
    sa=old['model'].predict_proba(av.features.tolist())[:,1]
    raw=new['model'].model.predict_proba(bv.features.tolist())[:,1]
    sb=new['model'].predict_proba(bv.features.tolist())[:,1]
    rows=[dict(version='original_frozen',**measures(y,sa,old['model'].threshold)),
          dict(version='updated_frozen',**measures(y,sb,new['model'].threshold)),
          dict(version='updated_raw_threshold_0.5_diagnostic',**measures(y,raw,.5))]
    pd.DataFrame(rows).to_csv(out/'validation_comparison.csv',index=False)
    # Component discrimination and score shift, before voting/calibration/thresholding.
    components=[]
    for version,wrapped,frame in [('original',old['model'],av),('updated',new['model'],bv)]:
        for name,member in zip(wrapped.model.named_estimators_.keys(),wrapped.model.estimators_):
            s=member.predict_proba(frame.features.tolist())[:,1]
            components.append(dict(version=version,member=name,**measures(y,s,.5),
                negative_score_median=float(np.median(s[y==0])),positive_score_median=float(np.median(s[y==1]))))
    pd.DataFrame(components).to_csv(out/'component_diagnostics.csv',index=False)
    previous=sa>=old['model'].threshold;current=sb>=new['model'].threshold
    newly_fp=(y==0)&current&~previous;correct_neg=(y==0)&~current
    records=[]
    # Shared feature changes: count associations among new FPs and correct negatives.
    features=sorted(set().union(*(x.keys() for x in av.features),*(x.keys() for x in bv.features)))
    for key in features:
        if key.startswith(('fact.','fact_time.','onset.','duration_log.','medication_role.')):continue
        delta=np.array([o.get(key)!=n.get(key) for o,n in zip(av.features,bv.features)])
        n=int((delta&newly_fp).sum());c=int((delta&correct_neg).sum())
        if n>=5:records.append(dict(feature=key,new_fp_changed=n,new_fp_total=int(newly_fp.sum()),
            correct_negative_changed=c,correct_negative_total=int(correct_neg.sum()),
            change_rate_difference=float(delta[newly_fp].mean()-delta[correct_neg].mean())))
    pd.DataFrame(records).sort_values('change_rate_difference',ascending=False).to_csv(out/'feature_change_associations.csv',index=False)
    extra_prefixes=('fact.','fact_time.','onset.','duration_log.','medication_role.')
    clinical=lambda d:{k:v for k,v in d.items() if not k.startswith(extra_prefixes) and k not in ('respiratory_rate_scaled','glucose_mgdl_scaled')}
    variants={
        'original_features':a.features.tolist(),
        'updated_all':b.features.tolist(),
        'updated_compatibility_only':[clinical(d) for d in b.features],
        'updated_without_demographics':[{k:v for k,v in d.items() if not k.startswith(('age_','reported_sex','fact.confirmed.demographic','fact_time.current.demographic'))} for d in b.features],
        'updated_without_new_medication_features':[{k:v for k,v in d.items() if not ('medication.' in k or k.startswith('medication_role.'))} for d in b.features],
    }
    # Diagnostic restorations are research probes only: old assertions/sex
    # inference are not copied back into patient state or deployed features.
    groups={
        'demographics':lambda k:k.startswith(('age_','reported_sex','missing.age','missing.reported_sex')),
        'mention_context':lambda k:k.startswith(('mention.','time.','current.','acute.','recent.','severity.','third_person_report')),
        'routine_chronic_context':lambda k:any(s in k for s in ('chronic_care','nonacute_duration','routine','elective')),
        'all_original_shared':lambda k:True,
    }
    for group,predicate in groups.items():
        restored=[]
        for o,n in zip(a.features,b.features):
            d={k:v for k,v in n.items() if not (predicate(k) and k in clinical(n))}
            d.update({k:v for k,v in o.items() if predicate(k)})
            restored.append(d)
        variants['diagnostic_restore_'+group]=restored
    probe=[]
    for name,values in variants.items():
        print('Fixed linear ablation:',name,flush=True)
        classifier=Pipeline([('vectorizer',DictVectorizer()),('classifier',LogisticRegression(C=1,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=42))])
        classifier.fit([d for d,keep in zip(values,train) if keep],a.loc[train,'emergency_label'])
        score=classifier.predict_proba([d for d,keep in zip(values,val) if keep])[:,1]
        probe.append(dict(representation=name,**measures(y,score,.5)))
    pd.DataFrame(probe).to_csv(out/'linear_feature_ablations.csv',index=False)
    # Fixed-model masking only; does not retrain or replace the ensemble.
    masked=[]
    for name,prefixes in [('new_context_features',extra_prefixes),('severity',('severity.',)),('current_clinical_mentions',('current.',))]:
        x=[{k:v for k,v in d.items() if not k.startswith(prefixes)} for d in bv.features]
        s=new['model'].predict_proba(x)[:,1]
        masked.append(dict(masked_group=name,**measures(y,s,new['model'].threshold),mean_score_change=float(np.mean(s-sb)),new_fp_score_change=float(np.mean((s-sb)[newly_fp]))))
    pd.DataFrame(masked).to_csv(out/'ensemble_masking_diagnostics.csv',index=False)
    source=pd.read_csv(ROOT/'train.csv',usecols=['question_en'],keep_default_na=False)
    # Only index permitted validation rows; no test question is inspected or written.
    review=av[['source_record_index','case_id','emergency_label']].copy()
    review['question_en']=[source.iloc[i].question_en for i in review.source_record_index]
    review['original_score']=sa;review['updated_raw_score']=raw;review['updated_score']=sb
    review['original_prediction']=previous.astype(int);review['updated_prediction']=current.astype(int)
    review['error_group']=np.select([newly_fp,(y==0)&current&previous,(y==1)&~current],['new_false_positive','persistent_false_positive','false_negative'],default='other')
    review['changed_shared_features']=[json.dumps({k:[o.get(k),n.get(k)] for k in set(o)|set(n) if not k.startswith(extra_prefixes) and o.get(k)!=n.get(k)}) for o,n in zip(av.features,bv.features)]
    errors=review[review.error_group.ne('other')]
    errors.to_csv(out/'validation_error_cases.csv',index=False,encoding='utf-8-sig')
    packet=pd.concat([group.sample(min(12,len(group)),random_state=42) for _,group in errors.groupby('error_group')])
    packet['human_assessment']='';packet['notes']='';packet.to_csv(out/'human_review_packet.csv',index=False,encoding='utf-8-sig')
    result=dict(split_counts=allowed.groupby('split').size().to_dict(),validation_comparison=rows,
        newly_false_positive_count=int(newly_fp.sum()),linear_ablations=probe,component_diagnostics=components,
        source_sha256=file_hash(ROOT/'train.csv'),nlp_signature=extraction_signature(),
        model_sha256=file_hash(ARTIFACTS/'state_v2/model.joblib'),baseline_model_sha256=file_hash(ARTIFACTS/'tuned/best_model.joblib'),
        caveats=['No test examples or predictions were used.','Validation was already used for calibration/threshold selection; results are exploratory.',
                 'Labels are weak annotations. Error means disagreement with an annotation, not verified clinical error.',
                 'Feature associations and fixed-model masking do not establish causal effects. Linear probes are not the deployed ensemble.',
                 'Extractor, state, ensemble and deployment remain unchanged.'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    return result


def evaluate_nlp(reference_path=None,output_dir=None):
    """Score fixed manual references independently of urgency labels or ML scores."""
    out=Path(output_dir) if output_dir else ARTIFACTS/'nlp_evaluation';out.mkdir(parents=True,exist_ok=True)
    path=Path(reference_path) if reference_path else ARTIFACTS/'nlp_evaluation/manual_reference.jsonl'
    cases=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not cases or any(not isinstance(c.get('facts'),list) for c in cases):
        raise ValueError('Complete reference fact annotations before evaluation; unannotated is not an empty gold label.')
    rows=[];tp=fp=fn=span_tp=span_only_tp=span_only_fp=span_only_fn=0;attribute_rows=[];by_kind={}
    assert len({c['id'] for c in cases})==len(cases),'Duplicate reference IDs.'
    for case in cases:
        for g in case['facts']:
            assert case['text'][g['start']:g['end']]==g['surface'],'Invalid reference evidence span.'
        predicted=[f for f in extract_info(case['text'])['facts'] if f['kind'] in ('symptom','condition','medication','vital')]
        gold={(f['kind'],f['name']):f for f in case['facts']};actual={(f['kind'],f['name']):f for f in predicted}
        gold_spans={(f['kind'],f['start'],f['end']) for f in case['facts']}
        actual_spans={(f['kind'],f['start'],f['end']) for f in predicted}
        span_only_tp+=len(gold_spans&actual_spans);span_only_fp+=len(actual_spans-gold_spans);span_only_fn+=len(gold_spans-actual_spans)
        common=set(gold)&set(actual);tp+=len(common);fp+=len(set(actual)-set(gold));fn+=len(set(gold)-set(actual))
        for key in set(gold)|set(actual):
            g=gold.get(key);p=actual.get(key);matched=g is not None and p is not None
            exact=matched and (g['start'],g['end'])==(p['start'],p['end']);span_tp+=int(exact)
            counts=by_kind.setdefault(key[0],dict(tp=0,fp=0,fn=0));counts['tp' if matched else 'fn' if g else 'fp']+=1
            rows.append(dict(case_id=case['id'],category=case['category'],text=case['text'],kind=key[0],name=key[1],
                entity_match=matched,exact_span=exact,expected=json.dumps(g),actual=json.dumps(p,ensure_ascii=False)))
            if not g:continue
            # Name normalization is scored above; don't call a recognized synonym a
            # language miss when evaluating assertion/attributes at an exact typed span.
            if p is None:p=next((f for f in predicted if (f['kind'],f['start'],f['end'])==(g['kind'],g['start'],g['end'])),None)
            attribute_entity_found=p is not None
            for attr in ('status','temporality','experiencer','severity','onset','duration_seconds','medication_role','dose_value','value'):
                if attr not in g:continue
                if not p:value=None
                elif attr in ('severity','onset'):value=(p.get(attr) or {}).get('value')
                elif attr=='duration_seconds':value=(p.get('duration') or {}).get('seconds')
                elif attr=='dose_value':value=(p.get('dose') or {}).get('value')
                else:value=p.get(attr)
                expected=g[attr]
                if attr=='severity' and isinstance(expected,str) and expected.endswith('/10'):
                    expected=int(expected.split('/')[0])  # Numeric 8 and reported 8/10 are equivalent.
                correct=bool(np.isclose(value,expected)) if isinstance(value,(int,float)) and isinstance(expected,(int,float)) else value==expected
                attribute_rows.append(dict(case_id=case['id'],kind=key[0],name=key[1],attribute=attr,entity_found=attribute_entity_found,correct=correct,expected=json.dumps(g[attr]),actual=json.dumps(value)))
    def prf(t,p,n):
        precision=t/(t+p) if t+p else 0.;recall=t/(t+n) if t+n else 0.
        return dict(tp=t,fp=p,fn=n,precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.)
    attrs=pd.DataFrame(attribute_rows);summary=[]
    for name,group in attrs.groupby('attribute'):
        found=group[group.entity_found]
        summary.append(dict(attribute=name,reference_count=len(group),entity_found_count=len(found),
            correct_count=int(group.correct.sum()),end_to_end_accuracy=float(group.correct.mean()),
            accuracy_on_found_entities=float(found.correct.mean()) if len(found) else None))
    statuses=pd.Series([c.get('reference_status','unspecified') for c in cases]).value_counts().to_dict()
    result=dict(reference_cases=len(cases),reference_sha256=file_hash(path),nlp_signature=extraction_signature(),
        entity_identity=prf(tp,fp,fn),entity_exact_span_and_name=prf(span_tp,tp+fp-span_tp,tp+fn-span_tp),
        entity_span_detection=prf(span_only_tp,span_only_fp,span_only_fn),
        per_kind={k:prf(**{ 't':v['tp'],'p':v['fp'],'n':v['fn']}) for k,v in by_kind.items()},attributes=summary,
        reference_status='; '.join(sorted(statuses)),reference_status_counts=statuses,
        reference_reviewers=sorted({str(c['reviewer']) for c in cases if c.get('reviewer')}),
        caveats=['Not a clinician-validated clinical benchmark or independent human annotation.',
            'Entities are latest structured facts in four scored domains; other fact kinds are outside scope.',
            'Name identity and typed evidence-span detection are separate metrics. Attributes may align by exact typed span when names differ.',
            'Attribute accuracy on found entities excludes misses; end-to-end accuracy includes misses.',
            'Attribute rows cover specified reference attributes only; no claim of attribute precision on unspecified values.',
            'No urgency labels, doctor answers, original urgency metadata or ML outputs used.'])
    pd.DataFrame(rows).to_csv(out/'entity_results.csv',index=False,encoding='utf-8-sig');attrs.to_csv(out/'attribute_results.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'attribute_summary.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('NLP entity identity:',result['entity_identity'],flush=True)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    return result


def run_ensemble_ablation():
    """Requested full-architecture comparison; validation only, no deployment/HPO."""
    from sklearn.base import clone
    from threadpoolctl import threadpool_limits
    from triage_tuning import ScoreCalibrator,choose_threshold,score_diagnostics,THREADS
    out=ARTIFACTS/'ml_regression';out.mkdir(exist_ok=True)
    result_path=out/'ensemble_ablation.json'
    original=joblib.load(ARTIFACTS/'tuned/best_model.joblib')
    updated=joblib.load(ARTIFACTS/'state_v2/model.joblib')
    sig=extraction_signature();source=file_hash(ROOT/'train.csv')
    assert updated['feature_signature']==sig and source==updated['metrics']['source_sha256']
    provenance=dict(nlp_signature=sig,source_sha256=source,baseline_model_sha256=file_hash(ARTIFACTS/'tuned/best_model.joblib'),
                    updated_model_sha256=file_hash(ARTIFACTS/'state_v2/model.joblib'))
    if result_path.exists():
        result=json.loads(result_path.read_text())
        assert all(result[k]==v for k,v in provenance.items()),'Changed inputs require a new comparison directory.'
        return result
    default_hash=file_hash(ARTIFACTS/'triage_model.joblib')
    assignments=pd.read_csv(ARTIFACTS/'state_v2/data_splits.csv')
    allowed=assignments[assignments.split.ne('test')].drop_duplicates('case_id')
    ids=allowed.case_id.tolist()
    def load(path):
        cache=pd.read_parquet(path,filters=[('case_id','in',ids)])
        frame=allowed.merge(cache.drop_duplicates('case_id')[['case_id','features_json']],on='case_id',validate='one_to_one')
        frame['features']=frame.features_json.map(json.loads)
        return frame
    old=load(ARTIFACTS/'nlp_features.parquet');new=load(ARTIFACTS/'state_v2/nlp_features.parquet')
    assert old.case_id.tolist()==new.case_id.tolist()
    detailed=('fact.','fact_time.','onset.','duration_log.','medication_role.')
    reduced=new.copy()
    reduced['features']=reduced.features.map(lambda d:{k:v for k,v in d.items() if not k.startswith(detailed) and k not in ('respiratory_rate_scaled','glucose_mgdl_scaled')})
    train=reduced[reduced.split.eq('train')]
    estimator=clone(updated['model'].model)
    print('Refitting same ensemble architecture without detailed fact features; unchanged hyperparameters.',flush=True)
    with threadpool_limits(limits=THREADS):estimator.fit(train.features.tolist(),train.emergency_label)
    joblib.dump(dict(model=estimator,provenance=provenance,excluded_prefixes=detailed,diagnostic_only=True),out/'reduced_features_ensemble.joblib',compress=3)
    rows=[];predictions=[];curves=[]
    for name,frame,model in [('old_features_old_classifier',old,original['model'].model),
                             ('new_features_same_architecture',new,updated['model'].model),
                             ('new_features_without_detailed_facts',reduced,estimator)]:
        calibration=frame[frame.split.eq('calibration_fit')];selection=frame[frame.split.eq('validation_selection')]
        cal_score=model.predict_proba(calibration.features.tolist())[:,1]
        raw=model.predict_proba(selection.features.tolist())[:,1]
        y=selection.emergency_label.to_numpy()
        # Keep the original deployed threshold as a benchmark; new mappings and
        # thresholds below use selection validation, never the inspected test.
        if name=='old_features_old_classifier':
            rows.append(dict(representation=name,mapping='original_deployed',**measures(y,raw,original['model'].threshold),**score_diagnostics(y,raw)))
        for method in ('none','sigmoid'):
            mapper=ScoreCalibrator(method).fit(cal_score,calibration.emergency_label)
            score=mapper.transform(raw);best,curve=choose_threshold(y,score)
            rows.append(dict(representation=name,mapping=method,**measures(y,score,best['threshold']),**score_diagnostics(y,score)))
            curve['representation']=name;curve['mapping']=method;curves.append(curve)
            predictions.append(pd.DataFrame(dict(case_id=selection.case_id,emergency_label=y,representation=name,mapping=method,raw_score=raw,mapped_score=score,prediction=(score>=best['threshold']).astype(int))))
    pd.DataFrame(rows).to_csv(out/'ensemble_ablation_validation.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(out/'ensemble_threshold_curves.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'ensemble_validation_scores.csv',index=False)
    result=dict(**provenance,split_counts=allowed.groupby('split').size().to_dict(),comparison=rows,
        same_architecture_and_hyperparameters=True,threshold_policy='Selection-validation maximum macro F1 among recall >=0.95 if attainable; exploratory, not final clinical policy.',
        removed_features=list(detailed)+['respiratory_rate_scaled','glucose_mgdl_scaled'],
        caveats=['No test rows fitted, predicted, inspected or scored.','Selection validation was used previously; this is exploratory, not independent.',
                 'Original and updated fitted ensembles and deployed model unchanged.','Sigmoid fits on separate calibration groups.',
                 'No optimization or decision/adaptive engine implemented.','Targets are weak annotations, not clinical ground truth.'])
    assert file_hash(ARTIFACTS/'triage_model.joblib')==default_hash
    assert file_hash(ARTIFACTS/'tuned/best_model.joblib')==provenance['baseline_model_sha256']
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    return result


def finalize_integration():
    """Refit only the existing ensemble; train/validation only; NEVER deploy."""
    from sklearn.base import clone
    from threadpoolctl import threadpool_limits
    from triage_tuning import ScoreCalibrator,StateEnsemble,choose_threshold,score_diagnostics,THREADS
    from triage_pipeline import to_features
    from ml_features import FEATURE_SCHEMA
    out=ARTIFACTS/'integration_v3';out.mkdir(exist_ok=True)
    preservation=json.loads((out/'preservation.json').read_text())
    signature=extraction_signature();source=file_hash(ROOT/'train.csv')
    assert source==preservation['source_sha256']
    result_path=out/'metrics.json'
    if result_path.exists():
        result=json.loads(result_path.read_text());assert result['nlp_signature']==signature
        assert file_hash(ARTIFACTS/'triage_model.joblib')==preservation['deployment_sha256_before']
        return result
    assignments=pd.read_csv(ARTIFACTS/'state_v2/data_splits.csv')
    frame=assignments[assignments.split.ne('test')].drop_duplicates('case_id').copy()
    text=pd.read_csv(ROOT/'train.csv',usecols=['question_en'],keep_default_na=False)
    frame['question_en']=[text.iloc[i].question_en for i in frame.source_record_index]
    cache=out/'clinical_features.parquet';manifest=out/'feature_cache.json'
    stamp=dict(nlp_signature=signature,source_sha256=source,feature_schema=FEATURE_SCHEMA)
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text())==stamp:
        data=pd.read_parquet(cache);assert data.case_id.tolist()==frame.case_id.tolist()
    else:
        records=[]
        for i,row in enumerate(frame.itertuples(),1):
            info=extract_info(row.question_en)
            records.append(dict(case_id=row.case_id,features_json=json.dumps(to_features(info)),clinical_info_json=json.dumps(info,ensure_ascii=False)))
            if i%5000==0:print('Clinical extraction / repaired projection:',i,'/',len(frame),flush=True)
        data=pd.DataFrame(records);data.to_parquet(cache,index=False);manifest.write_text(json.dumps(stamp))
    assert data.case_id.tolist()==frame.case_id.tolist()
    frame['features']=data.features_json.map(json.loads).tolist()
    assert all(isinstance(value,dict) for value in frame.features)
    frame[['source_record_index','case_id','emergency_label','split']].to_csv(out/'data_splits.csv',index=False)
    parts={name:frame[frame.split.eq(name)] for name in ('train','calibration_fit','validation_selection')}
    groups=[set(p.case_id) for p in parts.values()]
    assert all(not x&y for i,x in enumerate(groups) for y in groups[i+1:])
    original=joblib.load(ARTIFACTS/'tuned/best_model.joblib')
    model=clone(original['model'].model)
    print('Refitting preserved ensemble architecture and hyperparameters on repaired features; candidate only.',flush=True)
    with threadpool_limits(limits=THREADS):model.fit(parts['train'].features.tolist(),parts['train'].emergency_label)
    calibration=parts['calibration_fit'];selection=parts['validation_selection'];y=selection.emergency_label.to_numpy()
    cal_score=model.predict_proba(calibration.features.tolist())[:,1];raw=model.predict_proba(selection.features.tolist())[:,1]
    mappings={};rows=[];threshold_curves=[]
    for method in ('none','sigmoid'):
        mapper=ScoreCalibrator(method).fit(cal_score,calibration.emergency_label);mappings[method]=mapper
        score=mapper.transform(raw);best,curve=choose_threshold(y,score)
        rows.append(dict(mapping=method,**measures(y,score,best['threshold']),**score_diagnostics(y,score)))
        curve['mapping']=method;threshold_curves.append(curve)
    table=pd.DataFrame(rows).sort_values(['brier','log_loss']);table.to_csv(out/'calibration_validation.csv',index=False)
    chosen=table.iloc[0];method=chosen.mapping;wrapped=StateEnsemble(model,mappings[method],float(chosen.threshold))
    scores=wrapped.predict_proba(selection.features.tolist())[:,1]
    prediction=selection[['source_record_index','case_id','emergency_label']].copy()
    prediction['raw_score_emergency']=raw;prediction['score_emergency']=scores;prediction['score_non_emergency']=1-scores
    prediction['prediction']=wrapped.predict(selection.features.tolist());prediction.to_csv(out/'validation_predictions.csv',index=False)
    pd.concat(threshold_curves).to_csv(out/'threshold_curves.csv',index=False)
    previous=json.loads((ARTIFACTS/'ml_regression/ensemble_ablation.json').read_text())
    comparison=[dict(version=r['representation']+'/'+r['mapping'],**{k:v for k,v in r.items() if k not in ('representation','mapping')}) for r in previous['comparison'] if r['mapping'] in ('original_deployed','sigmoid')]
    comparison.extend(dict(version='repaired/'+r['mapping'],**{k:v for k,v in r.items() if k!='mapping'}) for r in rows)
    pd.DataFrame(comparison).to_csv(out/'validation_comparison.csv',index=False)
    result=dict(**stamp,calibration_method=method,threshold=float(chosen.threshold),validation_comparison=comparison,
        split_counts={n:len(p) for n,p in parts.items()},baseline_model_sha256=file_hash(ARTIFACTS/'tuned/best_model.joblib'),
        deployed_model_sha256=file_hash(ARTIFACTS/'triage_model.joblib'),same_ensemble_parameters=True,
        clinical_validation='Known-example NLP repair and software invariants only; not independent clinical validation.',
        score_reference='weak_binary_annotations',score_is_clinically_calibrated=False,candidate_only=True,
        threshold_reference='Experimental validation selection, not a HackCure decision-policy threshold.',
        holdout_note='No test cases extracted, fitted or evaluated in this run. Validation previously informed investigations; not a new independent holdout.')
    bundle=dict(model=wrapped,feature_signature=signature,feature_schema=FEATURE_SCHEMA,label_mapping={0:'non_emergency',1:'emergency'},
        calibration_method=method,candidate_only=True,metrics=result)
    joblib.dump(bundle,out/'candidate_model.joblib',compress=3)
    assert file_hash(ARTIFACTS/'triage_model.joblib')==preservation['deployment_sha256_before']
    assert file_hash(ARTIFACTS/'tuned/best_model.joblib')==preservation['original_model_sha256']
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)
    print('Candidate saved separately. Deployment unchanged.',flush=True)
    return result


if __name__=='__main__':
    evaluate_nlp()
    run_regression_audit()

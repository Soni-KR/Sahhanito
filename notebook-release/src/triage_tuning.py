"""Reproducible multi-model Bayesian tuning; all models consume extracted facts only."""
import json,time,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd,optuna
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix,roc_auc_score
from threadpoolctl import threadpool_limits
from triage_pipeline import ROOT,ARTIFACTS,extract_info,to_features,case_id,file_hash,extraction_signature

OUT=ARTIFACTS/'tuned';SEED=20261009;THREADS=4
FAMILIES=['logistic_regression','sgd','random_forest','extra_trees','hist_gradient_boosting','xgboost','lightgbm','catboost']

def dense_array(x):return x.toarray() if hasattr(x,'toarray') else x

class ThresholdModel(ClassifierMixin,BaseEstimator):
    """Fitted pipeline wrapper compatible with the existing predict_case() function."""
    def __init__(self,model,threshold=.5):self.model=model;self.threshold=threshold
    def predict_proba(self,x):return self.model.predict_proba(x)
    def predict(self,x):return (self.predict_proba(x)[:,1]>=self.threshold).astype('int8')
    @property
    def classes_(self):return np.array([0,1])

def sample_parameters(t,name):
    if name in ['logistic_regression','sgd']:
        p={'weight':t.suggest_categorical('weight',['none','balanced'])}
        if name=='logistic_regression':p.update(C=t.suggest_float('C',.01,100,log=True),penalty=t.suggest_categorical('penalty',['l1','l2']))
        else:p.update(alpha=t.suggest_float('alpha',1e-6,.01,log=True),l1_ratio=t.suggest_float('l1_ratio',.05,.95))
        return p
    choices=['none','balanced'] if name in ['random_forest','extra_trees','hist_gradient_boosting'] else ['none','balanced','sqrt']
    p={'weight':t.suggest_categorical('weight',choices)}
    if name in ['random_forest','extra_trees']:
        p.update(n_estimators=t.suggest_int('n_estimators',150,450,step=100),max_depth=t.suggest_categorical('max_depth',[None,12,24,40]),
            min_samples_leaf=t.suggest_int('min_samples_leaf',1,6),min_samples_split=t.suggest_int('min_samples_split',2,12),
            max_features=t.suggest_categorical('max_features',['sqrt',.25,.5,1.0]))
    elif name=='hist_gradient_boosting':
        p.update(max_iter=t.suggest_int('max_iter',150,550,step=100),learning_rate=t.suggest_float('learning_rate',.025,.2,log=True),
            max_leaf_nodes=t.suggest_categorical('max_leaf_nodes',[15,31,63]),min_samples_leaf=t.suggest_int('min_samples_leaf',10,80),l2_regularization=t.suggest_float('l2_regularization',.001,30,log=True))
    elif name=='catboost':
        p.update(iterations=t.suggest_int('iterations',200,600,step=100),depth=t.suggest_int('depth',4,7),learning_rate=t.suggest_float('learning_rate',.025,.2,log=True),
            l2_leaf_reg=t.suggest_float('l2_leaf_reg',1,20,log=True),random_strength=t.suggest_float('random_strength',.1,2))
    else:
        p.update(n_estimators=t.suggest_int('n_estimators',200,600,step=100),learning_rate=t.suggest_float('learning_rate',.025,.2,log=True),
            subsample=t.suggest_float('subsample',.65,1),colsample_bytree=t.suggest_float('colsample_bytree',.65,1),
            reg_alpha=t.suggest_float('reg_alpha',.0001,3,log=True),reg_lambda=t.suggest_float('reg_lambda',.01,30,log=True))
        if name=='xgboost':p.update(max_depth=t.suggest_int('max_depth',3,8),min_child_weight=t.suggest_float('min_child_weight',1,15,log=True))
        else:p.update(num_leaves=t.suggest_int('num_leaves',15,63),max_depth=t.suggest_categorical('max_depth',[-1,4,6,8]),min_child_samples=t.suggest_int('min_child_samples',10,80))
    return p

def classifier(name,parameters,y):
    p=dict(parameters);weight=p.pop('weight');ratio=float((np.asarray(y)==0).sum()/(np.asarray(y)==1).sum())
    cw=None if weight=='none' else 'balanced';positive=1 if weight=='none' else ratio if weight=='balanced' else np.sqrt(ratio)
    if name=='logistic_regression':return LogisticRegression(**p,class_weight=cw,solver='liblinear',max_iter=2000,random_state=SEED)
    if name=='sgd':return SGDClassifier(**p,class_weight=cw,loss='log_loss',penalty='elasticnet',max_iter=3000,tol=1e-4,random_state=SEED)
    if name in ['random_forest','extra_trees']:
        return (RandomForestClassifier if name=='random_forest' else ExtraTreesClassifier)(**p,class_weight=cw,n_jobs=THREADS,random_state=SEED)
    if name=='hist_gradient_boosting':return HistGradientBoostingClassifier(**p,class_weight=cw,max_bins=63,early_stopping=False,random_state=SEED)
    if name=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(**p,scale_pos_weight=positive,tree_method='hist',max_bin=64,n_jobs=THREADS,eval_metric='logloss',random_state=SEED,verbosity=0)
    if name=='lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**p,scale_pos_weight=positive,subsample_freq=1,max_bin=63,n_jobs=THREADS,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**p,auto_class_weights=None if weight=='none' else 'Balanced' if weight=='balanced' else 'SqrtBalanced',
        thread_count=THREADS,random_seed=SEED,verbose=False,allow_writing_files=False,loss_function='Logloss')

def pipeline(name,p,y):
    steps=[('vectorizer',DictVectorizer(sparse=True,dtype=np.float32))]
    if name=='hist_gradient_boosting':steps.append(('dense',FunctionTransformer(dense_array,accept_sparse=True)))
    return Pipeline(steps+[('classifier',classifier(name,p,y))])

def choose_threshold(y,score):
    rows=[]
    for threshold in np.linspace(.02,.98,193):
        prediction=(score>=threshold).astype(int)
        rows.append(dict(threshold=float(threshold),macro_f1=float(f1_score(y,prediction,average='macro',zero_division=0)),
            emergency_recall=float(recall_score(y,prediction,zero_division=0)),emergency_precision=float(precision_score(y,prediction,zero_division=0))))
    table=pd.DataFrame(rows);eligible=table[table.emergency_recall>=.95]
    if eligible.empty:eligible=table
    best=eligible.sort_values(['macro_f1','emergency_precision','emergency_recall'],ascending=False).iloc[0].to_dict()
    return best,table

def run_tuning(base_trials=12,refine_trials=20):
    OUT.mkdir(exist_ok=True)
    if (OUT/'metrics.json').exists():
        saved=json.loads((OUT/'metrics.json').read_text())
        assert saved['source_sha256']==file_hash(ROOT/'train.csv') and saved['nlp_signature']==extraction_signature(),'Data/NLP changed; use a new tuning output directory.'
        return saved
    data=pd.read_csv(ROOT/'train.csv',usecols=['question_en','urgency'],keep_default_na=False)
    data['case_id']=data.question_en.map(case_id);data['emergency_label']=data.urgency.map({'non_emergency':0,'emergency':1})
    cached=pd.read_parquet(ARTIFACTS/'nlp_features.parquet',columns=['case_id','features_json'])
    manifest=json.loads((ARTIFACTS/'feature_cache.json').read_text())
    assert manifest=={'source_hash':file_hash(ROOT/'train.csv'),'nlp_signature':extraction_signature()}
    assert data.case_id.tolist()==cached.case_id.tolist()
    data['features']=cached.features_json.map(json.loads);data['source_record_index']=np.arange(len(data))
    unique=data.drop_duplicates('case_id')
    train,rest=train_test_split(unique,test_size=.30,stratify=unique.emergency_label,random_state=SEED)
    validation,test=train_test_split(rest,test_size=.5,stratify=rest.emergency_label,random_state=SEED)
    assert not(set(train.case_id)&set(validation.case_id) or set(train.case_id)&set(test.case_id) or set(validation.case_id)&set(test.case_id))
    splitmap={g:name for name,part in [('train',train),('validation',validation),('test',test)] for g in part.case_id}
    data['split']=data.case_id.map(splitmap);data[['source_record_index','case_id','emergency_label','split']].to_csv(OUT/'data_splits.csv',index=False)
    provenance={'source_hash':file_hash(ROOT/'train.csv'),'nlp_signature':extraction_signature(),'tuning_code_hash':file_hash(Path(__file__)),'seed':SEED,'base_trials':base_trials,'refine_trials':refine_trials}
    runfile=OUT/'run_manifest.json'
    if runfile.exists():assert json.loads(runfile.read_text())==provenance,'Different search inputs; use a separate output folder.'
    else:runfile.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    X=train.features.tolist();y=train.emergency_label.to_numpy();folds=[]
    for fit_idx,eval_idx in StratifiedKFold(3,shuffle=True,random_state=SEED).split(X,y):
        encoder=DictVectorizer(sparse=True,dtype=np.float32)
        folds.append(dict(x=encoder.fit_transform([X[i] for i in fit_idx]),v=encoder.transform([X[i] for i in eval_idx]),y=y[fit_idx],truth=y[eval_idx]))
    optuna.logging.set_verbosity(optuna.logging.WARNING);studies={};start=time.perf_counter()
    def search(name,budget):
        study=studies.setdefault(name,optuna.create_study(study_name=name,direction='maximize',storage='sqlite:///'+str((OUT/'search.sqlite3').resolve()),
            load_if_exists=True,sampler=optuna.samplers.TPESampler(seed=SEED+FAMILIES.index(name),n_startup_trials=6),pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=1)))
        def objective(trial):
            p=sample_parameters(trial,name);scores=[]
            for i,fold in enumerate(folds):
                estimator=classifier(name,p,fold['y']);a,b=fold['x'],fold['v']
                if name=='hist_gradient_boosting':
                    if 'dense_x' not in fold:fold['dense_x']=a.toarray();fold['dense_v']=b.toarray()
                    a,b=fold['dense_x'],fold['dense_v']
                with threadpool_limits(limits=THREADS):estimator.fit(a,fold['y']);score=estimator.predict_proba(b)[:,1]
                scores.append(float(average_precision_score(fold['truth'],score)))
                trial.report(float(np.mean(scores)),i)
                if trial.should_prune():raise optuna.TrialPruned()
            trial.set_user_attr('fold_scores',scores);trial.set_user_attr('std_ap',float(np.std(scores)))
            return float(np.mean(scores))
        remaining=max(0,budget-sum(t.state!=optuna.trial.TrialState.FAIL for t in study.trials));print(f'Tuning {name}: {remaining} additional trials',flush=True)
        study.optimize(objective,n_trials=remaining,gc_after_trial=True)
        study.trials_dataframe().to_csv(OUT/(name+'_trials.csv'),index=False)
        print(f'{name}: best 3-fold average precision {study.best_value:.4f} ({len(study.trials)} trials)',flush=True)
    for name in FAMILIES:search(name,base_trials)
    strongest=sorted(FAMILIES,key=lambda name:studies[name].best_value,reverse=True)[:3]
    print('Additional tuning for:',strongest,flush=True)
    for name in strongest:search(name,base_trials+refine_trials)
    models={};scores={};leaderboard=[];best_parameters={}
    for name in FAMILIES:
        study=studies[name];parameters=study.best_params;best_parameters[name]=parameters
        model=pipeline(name,parameters,y)
        with threadpool_limits(limits=THREADS):model.fit(X,y);score=model.predict_proba(validation.features.tolist())[:,1]
        models[name]=model;scores[name]=score;best,curve=choose_threshold(validation.emergency_label,score)
        curve.to_csv(OUT/(name+'_thresholds.csv'),index=False)
        leaderboard.append(dict(model=name,cv_average_precision=study.best_value,cv_std=study.best_trial.user_attrs['std_ap'],trials=len(study.trials),**best))
    ranking=pd.DataFrame(leaderboard).sort_values('macro_f1',ascending=False)
    top=ranking.model.tolist()[:3]
    # Compare compact soft-voting alternatives on validation only.
    for members,weights in [(top[:2],[.25,.75]),(top[:2],[.5,.5]),(top[:2],[.75,.25]),(top,[1,1,1])]:
        name='vote_'+ '_'.join(members)+'_'+ '_'.join(str(w) for w in weights)
        score=np.average([scores[m] for m in members],axis=0,weights=weights)
        best,curve=choose_threshold(validation.emergency_label,score)
        vote=VotingClassifier([(m,pipeline(m,best_parameters[m],y)) for m in members],voting='soft',weights=weights,n_jobs=1)
        models[name]=vote;leaderboard.append(dict(model=name,cv_average_precision=None,cv_std=None,trials=0,**best))
    board=pd.DataFrame(leaderboard).sort_values(['macro_f1','emergency_precision'],ascending=False).reset_index(drop=True)
    board.to_csv(OUT/'leaderboard.csv',index=False)
    winner=board.iloc[0];chosen=models[winner.model]
    # Preserve validation-tuned score distribution; do not refit on calibration data.
    if str(winner.model).startswith('vote_'):
        with threadpool_limits(limits=THREADS):chosen.fit(X,y)
    wrapped=ThresholdModel(chosen,float(winner.threshold))
    test_score=wrapped.predict_proba(test.features.tolist())[:,1];test_prediction=wrapped.predict(test.features.tolist())
    report=classification_report(test.emergency_label,test_prediction,labels=[0,1],target_names=['non_emergency','emergency'],output_dict=True,zero_division=0)
    matrix=confusion_matrix(test.emergency_label,test_prediction,labels=[0,1]).tolist()
    winner_info={k:None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for k,v in winner.to_dict().items()}
    metrics=dict(selected_model=winner.model,threshold=float(winner.threshold),validation=winner_info,test=dict(classification_report=report,confusion_matrix=matrix,
        roc_auc=float(roc_auc_score(test.emergency_label,test_score)),average_precision=float(average_precision_score(test.emergency_label,test_score))),
        search_trials=sum(len(study.trials) for study in studies.values()),completed_trials=sum(sum(t.state==optuna.trial.TrialState.COMPLETE for t in s.trials) for s in studies.values()),
        failed_trials=sum(sum(t.state==optuna.trial.TrialState.FAIL for t in s.trials) for s in studies.values()),
        model_families=FAMILIES,refined_families=strongest,best_parameters=best_parameters,seed=SEED,
        split_counts={name:len(part) for name,part in [('train',train),('validation',validation),('test',test)]},
        source_sha256=provenance['source_hash'],nlp_signature=extraction_signature(),runtime_seconds=time.perf_counter()-start,
        evaluation_reference='Current rule/policy annotations; not independently verified clinical ground truth.',
        holdout_note='New split for this tuning experiment. The dataset was used in earlier experiments; this is not an external or previously unseen clinical cohort.')
    bundle=dict(model=wrapped,feature_signature=extraction_signature(),label_mapping={0:'non_emergency',1:'emergency'},metrics=metrics)
    joblib.dump(bundle,OUT/'best_model.joblib',compress=3)
    prediction_frame=test[['source_record_index','case_id','question_en','emergency_label']].copy();prediction_frame['model_prediction']=test_prediction;prediction_frame['model_score_emergency']=test_score
    prediction_frame.to_csv(OUT/'test_predictions.csv',index=False,encoding='utf-8-sig')
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (OUT/'report.md').write_text(make_report(metrics,board),encoding='utf-8')
    backup=ARTIFACTS/'baseline_model.joblib'
    if not backup.exists():joblib.dump(joblib.load(ARTIFACTS/'triage_model.joblib'),backup,compress=3)
    joblib.dump(bundle,ARTIFACTS/'triage_model.joblib',compress=3)
    assert file_hash(ROOT/'train.csv')==provenance['source_hash']
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics

def make_report(m,board):
    r=m['test']['classification_report']
    return f'''# Tuned NLP-feature classifier

Best tested model: {m['selected_model']}. Validation-selected binary threshold: {m['threshold']:.3f}.

Eight model families, {m['search_trials']} Optuna trials, three-fold cross-validation, and additional trials on the three strongest families. Cross-validation optimizes average precision, with median pruning after fold scores. Hyperparameters cover depth/leaves, tree counts, learning rate, regularization, sampling and class balancing as applicable. Per-fold feature vocabularies are fitted only on that fold's training data. No labels or audit rules enter NLP features.

Validation chooses the highest macro F1 threshold among those reaching 95% emergency recall, and compares four small soft-voting combinations. This recall target is an experiment setting, not a clinical guarantee. The selected pipeline stays fitted on training cases; validation cases are reserved for threshold/model selection. The held-out test is evaluated only after selection. Duplicate question groups are disjoint; each case is fitted/scored once.

Held-out label agreement: accuracy {r['accuracy']:.4f}; macro F1 {r['macro avg']['f1-score']:.4f}; emergency precision {r['emergency']['precision']:.4f}; emergency recall {r['emergency']['recall']:.4f}. Confusion matrix [rows actual 0/1, columns predicted 0/1]: {m['test']['confusion_matrix']}.

These are agreement metrics against preliminary rule/policy labels. Related NLP patterns contributed to those labels, so high scores do not establish clinical reliability. {m['holdout_note']} No external clinical accuracy claim is made. The previous experiment uses a different split, so its test score is not a like-for-like comparison.

The best tested model replaces triage_artifacts/triage_model.joblib and remains compatible with predict_case(). The previous prototype is preserved as baseline_model.joblib. Complete studies are resumable in search.sqlite3; per-family trial histories, threshold curves, leaderboard, best parameters, split assignments and held-out predictions are saved here. No model can be guaranteed globally best; this is the winner of the documented search. Raw train.csv is unchanged. Only English NLP-extracted information is used. Adaptive questions remain deferred.

Source guidance for search parameters: [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html), [CatBoost](https://catboost.ai/docs/en/concepts/parameter-tuning).

Validation leaderboard:

{board.to_string(index=False)}
'''

class ScoreCalibrator:
    """Post-hoc score mapping fitted on held-out weak labels, never medical risk."""
    def __init__(self,method='none'):self.method=method;self.estimator=None
    def fit(self,score,y):
        if self.method=='sigmoid':
            self.estimator=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,random_state=SEED)
            self.estimator.fit(self._logit(score),y)
        elif self.method=='isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.estimator=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(score,y)
        elif self.method!='none':raise ValueError('Unknown calibration method.')
        return self
    @staticmethod
    def _logit(score):
        p=np.clip(np.asarray(score),1e-7,1-1e-7)
        return np.log(p/(1-p)).reshape(-1,1)
    def transform(self,score):
        if self.method=='sigmoid':return self.estimator.predict_proba(self._logit(score))[:,1]
        if self.method=='isotonic':return self.estimator.predict(score)
        return np.asarray(score)

class StateEnsemble(ThresholdModel):
    def __init__(self,model,calibrator,threshold=.5):
        super().__init__(model,threshold);self.calibrator=calibrator
    def predict_proba(self,x):
        p=self.calibrator.transform(self.model.predict_proba(x)[:,1])
        return np.column_stack([1-p,p])

def score_diagnostics(y,score):
    from sklearn.metrics import brier_score_loss,log_loss
    y=np.asarray(y);score=np.asarray(score);ece=0.
    bins=np.minimum((np.clip(score,0,1)*10).astype(int),9)
    for index in range(10):
        mask=bins==index
        if mask.any():ece+=mask.mean()*abs(score[mask].mean()-y[mask].mean())
    return dict(brier=float(brier_score_loss(y,score)),log_loss=float(log_loss(y,np.column_stack([1-score,score]),labels=[0,1])),ece_10_bins=float(ece))

def refit_state_ensemble():
    """Reuse the original ensemble and hyperparameters; refit only for changed NLP.

    Calibration-fit and calibration/threshold-selection groups are disjoint. The
    original test groups are reused for a paired research comparison, not a new
    independent clinical test. No HPO or label regeneration is performed.
    """
    from sklearn.base import clone
    from triage_pipeline import prepare_data,evaluate
    destination=ARTIFACTS/'state_v2';destination.mkdir(exist_ok=True)
    baseline_path=OUT/'best_model.joblib';baseline=joblib.load(baseline_path)
    signature=extraction_signature();source=file_hash(ROOT/'train.csv')
    assert source==baseline['metrics']['source_sha256'],'Source labels/data changed.'
    results_path=destination/'metrics.json'
    if results_path.exists():
        results=json.loads(results_path.read_text())
        assert results['nlp_signature']==signature and results['source_sha256']==source,'Changed inputs require a new experiment directory.'
        saved=joblib.load(destination/'model.joblib')
        joblib.dump(saved,ARTIFACTS/'triage_model.joblib',compress=3)
        return results
    frame=prepare_data(output_dir=destination)[0]
    assignments=pd.read_csv(OUT/'data_splits.csv')
    assert assignments.case_id.tolist()==frame.case_id.tolist()
    frame['split']=assignments.split.to_numpy()
    unique=frame.drop_duplicates('case_id')
    train=unique[unique.split=='train'];validation=unique[unique.split=='validation'];test=unique[unique.split=='test']
    calibration,selection=train_test_split(validation,test_size=.5,stratify=validation.emergency_label,random_state=SEED+1)
    names=['train','calibration_fit','validation_selection','test'];parts=[train,calibration,selection,test]
    group_sets=[set(part.case_id) for part in parts]
    assert all(not(a&b) for i,a in enumerate(group_sets) for b in group_sets[i+1:])
    mapping={g:name for name,part in zip(names,parts) for g in part.case_id}
    frame['split']=frame.case_id.map(mapping)
    frame[['source_record_index','case_id','emergency_label','split']].to_csv(destination/'data_splits.csv',index=False)
    original=baseline['model'].model
    model=clone(original)
    print('Refitting the existing CatBoost + histogram boosting + Extra Trees ensemble; unchanged hyperparameters.',flush=True)
    with threadpool_limits(limits=THREADS):model.fit(train.features.tolist(),train.emergency_label)
    cal_score=model.predict_proba(calibration.features.tolist())[:,1]
    val_score=model.predict_proba(selection.features.tolist())[:,1]
    calibrators={};rows=[]
    for method in ['none','sigmoid','isotonic']:
        calibrator=ScoreCalibrator(method).fit(cal_score,calibration.emergency_label)
        calibrators[method]=calibrator;score=calibrator.transform(val_score)
        best,_=choose_threshold(selection.emergency_label,score)
        rows.append(dict(method=method,**score_diagnostics(selection.emergency_label,score),**best))
    table=pd.DataFrame(rows).sort_values(['brier','log_loss']);table.to_csv(destination/'calibration_validation.csv',index=False)
    chosen=table.iloc[0];method=chosen.method
    deployed=StateEnsemble(model,calibrators[method],float(chosen.threshold))
    uncal_row=table[table.method=='none'].iloc[0]
    uncal=ThresholdModel(model,float(uncal_row.threshold))
    updated_metrics=evaluate(deployed,test);uncal_metrics=evaluate(uncal,test)
    test_score=deployed.predict_proba(test.features.tolist())[:,1]
    raw_score=model.predict_proba(test.features.tolist())[:,1]
    before=pd.read_csv(OUT/'test_predictions.csv').set_index('case_id').loc[test.case_id]
    assert np.array_equal(before.emergency_label.to_numpy(),test.emergency_label.to_numpy())
    original_metrics=baseline['metrics']['test']
    comparison=[]
    for label,metrics,score in [('original_deployed',original_metrics,before.model_score_emergency.to_numpy()),('updated_uncalibrated',uncal_metrics,raw_score),('updated_selected',updated_metrics,test_score)]:
        report=metrics['classification_report']
        comparison.append(dict(version=label,accuracy=report['accuracy'],macro_f1=report['macro avg']['f1-score'],emergency_recall=report['emergency']['recall'],emergency_precision=report['emergency']['precision'],**score_diagnostics(test.emergency_label,score)))
    pd.DataFrame(comparison).to_csv(destination/'comparison_before_after.csv',index=False)
    changed=sum(a!=b for a,b in zip(frame.features,pd.read_parquet(ARTIFACTS/'nlp_features.parquet').features_json.map(json.loads)))
    results=dict(source_sha256=source,nlp_signature=signature,baseline_model_sha256=file_hash(baseline_path),training_code_sha256=file_hash(Path(__file__)),
        ensemble_members=[name for name,_ in model.estimators],hyperparameters_preserved=True,features_changed_records=int(changed),
        calibration_method=method,threshold=float(chosen.threshold),split_counts={name:len(part) for name,part in zip(names,parts)},
        calibration_validation=table.to_dict('records'),test=updated_metrics,uncalibrated_test=uncal_metrics,comparison=comparison,
        score_reference='Calibration is against weak binary annotations, not clinical emergency probabilities.',
        comparison_note='Paired comparison on the previous experiment test groups. These cases were inspected previously; this is not a fresh or external clinical holdout. NLP fixtures are manually specified language tests, not clinical ground truth.')
    bundle=dict(model=deployed,feature_signature=signature,label_mapping={0:'non_emergency',1:'emergency'},calibration_method=method,metrics=results)
    joblib.dump(bundle,destination/'model.joblib',compress=3)
    predictions=test[['source_record_index','case_id','emergency_label']].copy()
    predictions['raw_score_emergency']=raw_score;predictions['score_emergency']=test_score;predictions['score_non_emergency']=1-test_score
    predictions['model_prediction']=deployed.predict(test.features.tolist());predictions.to_csv(destination/'test_predictions.csv',index=False)
    results_path.write_text(json.dumps(results,indent=2),encoding='utf-8')
    # Original fitted ensemble and its feature cache remain intact under tuned/.
    joblib.dump(bundle,ARTIFACTS/'triage_model.joblib',compress=3)
    assert file_hash(ROOT/'train.csv')==source
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)
    print('Selected calibration:',method,'| threshold:',chosen.threshold,flush=True)
    return results

if __name__=='__main__':
    # Importable classes are required for portable joblib inference after CLI use.
    from triage_tuning import refit_state_ensemble as run
    run()

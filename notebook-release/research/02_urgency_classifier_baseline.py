# Preserved historical experiment; requires its original environment and data.

# %% [markdown]
# # HackCure: initial urgency ML classifier
# Train question-only TF-IDF + class-weighted logistic regression, compare with a majority baseline, select on validation macro-F1, then evaluate the test set once. English is the initial language. Answers, generated summaries, titles, specialties and supplied entities are excluded from features.
# 
# Labels are dataset urgency categories, not validated emergency triage decisions or ESI levels. Results measure reproduction of weak labels. [Dataset and license: CC BY-SA 4.0](https://huggingface.co/datasets/Williamsanderson/MedQAData-v2).

# %%
# Dependencies installed in the project .venv before execution.

# %%
from pathlib import Path
import hashlib, json, re
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
SEED = 42
OUT = Path.cwd() / "outputs"
OUT.mkdir(exist_ok=True)
DATASET = "Williamsanderson/MedQAData-v2"
# Resolve and record an immutable revision for this run.
REVISION = HfApi().dataset_info(DATASET).sha
parquet_path = hf_hub_download(DATASET, "all/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
df = pd.read_parquet(parquet_path)
print("Loaded rows:", len(df))
required = {"question_en", "question_fr", "question_darija", "urgency", "entities"}
assert required <= set(df), f"Missing columns: {required - set(df)}"
def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()
df = df[df.question_en.notna() & df.question_en.astype(str).str.strip().ne("")].copy()
df["case_id"] = df.question_en.map(lambda t: hashlib.sha256(normalize(t).encode()).hexdigest())
conflicts = df.groupby("case_id").urgency.nunique()
bad_ids = conflicts[conflicts > 1].index
df[df.case_id.isin(bad_ids)].to_json(OUT / "conflicting_cases.jsonl", orient="records", lines=True, force_ascii=False)
df = df[~df.case_id.isin(bad_ids)].drop_duplicates("case_id").reset_index(drop=True)
LABELS = ["low", "medium", "high", "critical"]
assert df.urgency.isin(LABELS).all()
train, holdout = train_test_split(df, test_size=.30, stratify=df.urgency, random_state=SEED)
valid, test = train_test_split(holdout, test_size=.50, stratify=holdout.urgency, random_state=SEED)
assert not (set(train.case_id) & set(test.case_id))
assert not (set(train.case_id) & set(valid.case_id))
assert not (set(valid.case_id) & set(test.case_id))
manifest = pd.concat([part[["case_id", "urgency"]].assign(split=name) for name, part in [("train", train), ("validation", valid), ("test", test)]])
manifest.to_csv(OUT / "case_splits.csv", index=False)
(OUT / "dataset_revision.json").write_text(json.dumps({"dataset":DATASET, "revision":REVISION, "seed":SEED}, indent=2))
display(pd.crosstab(manifest.split, manifest.urgency))

# %% [markdown]
# ## Leakage boundaries
# Exact normalized English question duplicates share a case ID and are deduplicated before splitting. Conflicting labels are quarantined. Translations remain on the same row. Near duplicates and synthetic templates may still cross splits: audit them before trusting the results. This corpus has no documented patient identifiers, so patient-level separation cannot be guaranteed.
# The NLP model is kept separate initially so classifier performance is not confounded by extraction errors.

# %%
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay, average_precision_score
import matplotlib.pyplot as plt
import joblib
X_train, y_train = train.question_en, train.urgency
X_valid, y_valid = valid.question_en, valid.urgency
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(np.zeros((len(train),1)), y_train)
dummy_pred = dummy.predict(np.zeros((len(valid),1)))
scores = [{"model":"majority", "validation_macro_f1":f1_score(y_valid,dummy_pred,labels=LABELS,average="macro",zero_division=0)}]
candidates = {}
for c in [.5, 2.0]:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.98, max_features=100000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED))])
    model.fit(X_train, y_train)
    name = f"tfidf_logistic_C={c}"
    candidates[name] = model
    scores.append({"model":name,"validation_macro_f1":f1_score(y_valid,model.predict(X_valid),labels=LABELS,average="macro",zero_division=0)})
leaderboard = pd.DataFrame(scores).sort_values("validation_macro_f1", ascending=False)
display(leaderboard)
best_name = max(candidates, key=lambda name: next(s["validation_macro_f1"] for s in scores if s["model"] == name))
best = candidates[best_name]
print("Selected trained model:", best_name)

# %%
# Run after freezing preprocessing and model selection. Do not tune on these results.
pred = best.predict(test.question_en)
prob = best.predict_proba(test.question_en)
report = classification_report(test.urgency, pred, labels=LABELS, output_dict=True, zero_division=0)
display(pd.DataFrame(report).T)
ConfusionMatrixDisplay(confusion_matrix(test.urgency,pred,labels=LABELS),display_labels=LABELS).plot(cmap="Blues")
plt.title("Held-out test: counts")
plt.show()
critical_idx = list(best.classes_).index("critical")
critical_ap = average_precision_score((test.urgency == "critical").astype(int), prob[:,critical_idx])
print("Critical average precision:", round(critical_ap,4))
print("Critical prevalence:", round(float((test.urgency == "critical").mean()),4))
# Bootstrap uncertainty for critical recall, conditional on true critical cases.
mask = test.urgency.to_numpy() == "critical"
hits = (pred[mask] == "critical").astype(float)
rng = np.random.default_rng(SEED)
boot = [rng.choice(hits,size=len(hits),replace=True).mean() for _ in range(2000)]
print("Critical recall bootstrap 95% interval:", np.quantile(boot,[.025,.975]))
results = test[["case_id","question_en","urgency"]].copy()
results["prediction"] = pred
for i, label in enumerate(best.classes_): results[f"score_{label}"] = prob[:,i]
results.to_csv(OUT / "urgency_test_predictions.csv",index=False)
results[results.urgency.ne(results.prediction)].to_csv(OUT / "urgency_error_review.csv",index=False)
joblib.dump(best,OUT / "urgency_tfidf_logistic.joblib")
metrics = {"selected_model":best_name,"dataset_revision":REVISION,"report":report,
           "critical_average_precision":float(critical_ap),"seed":SEED}
(OUT / "urgency_metrics.json").write_text(json.dumps(metrics,indent=2))
leaderboard.to_csv(OUT / "validation_leaderboard.csv",index=False)

# %%
def predict_research_urgency(question):
    if not isinstance(question,str) or not question.strip():
        raise ValueError("Provide a non-empty English patient question.")
    scores = best.predict_proba([question])[0]
    return {"label":str(best.classes_[scores.argmax()]),
            "model_scores":dict(zip(best.classes_,map(float,scores))),
            "status":"research_only_unvalidated"}
predict_research_urgency("I have chest pain and difficulty breathing.")

# %% [markdown]
# ## Interpretation and next steps
# Class weighting changes the objective; scores are not calibrated clinical probabilities. Review critical false negatives first, alongside high-urgency errors and insufficient-information cases. A majority classifier can obtain high accuracy because medium dominates; use macro-F1 and per-class recall instead.
# 
# Before multilingual training, audit translations and expand cases only after splitting. Before clinical evaluation, obtain independently reviewed labels and a Tunisian test set. Never map these four classes directly to ESI. The small number of critical test cases limits precision of estimates.

# %% [markdown]
# ## Measured run results
# Executed successfully: test macro-F1 0.8101; accuracy 93.09%; critical recall 34/52 (65.4%). Eighteen critical cases were missed. These are dataset-label agreement metrics, not clinically validated triage performance. See RESULTS.md.

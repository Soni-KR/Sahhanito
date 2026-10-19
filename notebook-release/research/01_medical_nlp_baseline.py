# Preserved historical experiment; requires its original environment and data.

# %% [markdown]
# # HackCure: initial medical NLP model
# English pretrained biomedical NER baseline, followed by a weak-label audit. This is an executable starting model, not a trained Tunisian Arabic extractor. French and Darija are retained for human annotation; do not feed them to this English model.
# 
# Research prototype only. Source annotations are weak labels. No clinical performance claims are made.
# 
# Sources: [MedQAData-v2](https://huggingface.co/datasets/Williamsanderson/MedQAData-v2), [NER model](https://huggingface.co/d4data/biomedical-ner-all), [AHD v5](https://data.mendeley.com/datasets/mgj29ndgrk/5). MedQAData is CC BY-SA 4.0; AHD is CC BY 4.0; the model lists Apache 2.0.

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

# %%
import subprocess, sys
import torch
from transformers import pipeline, AutoTokenizer
MODEL = "d4data/biomedical-ner-all"
MODEL_REVISION = HfApi().model_info(MODEL).sha
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
ner = pipeline("token-classification", model=MODEL, tokenizer=tokenizer,
               revision=MODEL_REVISION, aggregation_strategy="simple",
               stride=64, device=0 if torch.cuda.is_available() else -1)
def extract(text, language="en"):
    if language != "en":
        raise ValueError("This baseline supports English only; Arabic/French require a separate model and evaluation.")
    mentions = []
    for item in ner(text):
        start, end = int(item["start"]), int(item["end"])
        mentions.append({"text":text[start:end], "start":start, "end":end,
                         "type":item["entity_group"], "score":float(item["score"]),
                         "assertion":"unknown", "temporality":"unknown",
                         "experiencer":"unknown", "severity":"unknown"})
    return {"text":text, "language":language, "mentions":mentions}
examples = ["I have chest pain and difficulty breathing for 10 minutes.",
            "I do not have chest pain, but I am afraid I might have heart disease.",
            "My father had a stroke five years ago."]
for example in examples:
    print(json.dumps(extract(example), indent=2))

# %% [markdown]
# ## Development audit
# Run inference on 100 training cases. Match source entity strings against the question and answer to flag possible answer leakage. A missing literal match is only a review flag: paraphrases and normalization can also explain it. Do not interpret agreement with these weak labels as NER accuracy.
# Assertion, temporality and experiencer deliberately remain unknown until an evaluated component is added.

# %%
audit = train.sample(n=min(100, len(train)), random_state=SEED).copy()
def values(entities):
    if not isinstance(entities, dict): return []
    return [(key, str(v)) for key, vals in entities.items() if vals is not None
            for v in (vals if isinstance(vals, (list, tuple, np.ndarray)) else [vals])]
rows = []
for _, row in audit.iterrows():
    prediction = extract(row.question_en)
    flags = [{"type":key, "value":value,
              "in_question":normalize(value) in normalize(row.question_en),
              "in_answer":normalize(value) in normalize(row.get("answer_en", ""))}
             for key, value in values(row.entities) if normalize(value)]
    rows.append({"case_id":row.case_id, "prediction":prediction, "source_entities":flags})
with (OUT / "nlp_development_predictions.jsonl").open("w", encoding="utf-8") as f:
    for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
annotation = audit[["case_id", "question_en", "question_fr", "question_darija"]].copy()
annotation["gold_mentions_json"] = ""
annotation["reviewer_notes"] = ""
annotation.to_csv(OUT / "nlp_annotation_template.csv", index=False)
(OUT / "ner_model_revision.json").write_text(json.dumps({"model":MODEL,"revision":MODEL_REVISION},indent=2))
display(pd.DataFrame(rows).head())

# %% [markdown]
# ## Next training milestone
# Annotate character spans and context attributes on a development sample; independently annotate a held-out test sample and adjudicate disagreements. Measure exact-span entity precision/recall/F1 and assertion accuracy on matched spans. Add Tunisian Darija and French examples with separate per-language results before selecting a multilingual extractor.
# 
# AHD is optional follow-up data: use patient questions for annotation, keep translated versions in the same case split, and do not derive urgency from specialty categories. This notebook does not download the large AHD workbook.

# %% [markdown]
# ## Measured run results
# Executed successfully: 100 questions, 1,478 predicted mentions. No gold-span accuracy is available. Of 1,158 supplied entity values, 89 matched only the answer; manually review these flags. See RESULTS.md.

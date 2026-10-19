"""Local publication-notebook setup and concise fact display."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import pandas as pd

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def initialize(package):
    package = Path(package).resolve()
    expected_sources = read_json(package / 'source_manifest.json')
    for name, expected in expected_sources.items():
        assert hashlib.sha256((package/'src'/name).read_bytes()).hexdigest() == expected, name
    work = package / '.notebook_runtime'
    artifacts = work / 'triage_artifacts'
    shutil.copytree(package/'artifacts', artifacts, dirs_exist_ok=True)
    for name, expected in read_json(package/'model_manifest.json').items():
        source = package / 'models' / Path(name).name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name
        destination = work / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    names = {Path(p).stem for p in expected_sources}
    for name in list(sys.modules):
        if name in names or name == 'labeling_binary' or name.startswith('labeling_binary.'):
            del sys.modules[name]
    import triage_pipeline
    triage_pipeline.ARTIFACTS = artifacts
    return artifacts

def facts_frame(facts):
    columns = ['text', 'kind', 'name', 'status', 'temporality', 'experiencer', 'severity']
    return pd.DataFrame([{key: fact.get(key) for key in columns} for fact in facts], columns=columns)

def prediction_row(result):
    keys = ['emergency_label', 'model_score_emergency', 'model_score_non_emergency',
            'classification_threshold', 'model_version', 'model_candidate_only']
    return {key: result[key] for key in keys if key in result}

# HackCure publication notebooks

Start with [NLP extractor and trials](01_nlp_extractor_trials.ipynb), then [final ML models](02_final_ml_model.ipynb). Both notebooks follow the development story with concise tables and a separate discussion of current inference.

| Folder | Contents |
|---|---|
| `src/` | Original extraction/runtime/training modules and a small notebook helper |
| `artifacts/` | Saved JSON references, metrics, frozen preprocessing and score tables |
| `models/` | Unchanged original English ensemble and experimental v3 candidate |
| `research/` | Historical trial scripts, initial baseline exports and phase-1 snapshots |

## Run and push

Use Python 3.12, install `pip install -r requirements.txt`, restart the notebook kernel and **Run All**. Keep supporting folders beside the notebooks. Default execution needs no downloads, fitting or training dataset. Writable outputs go to the ignored `.notebook_runtime/` folder.

Push this whole folder, including the supporting files and both models. Encoded data blobs and source dumps have been removed from the notebooks. Large transformer caches, training data and local environments are excluded.

Historical metrics retain their provenance and original split. Current inference/reference checks are labelled separately. The candidate remains experimental; the original deployment was not replaced. Full fitting needs the original data/caches/splits. Source/model hashes are checked before inference. Only load trusted project joblib files.

See [research/README.md](research/README.md) for historical rerun requirements.

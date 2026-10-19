# Current full-fact evaluation

Phase 2 retains aligned clinical dictionaries/context as the default. Optional FP32 GLiNER proposals are implemented in the existing multilingual_nlp.py and process_turn(semantic=...). Model-only interpretations remain uncertain, literal facts own overlapping spans, and proposed facts never enter ML features until a patient confirms them. Known critical flags return before transformer/ML loading. The ML bundle, original benchmark, deployed model, data and ML projection are unchanged. Scores/threshold remain experimental.

hybrid_reference.jsonl was written before implementation: 12 synthetic development cases and 24 fresh synthetic challenge cases (31 clinical entities). These assistant-authored examples lack native/clinical review. Exact spans, overlapping complete canonical facts (including annotated assertion, time, owner and numerical/medication attributes), unsupported spans and critical symptoms were checked. Attribute accuracy is conditional on matched entities, not end-to-end recall; unknown/missing outputs count as misses in complete facts. No reported number is clinical accuracy. Challenge failures were recorded without adding their vocabulary to dictionaries.

| Configuration | Exact entity F1 | Complete-fact F1 | Critical symptom recall | Unsupported span rate | Median extraction ms |
|---|---:|---:|---:|---:|---:|
| Phase-1 source | .915 | .814 | 1/3 | 0 | .45 |
| Improved dictionaries (default) | .933 | .867 | 1/3 | 0 | .45 |
| GLiNER INT8 native/context | 0 | 0 | 0/3 | 0 (no output) | 47.6 |
| MiniLM span retrieval/context | .255 | .218 | 1/3 | .759 | 32.1 |
| GLiNER FP32 native/context | .488 | .439 | 0/3 | 0 | 394.8 |
| Dictionaries + uncertain FP32 proposals | .933 | .867 | 1/3 | 0 | 380.9 |

The full comparable fact outputs are in hybrid_final.json. MiniLM used word-span retrieval with fixed development similarity/margin, followed by the same clinical context; it has no trained NER head. It is no longer compared using whole-question word accuracy. GLiNER uses 22 explicit canonical concept labels; this prompt inventory is not a trained clinical head. Its INT8 export failed an obvious chest-pain/asthma smoke control whereas FP32 with identical preprocessing detected both. This is a measured export/configuration result, not proof that all quantization or GLiNER models fail. Tokenizer IDs match the upstream HF fast tokenizer, and word masks/span enumeration follow upstream GLiNER 0.2.21. FP32 uses about 1.5 GB process RSS and weights occupy 1.157 GB; no GPU is available. No evaluated transformer improved the complete-fact challenge, so none is enabled by default. Timing is local CPU wall time and varies with load.

Two unfamiliar respiratory descriptions were missed, confirmed critical recall 1/3. Remaining uncertainty/actor/time failures are retained in the full output for future independent review. Gaps are explicit; missing facts do not silently imply non-emergency. The default critical API returned the equivalent chest-pain + breathlessness scenarios in 2–5 ms; independent ML afterwards took about 72–120 ms with the model loaded.

Research choices and primary sources:

- [GLiNER multilingual](https://huggingface.co/urchade/gliner_multi-v2.1): mDeBERTa general span NER; Apache-2.0. Upstream reports synthetic teacher data; not a Tunisian clinical head. [Pinned ONNX export](https://huggingface.co/onnx-community/gliner_multi-v2.1), revision 6ddaeb9413b0e71ad8457da1aab378a165b24058. Optional setup downloads only one FP32 checkpoint/tokenizer and verifies hashes; no custom model code.
- [TunBERT](https://huggingface.co/tunis-ai/TunBERT): Tunisian sentiment checkpoint, MIT; [original research](https://arxiv.org/abs/2111.13138) does not make this checkpoint a clinical extractor.
- [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base): general 100-language masked language encoder, MIT; needs annotated extraction adaptation.
- [Arabic vioBERT-v3](https://huggingface.co/Vionex-digital/vioBERT-v3): Arabic medical domain adaptation, Apache-2.0, released masked-language head. Its domain alone cannot supply symptoms/context labels.
- [OpenMed Disease Multi](https://huggingface.co/OpenMed/OpenMed-ZeroShot-NER-Disease-Multi-209M): biomedical disease NER, Apache-2.0; entity/domain coverage does not establish Tunisian symptom/time/negation recognition. Investigated, not locally scored.
- [ArSyra medical](https://huggingface.co/datasets/ArSyra/arsyra-medical): gated native medical dialect material, CC-BY-NC-SA-4.0, not full clinical span/context annotations. [Arabic disease corpus](https://www.mdpi.com/2306-5729/5/3/60): disease mentions in biomedical articles, not Tunisian patient symptom/state ground truth. MedQA weak binary labels and doctor answers were not reused as NER gold.

Actual tokenizer comparison (tokenization.json), nine requested mixed-language examples: no UNK tokens for any tested tokenizer. Mean subwords per space-separated word: TunBERT 2.51, XLM-R 2.02, vioBERT 2.37, GLiNER/mDeBERTa 2.14. Fragmentation/zero UNKs do not establish language understanding; no extraction claims are inferred from them. Model revisions and raw token sequences are retained. Fine-tuning is technically possible on CPU but unjustified without task-matched annotated data; no training dataset was invented.

english_validation.json: same 2,546 selection-validation weak annotations, frozen ensemble/calibration/0.15 threshold; macro F1 .9113, FP 131, FN 26. Original benchmark .9494. Context-feature regression remains unresolved. No fit, threshold search, clinical probability claim or held-out ML test tuning. demonstrations.json records real outputs for four equivalent multilingual conversations and an ambiguous complaint; features, independent rule findings and separately obtained scores agree. Notebook replays these artifacts without fitting/downloading.

Run the optional full-fact development experiment from the project root with existing research_requirements.txt. Frozen phase-1 source comes from the verified GitHub ZIP, not mutable runtime files. Published comparison outputs record each candidate's actual one-time challenge evaluation; do not tune against the challenge cases. The new FP32 model is downloaded with multilingual_nlp.py --download-semantic --cache-dir triage_artifacts/multilingual/research_cache. Old word-retrieval research is historical context below.

---

# Multilingual integration evidence

Runtime is the existing clinical-state/ML pipeline plus three modules: aligned phrase normalization, adaptive questions and independent clinical screens. No generative model, classifier fitting, final fusion or deployment occurs.

## Choice and research

**Selected:** finite clinical phrase normalization + existing bounded clinical context, with original-character alignment. This keeps explicit evidence, actor/time/assertion distinctions, clinical normalization and local inference auditable. It is a limited-vocabulary prototype; unknown complaints need clarification.

**Actually tested:** [multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, publisher's ONNX quint8 AVX2 export, CPU with two threads, mean pooling, fixed exploratory cosine cutoff 0.45. This is sentence-to-concept retrieval, not a fine-tuned clinical NER comparison. Standard Transformers import was rejected by Windows Application Control for `regex._regex`; the working ONNX runtime/tokenizer were used without changing security settings.

| Approach | Known development concepts | Extra spelling/idiom stress | Mean local latency |
|---|---:|---:|---:|
| Frozen previous English extractor | 14/44 | 1/10 | ~0.18 ms |
| Aligned clinical phrases | 44/44 | 3/10 | ~0.35 ms |
| Multilingual MiniLM retrieval | 23/44 | 4/10 | ~2.28 ms |

The phrase vocabulary was developed using known examples; that 44/44 result is optimistic. Ten later stress examples were not used to expand it or tune the encoder. All fixtures are assistant-authored, pending independent Tunisian native-speaker/clinical review. The encoder recovered one more stress case but did not establish reliable clinical spans, assertion, timing or actors. Its ~128 MB cache and optional dependencies are unnecessary for runtime. These results do not rule out better fine-tuned transformers with appropriate annotated data.

Other investigated approaches, not run or scored locally:

- [TunBERT paper](https://arxiv.org/abs/2111.13138): Tunisian pretrained representations with sentiment, dialect and reading-comprehension evaluations. A clinical extraction head/reference corpus would still be required.
- [ANER](https://huggingface.co/boda/ANER): Arabic fine-grained named entities evaluated on Wikipedia/news; those results do not establish Tunisian clinical extraction performance.
- [MoRoBERTa](https://huggingface.co/AchrafSoltani/MoRoBERTa): Moroccan Arabizi PER/LOC/ORG/MISC recognition, with transliterated and LLM-teacher silver training data. Different dialect and entity task; not selected for clinical normalization.

Question selection uses conditional missing slots, with complaint relevance and deterministic priorities. [Rasa's dynamic required-slot documentation](https://legacy-docs-oss.rasa.com/docs/rasa/forms/) illustrates the underlying strategy; no Rasa dependency/framework is used. Our priorities and translations are draft choices requiring clinical review.

## Rule provenance

Rules retain their own sources and original conversation evidence. These are selected screens, **not the complete protocols** and not remotely validated triage.

- [NHS chest pain](https://www.nhs.uk/symptoms/chest-pain/): current chest pain with breathlessness triggers an immediate screen.
- [NHS breathing guidance](https://www.nhs.uk/symptoms/shortness-of-breath/): explicit severe/functional respiratory impairment. An intensity word alone does not establish inability to speak; a specific screening question is available.
- [NHS stroke guidance](https://www.nhs.uk/conditions/stroke/symptoms/): sudden focal signs and reported events within 24 hours, including resolved signs. Unknown timing is retained.
- [WHO IITT adult chart](https://cdn.who.int/media/docs/default-source/integrated-health-services-(ihs)/csy/iitt/iitt_adult.pdf?sfvrsn=b2a91431_1), applicable from age 12: selected red criteria and separate high-risk-vital criteria. High-risk vitals request clinician review and are not automatically treated as red criteria. Unknown age prevents the age-specific numeric rules from triggering.
- [Tunisian Ministry of Health SAMU information](https://santetunisie.rns.tn/fr/sante-en-tunisie/medecine-d-urgence/les-services-d-aide-medicale-urgente-samu): local emergency contact **190**.

Rules are evaluated before questions and ML each turn. ML failure does not erase clinical flags. No triggered flag is never a safety finding. No cross-patient combinations or final ML/rule decision are produced.

## Model compatibility and results

`compatibility.json` pins unchanged model/data hashes, the original saved v3 extraction signature, the unchanged `ml_features.py`, and the exact fact-to-info projection function/schema. The explicit adapter does not rewrite the saved model signature. Nine pinned English feature dictionaries are checked; multilingual equivalents and multi-turn cases are also tested. Representation compatibility is not clinical validation.

`english_validation.json` reevaluates the unchanged v3 candidate on 2,546 already-used selection-validation weak annotations, at the saved 0.15 sigmoid threshold: F1 0.9120 → 0.9108, FP 131 → 132, FN 25 → 26; two decisions changed. No fitting, threshold/calibration optimization or old test-question extraction occurred. The prior context-feature regression is still unresolved. Original benchmark and deployment remain byte-preserved.

`clinical_nlp_evaluation/` scores 15 multilingual development references separately from ML; `english_nlp_evaluation/` checks the 36 previous English development references. Entity and original-span checks passed, as did the explicitly annotated attributes. Their small, inspected references do not measure open-ended clinical performance or independently validated accuracy.

`demonstrations.json` contains actually executed equivalent Arabizi/English conversations, clinical facts, independent ML outputs and rule evidence. In both, chest pain for 20 minutes followed by affirmed breathing difficulty triggers `chest_dyspnea`, stops questions and exposes guidance. The experimental mapped emergency score is about 0.9971, raw score about 0.9938; neither is a clinical risk estimate.

## Reproduce

From the project root:

```powershell
& '.\hackcure-nlp-urgency\.venv\Scripts\python.exe' -m unittest discover -q
# Optional research-only dependencies; runtime uses requirements.txt instead.
& '.\hackcure-nlp-urgency\.venv\Scripts\python.exe' -m pip install -r triage_artifacts\multilingual\research_requirements.txt
& '.\hackcure-nlp-urgency\.venv\Scripts\python.exe' triage_artifacts\multilingual\experiment.py
```

The main notebook runs without downloads. The optional experiment downloads only the pinned publisher model/tokenizer inside this project and never enables remote custom model code. Its frozen English reference source is preserved under `archive/multilingual-2026-10-10/`.

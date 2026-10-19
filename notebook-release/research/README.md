# Preserved historical research

The baseline Python exports retain original notebook code and measured notes. `experiment.py`, `hybrid_experiment.py` and `finalize_integration.py` preserve the original trial implementations. `phase1/` retains pre-integration snapshots.

These archived scripts keep their original paths and are not run by publication setup. To rerun them, use their original copies in the full project layout with the required frozen snapshots, data and caches. Initial baselines additionally need their original environment/download setup. Optional research packages are in `../requirements-research.txt`.

Default notebook execution reads recorded research and runs current packaged extraction/inference without repeating downloads or fitting. Detailed protocols, model revisions, source links and limitations are in `../artifacts/multilingual/README.md`.

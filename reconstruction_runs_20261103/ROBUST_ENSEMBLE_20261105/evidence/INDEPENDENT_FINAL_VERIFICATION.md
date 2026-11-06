# Independent final validation audit

Recorded 2026-11-05T13:22:44.734166+08:00 (Hong Kong).

PASS: completed saved predictions cover all 1,422 images, 859 Studies and 228 patients. Original labels were not changed.
All ten AUCs, single/multiple-image slices and per-slice patient-equal metrics reproduce the final report by independent positive-negative pair enumeration.
Maximum AUC discrepancy: 5.55e-16. Image/Study fixed fusion and all five frozen selection gates agree. Class 1 is exactly preserved.
Candidate selected by frozen gates: True. Input hashes stayed unchanged.

This is a CPU-only implementation audit of the sole candidate. No alternative fusion, GPU inference, training, test access or independent-generalization claim.

Full evidence: `/home/featurize/work/reconstruction_runs_20261103/ROBUST_ENSEMBLE_20261105/evidence/independent_final_verification.json`.

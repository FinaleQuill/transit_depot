# Tight-crop A/B, AP early stopping

State: failed
Stage: train_B_base
PID: 32068
Updated UTC: 2026-11-02T08:07:40.504736+00:00
Elapsed: 44485.9 seconds

Order: A_tiny, A_base, B_tiny, B_base. Max 25 epochs; AP-only patience 5.
Old A_tiny E1 is migrated with optimizer, EMA, scaler and RNG state.
Historical 044 is separate; no test prediction or production timing claim.

- A_tiny: completed; epoch 7/25; best AP 0.7393485860304196; failure retry 0/1
- A_base: completed; epoch 6/25; best AP 0.7376266331013407; failure retry 0/1
- B_tiny: completed; epoch 7/25; best AP 0.7397691138114503; failure retry 0/1
- B_base: running; epoch 3/25; best AP 0.7364903493721539; failure retry 0/1

Error: RuntimeError: B_base has STOP marker; refusing automatic retry

Details: `earlystop25/status.json`; logs: `earlystop25/logs/`.

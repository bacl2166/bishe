# Three domains and generative experiment implementation plan

**Goal:** Run the existing controlled experiment and a genuine generative experiment on three Amazon Reviews 2023 domains with comparable evaluation budgets.

**Architecture:** Reuse the frozen catalog, histories, requests and encoding mappings from experiment one. Train a separate encoder-decoder model for each encoding and seed. Compare actual constrained beam results with exhaustive, teacher-forced path scores for a predetermined diagnostic subset, under original vocabulary probabilities and local feasible-child normalization.

**Requirements:** Musical Instruments, Amazon Fashion and Appliances; identical catalog/request/training caps; three independent seeds; all twelve existing encoding adapters; no test labels in model selection or request selection; EOS included and no length normalization; unknown attributes fail the corresponding predicate; missing coverage and simulated rules explicitly reported.

**Review focus:** Variable length paths; padding and EOS; masking changes the objective; nondiagnostic requests have null reference metrics; stale source artifacts and resume; effective dataset sizes may fall below identical caps.

- [x] Add three dataset configurations, sequential suite CLI and audit documentation. Validate matching caps and command dispatch.
- [x] Add the conditional generator with causal teacher forcing, validation-only checkpoint selection and checkpoint loading. Test causal/padding behavior and deterministic training.
- [x] Add neural prefix decoding and exhaustive scoring using exactly the same transition function. Test full-width equivalence, local reranking, EOS and strict node budgets.
- [x] Add experiment-two runner, diagnostic selection, explicit reference metrics and source fingerprints. Test missing/stale artifacts and nullable reference aggregation.
- [x] Run the full unit suite, offline end-to-end integration for all encodings, and a small real Amazon pilot. Review and fix material findings. See `docs/VALIDATION.md` for evidence.

The user has authorized implementation and submission to the existing repository. Execute in the existing clean checkout; keep datasets and generated results outside Git.

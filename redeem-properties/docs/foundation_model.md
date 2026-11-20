# Foundation model

## Purpose

The foundation model predicts three peptide properties:

- retention time (RT)
- collision cross section (CCS)
- MS2 fragment intensities

The model trains from scratch. It does not use teacher distillation.

## Architecture

A shared peptide encoder produces pair representations. Three task heads use these representations:

- RT head for intrinsic peptide retention time
- CCS head for mobility prediction
- MS2 head for fragment presence and intensity

The model supports these opt-in features:

- harmonized RT targets
- CCS length and mass features
- MS2 fragment tokens
- per-epoch checkpoints
- deterministic initialization and resume

The default settings keep the base model behavior.

## Training

The current recipe uses the research curriculum for 60 epochs:

- AdamW optimizer
- learning rate 2e-5
- 500 warm-up steps
- cosine decay to 0.1
- gradient clipping at 1.0
- three independent seeds

The training data and split remain fixed for model comparison.

## Validation

Validation uses one fixed scorecard and three seeds. The measured minimum detectable effects are:

| Task | MDE |
| --- | ---: |
| RT MAE | 0.58 |
| CCS MAE | 2.28 |
| MS2 cosine | 0.124 |

Lower is better for RT and CCS. Higher is better for MS2.

Current validation means:

| Model | RT MAE | CCS MAE | MS2 cosine |
| --- | ---: | ---: | ---: |
| Historical v0.52 | 3.319 | 11.24 | 0.909 |
| Unified, harmonized RT | **3.230** | 10.63 | **0.921** |
| Unified, CCS exposure and length/mass | 3.362 | **9.88** | 0.918 |
| Unified, length/mass only | 3.277 | 10.67 | **0.921** |

The harmonized-RT model is the current balanced model. The CCS-exposure model has the best CCS score. The trained checkpoints are stored outside Git.

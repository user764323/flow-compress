## Flow Compress

Flow Compress is a unified research framework for neural network compression. It combines
three complementary approaches into a single, consistent API and experiment workflow:

- **IDAP++ (pruning)**: divergence-aware pruning across width and depth.
- **FAAQ (quantization)**: flow-aware adaptive bit allocation with post-training quantization.
- **FAD (distillation)**: flow-aligned knowledge distillation with teacher-student alignment.

The goal is to enable modular experimentation with pruning, quantization, and distillation
under a shared “information flow” perspective.

## Overview

The framework is organized into three core modules:

- `flow_compress/pruning`: IDAP++ pruning implementation and utilities.
- `flow_compress/quantization`: FAAQ quantization implementation and utilities.
- `flow_compress/distillation`: FAD distillation implementation and utilities.

Shared helpers live in `flow_compress/utils`. Experiment runners in `experiments/` mirror the
quantization structure for distillation and pruning with batch runners and result visualizers.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, recommended for GPU acceleration)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Pruning (IDAP++)

```bash
python scripts/idap_prune.py
```

### Quantization (FAAQ)

```bash
python scripts/faaq_quantize.py --model resnet50 --dataset cifar10 --target-bits 4
```

### Distillation (FAD)

```bash
python scripts/fad_train.py
```

### Distillation Precompute (FAD)

```bash
python scripts/fad_precompute_teacher_flow.py
```

### Experiments

```bash
python experiments/quantization/run_all_experiments.py
python experiments/distillation/run_all_experiments.py --num-epochs 1
python experiments/pruning/run_all_experiments.py --iterations 1
```

### Python API

```python
from flow_compress.pruning.divergence_aware_pruning import divergence_aware_pruning
from flow_compress.quantization.faaq import FAAQQuantizer
from flow_compress.distillation.trainer.fad_trainer import FADTrainer
```

## Project Structure

```
flow_compress/
├── pruning/
├── quantization/
├── distillation/
└── utils/

experiments/
├── quantization/
├── distillation/
└── pruning/

scripts/
docs/
```

## Licence

This project is distributed under the MIT License. See `LICENSE`.

## References

- IDAP++: divergence-aware pruning with joint filter/layer optimization.
- FAAQ: flow-aware adaptive quantization with per-layer bit allocation.
- FAD: flow-aligned knowledge distillation with teacher-student divergence alignment.

Additional details and original artifacts are preserved under:
- `docs/pruning/README.md`
- `docs/quantization/README.md`
- `docs/distillation/README.md`

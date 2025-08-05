---
layout: page
title: Features
description: "Explore Forgather's key features: distributed pipeline parallelism, template inheritance, dynamic code generation, and consumer GPU support."
permalink: /features/
---

# Features

Forgather combines configuration-driven development with distributed training capabilities, making large model training accessible to hobbyists and researchers.

## 🚀 Distributed Pipeline Parallelism

Train models larger than your GPU memory by splitting them across multiple consumer GPUs.

### Consumer GPU Friendly

Perfect for hobbyists with multiple RTX 3080/4080 cards:

```bash
# Train a 7B parameter model across 4 RTX 3080s
fgcli.py -t large_model.yaml train -d "0,1,2,3" --pipeline
```

### Advanced Pipeline Schedules

**1F1B (One Forward, One Backward)**
- Balanced pipeline utilization
- Good for most use cases
- Memory efficient

**Zero-Bubble Pipeline**  
- Eliminates pipeline bubbles
- Maximum GPU utilization
- Best performance for large models

**Interleaved 1F1B**
- Reduces memory peaks
- Better for memory-constrained setups
- Flexible micro-batch scheduling

### Memory Optimization

```yaml
-- block pipeline_config
    num_stages: 4
    micro_batch_size: 2
    gradient_checkpointing: true
    activation_offloading: true
    mixed_precision: "bf16"
-- endblock
```

## 📝 Template Inheritance System

Eliminate configuration duplication with a powerful template system.

### Hierarchical Configuration

```yaml
# Base transformer template
-- extends 'types/models/transformer/base.yaml'
-- block model_config
    hidden_size: 768
    num_layers: 12
    num_heads: 12
-- endblock

# Experiment: larger model
-- extends 'base_transformer.yaml'  
-- block model_config
    == super()                     # Inherit parent settings
    hidden_size: 1024              # Override specific values
    num_layers: 24
-- endblock
```

### Modular Components

Mix and match different components:

```yaml
-- block attention_factory
# Switch attention mechanisms
attention_factory: !partial:models.attention.RoPEAttention
    rope_theta: 10000
    max_seq_len: 2048
-- endblock

-- block optimizer
# Compare optimizers
optimizer: !partial:torch.optim.AdamW
    lr: 1.0e-4
    weight_decay: 0.1
-- endblock
```

### Template Reusability

```yaml
# Shared component library
-- include 'components/optimizers/adamw.yaml'
-- include 'components/models/llama_base.yaml'
-- include 'components/datasets/tiny_stories.yaml'

-- block experiment_name
    name: "LLaMA-tiny with AdamW"
-- endblock
```

## 🔧 Dynamic Code Generation

Models are materialized as standalone Python code with no framework dependencies.

### Generated Model Structure

```
output_models/my_model/
├── tiny_causal_transformer.py    # Main model class
├── causal_multihead_attn.py      # Attention implementation
├── feedforward_layer.py          # Feed-forward networks
├── layer_stack.py                # Layer composition
├── init_weights.py               # Weight initialization
└── runs/                         # Training artifacts
    └── my_model_2025-01-15/
        ├── config.yaml           # Full config snapshot
        ├── model.safetensors     # Trained weights
        └── trainer_logs.json     # Training metrics
```

### Framework Independence

Generated models work without Forgather:

```python
# Use generated model independently
from output_models.my_model.tiny_causal_transformer import TinyTransformer

model = TinyTransformer()
output = model(input_ids)
```

### Configuration Inspection

```bash
# View preprocessing pipeline
fgcli.py -t config.yaml pp                    # Jinja2 → YAML
fgcli.py -t config.yaml graph --format yaml   # Node graph
fgcli.py -t config.yaml code --target model   # Generated Python
```

## 🎯 Advanced Training Features

### Multiple Trainer Types

**SimpleTrainer**: Fast single-GPU training
```yaml
trainer: !partial:forgather.ml.SimpleTrainer
    max_steps: 10000
    gradient_clip_val: 1.0
```

**AccelTrainer**: Multi-GPU with Accelerate
```yaml  
trainer: !partial:forgather.ml.AccelTrainer
    mixed_precision: "fp16"
    gradient_accumulation_steps: 4
```

**PipelineTrainer**: Pipeline parallelism
```yaml
trainer: !partial:forgather.ml.PipelineTrainer
    num_stages: 4
    schedule: "zero_bubble"
```

### Custom Optimizers

Built-in implementations of research optimizers:

- **AdamW**: Standard with weight decay
- **AdaFactor**: Memory-efficient adaptive optimizer
- **Apollo**: Adaptive coordinate-wise optimization
- **GaLore**: Gradient low-rank projection

```yaml
-- block optimizer  
# Research optimizer with custom settings
optimizer: !partial:forgather.ml.optim.Apollo
    lr: 1.0e-3
    eps: 1.0e-4
    approximation_rank: 10
-- endblock
```

### Automatic Checkpointing

```yaml
-- block callbacks
callbacks:
  - !partial:forgather.ml.callbacks.ModelCheckpoint
      monitor: "val_loss"
      save_top_k: 3
      mode: "min"
  - !partial:forgather.ml.callbacks.EarlyStopping
      monitor: "val_loss"
      patience: 10
-- endblock
```

## 🔬 Systematic Experimentation

### Component Comparison

Compare different architectural choices systematically:

```bash
# Compare attention mechanisms
cd examples/tiny_experiments/attention
fgcli.py ls
# → alibi.yaml, rope.yaml, multihead.yaml, singlehead.yaml

# Compare feedforward layers  
cd examples/tiny_experiments/feedforward_layers
fgcli.py ls
# → glu_ff.yaml, mlp.yaml, swish.yaml
```

### Hyperparameter Sweeps

```yaml
# Base configuration
-- set experiments = [
    {"lr": 1e-3, "batch_size": 32},
    {"lr": 1e-4, "batch_size": 64}, 
    {"lr": 5e-4, "batch_size": 48}
  ]

-- for exp in experiments
-- block experiment_{{ loop.index }}
    == super()
    learning_rate: {{ exp.lr }}
    batch_size: {{ exp.batch_size }}
-- endblock
-- endfor
```

### Reproducibility

Every training run captures:
- Complete configuration snapshot
- Generated model source code  
- Training metrics and logs
- Environment information
- Git commit hash

## 🛠 Developer Experience

### Interactive Development

```python
# In project_index.ipynb
from forgather.project import Project

proj = Project("my_experiment.yaml")

# Inspect configuration
config = proj.environment.load("my_experiment.yaml")
print(config.keys())

# Materialize components
model_factory, dataset = proj("model", "train_dataset")
model = model_factory()
```

### Rich CLI Interface

```bash
fgcli.py index                    # Project overview as markdown
fgcli.py ls                       # List all configurations
fgcli.py tlist                    # Template hierarchy  
fgcli.py -t config.yaml pp        # Show preprocessed config
fgcli.py -t config.yaml train     # Train with configuration
fgcli.py -t config.yaml tb        # Start Tensorboard
```

### Error Diagnostics

```bash
# Debug configuration issues
fgcli.py -t broken_config.yaml pp     # Shows Jinja2 errors
fgcli.py ls                           # "PARSE ERROR" for broken configs
fgcli.py -t config.yaml graph         # Validates node dependencies
```

## Performance Characteristics

### Memory Efficiency
- Gradient checkpointing
- Activation offloading  
- Mixed precision training
- Pipeline parallelism memory optimization

### Training Speed
- Optimized multi-GPU communication
- Zero-bubble pipeline schedules
- Custom CUDA kernels where available
- Efficient data loading pipelines

### Scalability
- Consumer GPU support (RTX 3080+)
- Linear scaling with pipeline stages
- Memory usage independent of model size (with pipeline parallelism)
- Automatic micro-batch size optimization

Ready to start training large models on your hardware? [Get Started →](/getting-started)
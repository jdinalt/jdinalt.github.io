---
layout: page
title: Examples
description: "Explore the potential of distributed training and template inheritance with Forgather's example configurations and use cases."
permalink: /examples/
---

# Examples & Use Cases

Discover what's possible with Forgather's template inheritance and distributed training capabilities.

## 🚀 Distributed Pipeline Parallelism

### Large Model Training on Consumer Hardware

Successfully training 7B parameter models across multiple RTX 4090 cards using various pipeline schedules:

- **Zero-Bubble Pipeline**: Maximizes GPU utilization with minimal idle time
- **Interleaved 1F1B**: Balanced approach with good memory efficiency  
- **Standard 1F1B**: Most memory-efficient option for resource-constrained setups

### Real Hardware Testing

Current development uses a 6x RTX 4090 setup, with 4 cards sufficient for 7B models. The system demonstrates that consumer hardware can handle serious model training when properly orchestrated.

## 📝 Template Inheritance Patterns

### Systematic Architecture Comparison

Compare different model architectures without configuration duplication:

**Attention Mechanisms:**
- Multi-head attention with different head counts
- Single-head attention variants
- ALiBi positional bias implementations
- Rotary Position Embedding (RoPE) systems
- Grouped Query Attention experiments

**Feedforward Variations:**
- Standard MLP architectures
- GLU (Gated Linear Unit) implementations  
- Different activation functions (ReLU, GELU, SwiGLU)

**Optimizer Studies:**
- AdamW with various hyperparameters
- AdaFactor for memory-efficient training
- Apollo adaptive coordinate-wise optimization
- Custom research optimizer implementations

### Template Reusability

Build a library of reusable components:
- Base transformer architectures
- Training pipeline configurations
- Dataset preprocessing patterns
- Evaluation and monitoring setups

## 🔬 Research Applications

### Model Architecture Exploration

Forgather enables systematic comparison of:
- **Positional Encoding**: Compare sinusoidal, learned, ALiBi, and RoPE approaches
- **Layer Normalization**: Pre-norm vs post-norm architectures
- **Initialization Schemes**: Different weight initialization strategies
- **Activation Checkpointing**: Memory vs compute trade-offs

### Training Method Comparison

Compare different training approaches:
- **Single GPU**: Traditional training on individual cards
- **Multi-GPU**: Data parallel training with gradient synchronization
- **Pipeline Parallel**: Model parallel training across devices
- **Mixed Approaches**: Hybrid parallelism strategies

## 🎯 Framework Features in Action

### Dynamic Code Generation

Models become standalone Python modules:
- No runtime dependency on Forgather
- Portable across different environments
- Easy integration into existing pipelines
- Full model architecture preservation

### Configuration Validation

Template system catches errors early:
- Preprocessing validation before training
- Dependency checking across template inheritance
- Clear error messages for debugging
- Interactive exploration in Jupyter notebooks

### Reproducibility

Every training run captures:
- Complete configuration snapshots
- Generated model source code
- Training metrics and logs
- Environment and dependency information

## 🏗 Getting Involved

### For Researchers

Interested in systematic model comparison? Forgather's template system makes it easy to:
- Test hypotheses across model variants
- Maintain reproducible experiment records
- Share configurations with collaborators
- Build on others' template libraries

### For Engineers

Want to optimize large model training? Help us improve:
- Pipeline parallelism efficiency
- Memory optimization strategies
- Multi-node distributed training
- Performance profiling and monitoring

### For Hobbyists

Curious about training large models? Contribute by:
- Testing on different hardware configurations
- Sharing training results and observations
- Creating example configurations
- Documenting setup procedures

## 🔮 Future Directions

Areas where we're seeking collaboration:

**Multi-Node Training**: Extending pipeline parallelism across multiple machines

**Memory Optimization**: Advanced techniques for fitting larger models in consumer GPU memory

**Training Efficiency**: Optimizing pipeline schedules and reducing overhead

**Model Support**: Expanding beyond transformers to other architectures

**Tooling**: Better debugging, profiling, and monitoring capabilities

## Get Started

Ready to explore what's possible with consumer-grade distributed training?

[View the Repository →](https://github.com/jdinalt/forgather){: .btn .btn-primary}
[Read Getting Started →](/getting-started){: .btn .btn-outline}

---

*Examples represent current capabilities of alpha software. Performance and features continue to evolve.*
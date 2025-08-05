---
layout: home
title: "Forgather: Democratizing Large Model Training"
description: "Alpha-stage ML framework bringing distributed pipeline parallelism to consumer GPUs. Seeking collaborators to help develop the future of accessible AI training."
---

# Democratizing Large Model Training

**Forgather** is an alpha-stage ML framework that aims to make large model training accessible to hobbyists and researchers with consumer hardware.

## The Vision

### 🚀 **Pipeline Parallelism for Consumer GPUs**
Enable training of models larger than single GPU memory by distributing them across multiple consumer-grade cards. Our goal is to make 7B+ parameter model training accessible without enterprise hardware.

### 📝 **End Configuration Duplication**
Eliminate the copy-paste cycle of ML experiments through a powerful template inheritance system. Specify only what changes between experiments, not entire configurations.

### 🔧 **Framework-Independent Models**
Generate standalone Python code that works without dependencies on Forgather itself. Your trained models remain portable and deployable anywhere.

## Current Status: Alpha

Forgather is in active development with core functionality implemented:

- ✅ Template inheritance system working
- ✅ Pipeline parallelism implemented  
- ✅ Code generation pipeline functional
- ✅ Multi-GPU distributed training
- 🔄 Performance optimization ongoing
- 🔄 Documentation and examples expanding

## Seeking Collaborators

We're looking for contributors who share our vision of democratizing AI:

### 🧪 **Researchers & Experimenters**
Help test pipeline parallelism with different model architectures and share your findings.

### 💻 **ML Engineers**  
Contribute to performance optimization, memory efficiency, and distributed training improvements.

### 📚 **Documentation & Examples**
Help create tutorials, guides, and example configurations for the community.

### 🔧 **Core Development**
Contribute to framework architecture, code generation, and template systems.

## What We're Building

- **Accessible Training**: 7B+ models on consumer RTX setups
- **Template-Driven**: Systematic experimentation without configuration chaos
- **Pipeline Parallelism**: Multiple efficient scheduling algorithms
- **Framework Freedom**: Generated models work independently
- **Research Focus**: Built for exploration and comparison

## Early Results

Initial testing on 6x RTX 4090 setup shows promising results for 7B parameter model training with various pipeline schedules. We're particularly excited about zero-bubble pipeline performance, though optimization work continues.

[Explore the Code →](https://github.com/jdinalt/forgather){: .btn .btn-primary}
[Join Discussions →](https://github.com/jdinalt/forgather/discussions){: .btn .btn-outline}
[Report Issues →](https://github.com/jdinalt/forgather/issues){: .btn .btn-outline}

---

*Alpha software seeking alpha testers. Help us build the future of accessible AI training.*
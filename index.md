---
layout: home
title: "Forgather: Configuration-Driven ML Framework for Consumer-GPU Training"
description: "Forgather is a PyTorch ML framework with template-inheritance configs, framework-portable model export, pipeline parallelism for bandwidth-limited setups, and full-parameter 7B finetuning on a single 24 GB GPU."
---

# Forgather

[**Forgather**](https://github.com/jdinalt/forgather) is a configuration-driven ML framework built on template inheritance and code generation. It targets the gap between "research script that grows ten near-copies of itself" and "production framework you can't customise without forking."

The framework is open source, runs on PyTorch, and has been in active development since 2023.

## What's actually in the box

### Template-inheritance configs, no copy-paste

A Forgather project config extends a parent. Both are plain YAML with Jinja2 preprocessing, and *every knob* is an explicitly overridable block. To try a longer context window or a different optimizer, you write the override; the rest is inherited. Custom YAML tags (`!partial`, `!factory`, `!singleton`) let you swap in any Python class or function without touching Python source.

### Models that don't depend on Forgather

Each training run writes the equivalent PyTorch source into `output_models/`. The generated code has no Forgather dependency:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "output_models/my_run",
    trust_remote_code=True,
)
```

`forgather convert --reverse` will additionally emit a canonical Hugging Face checkpoint (Llama / Mistral / Qwen3 / Gemma-3) that loads without `trust_remote_code`. Train in Forgather, deploy anywhere.

### Pipeline parallelism for bandwidth-limited setups

The pipeline trainer (GPipe, 1F1B, Interleaved-1F1B, zero-bubble schedules) needs dramatically less cross-device communication than DDP or FSDP. Concrete results:

- **7B-parameter model trained across two machines connected by 1 Gbit Ethernet**, with bandwidth to spare.
- The same design avoids PCIe stalls that bottleneck FSDP on consumer GPUs without NVLink.
- DDP (with optional Post-Local-SGD), FSDP-2, and DiLoCo (Distributed Low-Communication Training, [arXiv:2311.08105](https://arxiv.org/abs/2311.08105)) are also first-class.

### Full-parameter 7B finetuning on a single 24 GB GPU

Not LoRA — full parameter, up to **~53 K tokens of context** on a single RTX 4090. Achieved by combining gradient checkpointing, CPU activation offload, fused optimizer step, fused linear + cross-entropy loss (Liger / Apple CCE / `torch.compile`), and packed sequences with Flex Attention.

A documented 9-way ablation of these techniques on a 1.6 B model showed an **81 % peak-memory reduction at ~2.7× throughput** versus the naive baseline. ([peak-memory experiment](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/peak_memory))

### Adafactor + AdamW with bf16 stochastic rounding

Forgather ships a fused Triton Adafactor that combines factored second-moment estimation with per-parameter stochastic rounding for bf16 weight updates, in a single kernel. To our knowledge this is the only Adafactor+SR implementation available, and it runs faster than every reference Adafactor we've benchmarked. SR is critical for *pure*-bf16 training (no fp32 master weights) — without it, sub-precision updates round systematically to zero and weight norms drift.

### Web UI with GPU-aware job queue

A single-user browser frontend over the same APIs as the CLI: project / file browsing, ▶ Run buttons that drop training jobs into a priority + GPU-policy queue, live job cards with TTY and training-stat pills, per-card process attribution, an in-browser editor with Forgather YAML+Jinja2 syntax highlighting, and a chat client wired to served inference jobs.

### Live job control across distributed workers

```bash
forgather control list
forgather control save JOB_ID        # save checkpoint on demand
forgather control save-stop JOB_ID   # save and gracefully stop
forgather control abort JOB_ID       # kill a failed experiment without saving
```

Distributed-safe — commands sent to any rank are coordinated across all DDP / FSDP-2 / pipeline workers.

### Reproducibility built in

Every run snapshots its full config *and* the generated model source. Resume restores optimizer state, LR scheduler, dataset position (stateful resume on C4-scale corpora is fast), RNG, and Tensorboard logs. Distributed checkpoints are written as standard HF Safetensors shards readable by `transformers`, vLLM, and llama.cpp conversion tools.

## Worked examples worth reading

These aren't toy demos — each has a README with reproducible commands and headline results:

- **[Long-context Lovecraft finetune + RoPE comparison](https://github.com/jdinalt/forgather/tree/main/examples/tutorials/hp_lovecraft_project)** — Mistral-7B / Llama-2-7B on the complete works of H. P. Lovecraft on a single 24 GB GPU, 53 K context. Companion document compares plain RoPE, YaRN, Llama-3 NTK-by-parts, and bumped `rope_theta`. Headline finding: **bumping `rope_theta` to 500 000 is the single biggest intervention for context extrapolation**.

- **[Optimizer comparison on a 30 M Llama](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/optimizers)** — empirical comparison of ten optimizers (Muon, Apollo, AdamW, Adafactor, SinkGD, SGD, …) on the SmolLM corpus. Headline: **Muon wins at small batch** (eval loss 2.6778 vs AdamW 2.7392), with `beta2` scaling becoming critical at large batch.

- **[Peak-memory ablation on a 1.6 B model](https://github.com/jdinalt/forgather/tree/main/examples/tiny_experiments/peak_memory)** — 9-way systematic ablation of memory-optimisation techniques. **81 % peak-memory reduction at ~2.7× throughput.** Pareto-frontier plots included.

- **[Pretraining at 162 M from scratch](https://github.com/jdinalt/forgather/tree/main/examples/pretrain/small-llm)** — Llama trained on the SmolLM corpus, ten production-ready configs covering 1× and 10× Chinchilla budgets, AdamW / Adafactor / bf16 variants, plus a "Canon-A" custom architecture variant.

- **[Multi-GPU 7B finetuning](https://github.com/jdinalt/forgather/tree/main/examples/finetune/samantha)** — Mistral-7B and Llama-3.2-1B on the Samantha conversational dataset across every trainer backend in the library. ~8.9 K tokens/sec on 4× RTX 4090 pipeline parallel.

## Getting started

```bash
git clone https://github.com/jdinalt/forgather.git
cd forgather
docker/build.sh
docker/run.sh

# Inside the container:
forgather server                     # web UI
# or
cd examples/tutorials/tiny_llama
forgather -t v2.yaml train           # 5M-param Llama in ~10 minutes
```

The [Tiny Llama tutorial](https://github.com/jdinalt/forgather/tree/main/examples/tutorials/tiny_llama) walks through the full train → monitor → control → eval → inference → export flow. Full documentation lives at [forgather.readthedocs.io](https://forgather.readthedocs.io/en/latest/).

## Project status

Active development since 2023. Used by the author for ongoing ML research; APIs are stable enough that the documentation lags behind the code rather than vice versa. Forgather is solo-maintained — feedback, issues, and pull requests are welcome.

[GitHub repository →](https://github.com/jdinalt/forgather){: .btn .btn-primary}
[Documentation →](https://forgather.readthedocs.io/en/latest/){: .btn .btn-outline}
[Discussions →](https://github.com/jdinalt/forgather/discussions){: .btn .btn-outline}

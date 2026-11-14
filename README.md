# AI Research Notebook

**Ellis C. Wu** · Personal AI research and paper-reading notes · **Work in progress — actively updated**

This repository is my evolving research notebook for studying **AI model architectures, mathematical foundations, and efficient training and inference systems**. It documents how I read research papers: working through equations, reconstructing derivations, connecting theory to implementation, and recording questions and ideas for further investigation.

The notes are primarily written in **Chinese**, with English technical terminology, mathematical notation, and occasional excerpts from the papers being studied. 

## How to use this notebook

### 1. Browse it on GitHub

Open any folder above and select a Markdown (`.md`) file. You can use the topic-based reading paths below as entry points. Note that **some Obsidian-specific image embeds (`![[...]]`) may not render in GitHub's Markdown viewer**.

### 2. Open it in Obsidian (recommended)

1. Download or clone the repository to your computer.
2. Install and launch [Obsidian](https://obsidian.md/).
3. Choose **Open folder as vault** and select the repository's `AI Research Notebook` folder (the folder containing `AI_Arch`, `llm_infra`, etc.).

## Suggested reading paths

### A. State space models → SSD → Mamba

For readers interested in the mathematical and architectural foundations, a useful path is:

1. [Mamba-2 reading notes, Part 1](AI_Arch/Mamba2/Mamba-2%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0Pt.1.md) — background and motivation.
2. [SSMs as structured / 1-SSS matrices](AI_Arch/Mamba2/%E9%AB%98%E4%B8%AD%E7%94%9F%E7%89%881-SSS%E7%9F%A9%E9%98%B5%E6%8E%A8%E5%AF%BC.md) — SSMs, semiseparable matrices, and the SSS representation.
3. [Masked and linear attention](AI_Arch/Mamba2/%E9%AB%98%E4%B8%AD%E7%94%9F%E7%89%88%E6%8E%A9%E7%A0%81%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%8E%A8%E5%AF%BC.md) — kernel, masked, and linear attention.
4. [State Space Duality (SSD)](AI_Arch/Mamba2/%E9%AB%98%E4%B8%AD%E7%94%9F%E7%89%88%E7%8A%B6%E6%80%81%E7%A9%BA%E9%97%B4%E5%AF%B9%E5%81%B6%E6%80%A7%E6%8E%A8%E5%AF%BC.md) — the link between structured SSMs and masked attention.
5. [Hardware-efficient SSD algorithms](AI_Arch/Mamba2/%E9%AB%98%E4%B8%AD%E7%94%9F%E7%89%88SSD%E6%A8%A1%E5%9E%8B%E7%A1%AC%E4%BB%B6%E9%AB%98%E6%95%88%E7%AE%97%E6%B3%95%E6%8E%A8%E5%AF%BC.md) — block decomposition and hardware-efficient SSD computation.
6. [Mamba-2 architecture](AI_Arch/Mamba2/%E9%AB%98%E4%B8%AD%E7%94%9F%E7%89%88Mamba-2%E6%9E%B6%E6%9E%84%E8%AE%B2%E8%A7%A3.md) — Mamba-2 block and architectural design.
7. [SSM system optimizations](AI_Arch/Mamba2/%E9%AB%98%E4%B8%AD%E7%94%9F%E7%89%88SSM%E7%B3%BB%E7%BB%9F%E4%BC%98%E5%8C%96%E8%AE%B2%E8%A7%A3.md) and [Mamba-2 empirical evaluations](AI_Arch/Mamba2/%E9%AB%98%E4%B8%AD%E7%94%9F%E7%89%88Mamba2%E5%AE%9E%E8%AF%81%E9%AA%8C%E8%AF%81%E8%AE%B2%E8%A7%A3.md) — parallelism, systems considerations, and experiments.

To continue with Mamba-3, start with [Mamba-3 preliminaries](AI_Arch/Mamba3/%E9%AB%98%E4%B8%AD%E7%94%9F%E7%89%88Mamba-3%20Preliminaries.md), then explore [Zero-order hold and exponential-Euler discretization](AI_Arch/Mamba3/%E9%9B%B6%E9%98%B6%E4%BF%9D%E6%8C%81%E5%92%8C%E6%8C%87%E6%95%B0-%E6%AC%A7%E6%8B%89%E7%A6%BB%E6%95%A3%E5%8C%96.md) and [Exponential-trapezoidal discretization](AI_Arch/Mamba3/%E6%8C%87%E6%95%B0%E6%A2%AF%E5%BD%A2%E7%A6%BB%E6%95%A3%E5%8C%96%E6%8E%A8%E5%AF%BC-%E8%80%83%E7%A0%94%E9%AB%98%E6%95%B0%E5%9B%9E%E6%9D%A5%E4%BA%86.md).

### B. Efficient LLM training and serving

- [From GEMM to collective communication](AI_infra/%E4%BB%8E%E9%80%9A%E7%94%A8%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95%E5%88%B0%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E7%AE%97%E5%AD%90.md) — linear algebra kernels, communication primitives, and their costs.
- [Megatron-LM](Pre_training/Megatron-LM%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md) — tensor/model parallelism for large-scale Transformer training.
- [vLLM / PagedAttention](llm_infra/vLLM%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md) — PagedAttention, memory fragmentation, and KV-cache management.
- [SGLang / RadixAttention](llm_infra/SGLang%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md) — LLM programming/runtime abstractions, RadixAttention, and scheduling.

### C. Architecture/Scaling Laws research questions

- [Scaling Laws Meet Model Architecture](Scalling_law/Scaling%20Laws%20Meet%20Model%20Architecture.md) and [GemStone takeaways](Scalling_law/GemStone%20take%20away.md) — observations about scaling, model architecture, and evaluation.
- [Mixture of a Million Experts (PEER)](MoE/Mixture%20of%20A%20Million%20Experts.md) — questions about expert specialization, scaling, and continual learning.


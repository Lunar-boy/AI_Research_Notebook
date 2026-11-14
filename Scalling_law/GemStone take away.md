核心发现：**他们注意到 ARC、HellaSwag 和 MMLU 这三个基准测试的准确率，在较小的计算规模（Smaller Compute Scales）下是最具有可预测性的。**
因此，在拟合准确率的缩放定律时，他们专门选用了这三个基准测试的子集。

这一**现象**也被 Magnusson 等人（2025年）在相同的基准测试中观察到了。

MMLU (Massive Multitask Language Understanding)核心考查：海量多任务学科知识（综合硬实力）。

ARC (AI2 Reasoning Challenge)科学常识与多步逻辑推理（拒绝死记硬背）。

HellaSwag基于常识的情境预测能力（理解日常生活逻辑）。

Establishing task scaling laws via compute-efficient model ladders.
arXiv preprint arXiv:2412.04403, 2024.

**随着深度的增加，精度也会提高**。我们观察到类似的模式
Gemma 2 报告的表 9 暗示了这一结果 Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024b.
其中作者指出，对于 90 亿规模的两个模型，更深的模型在下游基准测试中略优于更宽的模型。

最近的研究表明，网络中更深的层“作用较小”，可以被剪枝掉 
The unreasonable ineffectiveness of the deeper layers.
arXiv preprint arXiv:2403.17887, 2024.

但我们的后续评估表明，增加模型深度也有优势。我们在每个基准测试中都观察到了类似的性能模式，并将这些图表包含在附录图16中
当限制在大致相同的 FLOP 预算（垂直对齐的点）时，模型显示准确率随深度增加而提高。这种深度与准确率之间的关系也可以在许多单独的基准测试中观察到。
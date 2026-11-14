只测了稠密模型，没有MoE 2026年5月13日

推理成本已成为一个亟待解决的问题。尽管推理成本至关重要，但模型准确率和推理效率之间的权衡仍未得到充分研究。

本文探讨了关键架构因素（包括隐藏层大小、多层感知器（MLP）和注意力机制之间的参数分配比例（MLP/注意力机制比率）以及分组查询注意力（GQA））如何影响推理成本和准确率。

![[Scalling_law.png]]

我们发现分组查询注意力机制 (GQA) 对推理效率有着至关重要的影响，尽管它只是略微减少了注意力参数的数量（通过缩小键值矩阵）。为了区分这些影响，我们在以下设置下对隐藏层大小、MLP 与注意力机制的比例以及 GQA 进行了受控消融实验：

![[分组查询注意力机制参数设置.png]]

在 LLaMA-3.1-8B 模型变体上，我们观察到更大的隐藏层大小（或更少的注意力头）和更高的多层感知器与注意力机制的比率可以提高推理吞吐量。在 LLaMA-3.2-1B 和 3B 模型变体中也观察到了类似的趋势（附录 F）。这些提升部分源于更大的
𝑑模型及以上𝑟mlp/
注意 如推理浮点运算次数分析（附录 K）中所述，这些架构选择可以减少总浮点运算次数。

Gqa: training generalized multi-query transformer models from multi-head checkpoints.
arXiv preprint arXiv:2305.13245.

此外，这些架构选择还能缩小键值缓存，降低推理期间的 I/O 开销，从而进一步提高吞吐量。Keyformer: kv cache reduction through key tokens selection for efficient generative inference.
Proceedings of Machine Learning and Systems 6, pp. 114–127.

增加 GQA 值能够持续提高推理吞吐量。附录F中报告的一组针对 Qwen3 模型的类似消融实验 进一步证实了这些发现。

过大的隐藏层尺寸会减少注意力头的数量。𝑛头
可能会降低准确性——这种现象在之前对变压器容量和水头分配的分析中一直被观察到Scaling laws for neural language models.
arXiv preprint arXiv:2001.08361.

Training compute-optimal large language models.
arXiv preprint arXiv:2203.15556.

---

下面是标准差的计算公式：
$$ \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2} $$
$a$

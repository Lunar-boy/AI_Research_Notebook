# Abstract

扩展**推理时间计算能力**已成为提升LLM性能的重要驱动因素，使得推理效率与模型质量一样，成为模型设计的核心关注点。虽然目前基于Transformer的模型能够提供强大的模型质量，但其二次计算和线性内存需求使得推理成本高昂。这促使人们开发亚二次模型，以降低线性计算需求并保持恒定的内存要求。然而，许多近期的线性模型为了算法效率而牺牲了模型质量和能力，导致其在状态跟踪等任务上表现不佳。此外，它们理论上的线性推理在实践中仍然缺乏硬件效率。基于推理优先的理念，我们引入了三项核心方法改进，这些改进借鉴了线性模型的状态空间模型（SSM）视角。我们结合了：（1）源自SSM离散化的更具表达力的递归式；（2）能够实现更丰富状态跟踪的复值状态更新规则；以及（3）多输入多输出（MIMO）公式，以在不增加解码延迟的情况下提升模型性能。结合架构改进，我们的Mamba-3模型在检索、状态跟踪和下游语言建模任务中均取得了显著提升。在 15 亿数据规模下，Mamba-3 的平均下游准确率比次优模型（Gated DeltaNet）提高了 0.6 个百分点，而 Mamba-3 的 MIMO 变体在此基础上又提升了 1.2 个百分点，总共提升了 1.8 个百分点。在状态规模实验中，Mamba-3 仅使用了其前代模型一半的状态规模，却达到了与 Mamba-2 相当的困惑度。我们的评估结果表明，Mamba-3 能够进一步提升性能效率的帕累托前沿。
```
@inproceedings{
lahoti2026mamba,
title={Mamba-3: Improved Sequence Modeling using State Space Principles},
author={Aakash Lahoti and Kevin Li and Berlin Chen and Caitlin Wang and Aviv Bick and J Zico Kolter and Tri Dao and Albert Gu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=HwCvaJOiCj}
}
```
# introduction

测试时计算已成为 LLM 进步的关键驱动力，诸如思维链推理和迭代改进等技术表明，推理时扩展可以释放新的能力 , 并行、智能体工作流程的快速兴起，更加凸显了对高效推理和部署此类模型的需求,这种范式转变提高了推理效率. 至关重要的是，人工智能系统的实际影响如今在很大程度上取决于它们在部署期间执行大规模推理的能力。模型架构设计在决定推理效率方面起着根本性的作用，因为架构选择直接决定了生成过程中的计算和内存需求。虽然基于Transformer的模型是当前的行业标准，但它们的根本瓶颈在于 KV 缓存线性增长的内存需求和自注意力机制二次方增长的计算需求。这些缺点促使人们研究亚二次方模型，例如状态空间模型 (SSM) 和线性注意力机制。这些模型在保持恒定内存和线性计算需求的同时，还能达到与 Transformer 相当甚至更高的性能。这些模型已经成为主流，例如 Mamba-2  和门控 DeltaNet (GDN) 最近被纳入大规模混合模型中，这些模型在性能上与纯Transformer替代方案相当，但效率更高 
Qwen3 technical report.
Kimi linear: an expressive, efficient attention architecture. 
Hunyuan-turbos: advancing large language models through mamba-transformer synergy and adaptive chain-of-thought.
Nemotron-h: a family of accurate and efficient hybrid mamba-transformer models.

尽管线性模型取得了成功，但在提高其性能方面仍有很大的提升空间，尤其是在推进模型质量和推理效率之间的帕累托前沿方面。例如，Mamba-2 的开发旨在提高训练速度和简化操作，使其优于 Mamba-1  ，通过牺牲一些表达能力，因此在推理匹配模型中表现更差。此外，它们还被证明缺乏某些能力，例如较差的状态跟踪能力，例如，简单地确定比特序列的奇偶性 . 最后，尽管这些亚二次模型因其理论上的高效推理能力​​而备受推崇，并因此得到广泛应用，但它们的推理算法在硬件效率方面却不尽如人意。特别是，由于这些算法是从训练的角度开发的，它们的解码阶段算术强度（浮点运算次数与内存流量之比）较低，导致大量硬件资源处于闲置状态。

为了从推理优先范式开发性能更优的模型，我们在 Mamba-2 的基础上引入了三个核心方法论的改变，这些改变受到了以 SSM 为中心的亚二次模型观点的影响。

**指数梯形离散化。**

我们提供了一种用于离散化时变选择性SSM的简单技术。通过我们的框架，我们可以推导出几种新的离散化方法。其中一种称为“指数欧拉”的实例，形式化了之前缺乏理论依据的Mamba-1和Mamba-2的启发式离散化方法。我们新的“指数梯形”实例是“指数欧拉”的更具表现力的推广，其中递归式可以展开以揭示应用于SSM输入的隐式卷积。结合显式卷积，该方法能够更好地处理SSM输入。就偏差而言，Mamba-3 可以经验性地取代语言模型架构中的短因果卷积，而此前人们假设短因果卷积对于循环模型至关重要。

**复值状态空间模型。**
通过将 Mamba-3 的底层 SSM 视为复值，我们实现了比 Mamba-2 更具表现力的状态更新。这种更新规则的改变旨在减轻训练和推理的负担，克服了许多当前线性模型普遍存在的缺乏状态跟踪能力的问题。我们证明，我们的复值更新规则等价于数据相关的旋转嵌入，并且可以高效地计算，并通过实证证明其能够解决先前线性模型无法解决的合成任务。RoFormer: enhanced transformer with rotary position embedding.

**多输入多输出（MIMO）SSM。**

为了提高解码过程中的浮点运算效率，我们将基于外积的状态更新方式替换为基于矩阵乘法的状态更新方式。从SSM的信号处理基础来看，这种转变恰好对应于从单输入单输出（SISO）序列动态到多输入多输出（MIMO）序列动态的泛化。我们发现，MIMO尤其适用于推理，因为其额外的表达能力使得在解码过程中内存受限的状态更新能够进行更多计算，而无需增加状态大小或降低速度。

综上所述，这些改进构成了我们Mamba-3层的核心。方法论上，我们注意到这些改进都自然而然地源于以 SSM 为中心的视角，但从其他流行的现代线性层视角（例如线性注意力或测试时回归）来看，它们并非显而易见；

-  更佳质量。 在 15 亿规模下，Mamba-3 (MIMO) 的下游语言建模准确率比 Transformer 模型高出2.2 个百分点，比 Mamba-2模型高出1.9 个百分点，比 GDN 模型高出1.8 个百分点；而 Mamba-3 (SISO) 模型比次优模型 GDN 也高出0.6 个百分点。此外，在状态大小实验中，状态大小为 64 的 Mamba-3 (MIMO) 模型与状态大小为 128 的 Mamba-2 模型的困惑度相当，在延迟减半的情况下，有效实现了相同的语言建模性能。
- 新功能。Mamba -3 对 SSM 状态的复杂化使其能够解决 Mamba-2 无法解决的合成状态跟踪任务。我们通过实验证明，高效的类 RoPE 计算能够近乎完美地解决算术任务，而未采用 RoPE 的 Mamba-3 和 Mamba-2 的性能与随机猜测无异。
- 推理效率。Mamba -3（MIMO）提高了硬件利用率，解码浮点运算次数最多可增加4倍。在保持相似实际解码延迟的情况下，与 Mamba-2 相比，Mamba-3 在固定状态大小下具有更快的训练和推理性能，同时提升了困惑度和下游性能。我们发布了适用于 Mamba-3 的快速训练和推理内核

Mamba-3 (SISO) 在质量和性能方面超越了之前的线性模型，而 Mamba-3 (MIMO) 则在 Mamba-3 (SISO) 和其他优秀基线模型的基础上进一步提升了性能，同时推理速度与 Mamba-2 持平。我们提出的这两个 Mamba-3 变体凭借其强大的建模能力和高效的硬件设计，进一步拓展了性能-延迟的帕累托边界。

# 2.1 Preliminaries-Notation

Scalars are denoted by plain-text letters (e.g., $x, y$). 
Tensors, including vectors and matrices, are denoted by bold letters (e.g., $\mathbf{h}, \mathbf{C}$). 

The shape of the tensor can be inferred from the context.
We denote the input sequence length as $T$, the model dimension as $D$, and the SSM state size as $N$. 

For time indices, we use subscripts (e.g., $x_t$ for the input at time $t$).
The Hadamard product between two tensors is denoted by $\odot$.
For a vector $v \in \mathbb{R}^d$, we denote $\mathrm{Diag}(v) \in \mathbb{R}^{d \times d}$ as the diagonal matrix with the vector $v$ as the diagonal, and for products of scalars across time steps, we use the notation $\alpha_{t \cdots s} = \alpha^\times_{t:s} = \prod_{i=s}^{t} \alpha_i$.

# 2.2 SSM Preliminaries

State Space Models (SSMs) describe continuous-time linear dynamics via
$$\begin{align*}
\dot{\mathbf{h(t)}} &= \mathbf{A}(t)h(t) + \mathbf {B(t)}\,x(t), &
y(t) &=\mathbf C(t)^\top \mathbf h(t),
\end{align*}$$
where $\mathbf h(t)\!\in\!\mathbb{R}^N$ is the hidden state, $x(t)\!\in\!\mathbb{R}$ the input, and $\mathbf A(t)\!\in\!\mathbb{R}^{N\times N}$, $\mathbf B(t),\mathbf C(t)\!\in\!\mathbb{R}^N$.

We will occasionally偶尔 refer to $\mathbf A(t)$ as the state-transition and $\mathbf B(t)x(t)$ as the state-input; this also extends to their discretized 离散化 counterparts. 对应物
For discrete sequences with step size $\Delta_t$,
Mamba-1 and Mamba-2 discretized the system to the following recurrence
$$\begin{align*}
\mathbf h_t &= e^{\Delta_t A_t}\,\mathbf h_{t-1} + \Delta_t\,\mathbf B_t\,x_t, &
y_t &= \mathbf C_t^\top \mathbf h_t .
\end{align*}$$
**Mamba-2's Parameterization.** The core of the Mamba-2 layer is a data-dependent and hardware-efficient SSM.
Both the state-transition and state-input are made data-dependent through the projection of $\Delta_t \in \mathbb{R}_{>0}$ and $\mathbf{B, C}\in\mathbb{R}^N$ from the current token.

By parameterizing the state-transition $\mathbf A_t$ as a scalar times identity ($\mathbf{A_t} = A_t \mathbf{I}_{N\times N}$, where $A_t \in \mathbb{R}_{<0}$), the SSM recurrence can be efficiently computed with the matrix multiplication tensor cores of GPUs. Defining $\alpha_t \coloneqq e^{\Delta_t A_t}\in(0,1)$ and $\gamma_t \coloneqq \Delta_t$, the update becomes
$$\begin{equation}

\mathbf h_t = \alpha_t\,\mathbf h_{t-1} + \gamma_t\,\mathbf B_t\,x_t, \qquad
y_t = \mathbf C_t^\top \mathbf h_t .
\end{equation}$$ 
The data-dependent state-transition $\alpha_t$ controls the memory horizon of each SSM within the layer.

$\Delta_t$ in particular尤其 modulates调节 both the state-transition and state-input: a larger $\Delta_t$ forgets faster and up-weights the current token more strongly, while a smaller $\Delta_t$ retains 保留 the hidden state with minimal contributions from the current token.

In Mamba-2, $A_t$ is data-independent, since the overall discrete transition $\alpha_t \coloneqq e^{\Delta_t A_t}$ is data-dependent through $\Delta_t$.
In Mamba-3, we empirically found **that data-dependent $A_t$ has similar performance to data-independent $A_t$, and chose the former as a default for consistency so that all SSM parameters are data-dependent.**

# 2.3 Structured Masked Representation and State Space Duality

Mamba-2 showed that a large class of SSMs admit可以表示为 a matrix form that vectorizes the time-step recurrence. 
Mamba-2 实验表明，一大类状态空间模型（SSM）可以表示为一种矩阵形式，该形式能够向量化时间步长的递归过程。通过状态空间对偶性（SSD）框架，递归状态空间模型可以表示为一种并行形式，该形式包含一个逐元素掩码来模拟状态转移的衰减。SSD 为线性递归和可并行化（基于矩阵乘法）计算形式之间的对偶性提供了一个通用框架。$$\begin{align}
    \mathbf Y 
    =
    (   \mathbf L
        \odot 
        \mathbf C {\mathbf B}^{\top}
    )\mathbf X \quad \quad \quad \text{(2)}
\end{align}$$
where $L \in \mathbb{R}^{T \times T}$ is a structured mask,
$B,C \in \mathbb{R}^{T \times N}$, $X \in \mathbb{R}^{T \times D}$ are the inputs to the SSM and $Y \in \mathbb{R}^{T \times D}$ is its output. Different structures on $L$ give rise to various instantiations of SSD.

Eq.2 also draws a general connection between recurrence and attention, 
by setting $Q\coloneqq C$, $K\coloneqq B$, $V\coloneqq X$ and viewing $L$ as a data-dependent mask.
In fact, the simplest case of SSD is (causal) linear attention, where $L$ is the causal triangular mask.

Mamba-2 is a generalization where$$\begin{align}
    \mathbf L
    =
        \begin{bmatrix}
            1 \\
            \alpha_1 & 1 \\
            \vdots & & \ddots \\
            \alpha_{T...1} & \cdots & \alpha_T & 1
        \end{bmatrix}
        \cdot
        \operatorname*{Diag}(\gamma)
\end{align}$$
composed of terms $\alpha_t, \gamma_t$ 


In the original Mamba-2 paper, $\gamma$ does not appear because it is viewed as folded into the $\mathbf B$ term. In this paper, $\mathbf B_t$ represents the continuous parameter, whereas in Mamba-2, $\mathbf B_t$ represents the discretized parameter which is equivalent to $\gamma_t \mathbf B_t$.



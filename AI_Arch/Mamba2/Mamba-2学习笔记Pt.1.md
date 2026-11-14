# Abstract

尽管Transformer一直是深度学习在语言建模领域取得成功的主要架构，但最近的研究表明，诸如Mamba之类的状态空间模型（SSMs）在中小规模应用中能够与Transformer匹敌甚至超越。我们证明，这些模型家族实际上密切相关，并构建了一个丰富的理论框架，将SSMs与注意力机制的各种变体联系起来，这些联系是通过对一类经过充分研究的结构化半可分矩阵的各种分解来实现的semi separable matrices。我们的状态空间对偶性（SSD）state space duality框架使我们能够设计一种新的架构（Mamba-2），其核心层是对Mamba选择性SSM的改进，快2-8倍，whose core layer is an a refinement of Mamba’s selective SSM that is 2-8×faster同时在语言建模方面继续与 Transformer 保持竞争力。 
```
@article{dao2024transformers,
  title={Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality},
  author={Dao, Tri and Gu, Albert},
  journal={arXiv preprint arXiv:2405.21060},
  year={2024}
}
```

---
# 1.introduction

仅解码器transformer例如 GPT以因果方式处理输入序列，是现代深度学习成功的主要驱动力之一。许多方法试图近似核心注意力层以解决其效率问题“Efficient Transformers: A Survey”例如，在训练过程中，模型规模随序列长度呈二次方增长；在自回归生成过程中，缓存大小与序列长度呈线性关系。与此同时，一类替代序列模型——结构化状态空间模型（SSM）——应运而生，它们在训练过程中随序列长度呈线性增长，在生成过程中状态大小保持不变。它们在长程任务（例如 S4  ）上表现出色。“Efficiently Modeling Long Sequences with Structured State Spaces”
并且最近在语言建模方面与 Transformer 相媲美或更胜一筹（例如 Mamba)在小规模到中等规模的场景下， SSM的应用前景广阔。然而，SSM 的发展似乎与社区为改进 Transformer 所做的集体努力脱节，例如对 Transformer 的理论理解以及在现代硬件上的优化。因此，与 Transformer 相比，理解和实验 SSM 更加困难，而且从算法和系统角度来看，如何像训练 Transformer 一样高效地训练 SSM 仍然是一个挑战。

我们的主要目标是建立结构化SSM与注意力机制变体之间丰富的理论联系。这将使我们能够将最初为Transformer开发的算法和系统优化方法应用于SSM，从而构建性能优于Transformer且在序列长度上扩展效率更高的基础模型。线性注意力（LA）框架是这方面的一个里程碑式贡献“Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention”通过证明二次核注意力机制的“对偶形式”与特定线性递归等价，建立了自回归注意力机制与线性循环神经网络之间的联系。这种对偶性带来了新的能力，例如能够同时实现高效的并行训练和高效的自回归推理。本着同样的思路，本文从多个角度将线性复杂度的SSM与二次复杂度的形式联系起来，以结合SSM和注意力机制的优势。

**状态空间对偶性。State Space Duality.**
我们构建了一个连接结构化状态空间模型（SSM）和注意力机制变体的框架，我们称之为结构化状态空间对偶性（SSD）。该框架基于结构化矩阵的抽象：即具有亚二次参数和乘法复杂度的矩阵。我们开发了两种用于表示序列模型的框架，一种是矩阵变换，另一种是张量收缩，这两种框架分别揭示了对偶性的不同视角。我们的技术贡献包括：
- 我们证明了状态空间模型与一类研究充分的结构化矩阵（称为半可分矩阵）之间的等价性（第 3节）。 semiseparable matrices这种联系是我们框架的核心，它揭示了状态空间模型的新性质和算法。本文的一个核心观点是，计算状态空间模型的不同方法可以重新表述为结构化矩阵上的各种矩阵乘法算法。
- 我们显著改进了线性注意力理论(Katharopoulos等人，2020)首先，我们通过张量收缩的语言给出了其递归形式的有力证明，然后将其推广到一类新的结构化掩蔽注意力（SMA）（第 4节）。structured masked attention
-  我们将SSM和SMA联系起来，证明它们存在一个互为对偶的大型交集，该交集同时具有类似SSM的线性形式和类似注意力机制的二次形式（第 5节）。我们还证明，任何具有快速递归形式的核注意力机制都必然是SSM。

**Efficient Algorithms.**
首先，也是最重要的是，我们的框架提出了计算半可分矩阵 (SSM) 的高效且易于实现的新算法（第6节 ）。我们引入了一种新的SSD 算法，该算法基于半可分矩阵的块分解，利用了 SSM 的线性递推关系和二次对偶形式，在所有主要效率维度（例如训练和推理计算、内存使用以及在现代硬件上利用矩阵乘法单元的能力）上都获得了最优平衡。SSD 的专用实现比 Mamba 的优化选择性扫描实现速度快2−8倍，同时允许更大的循环状态大小（大小比 Mamba 大8倍甚至更大，速度下降也极小）。SSD 与优化的 softmax 注意力机制实现（FlashAttention-2  ）相比极具竞争力。“FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning”
crossing over at sequence length 2K and 6× faster at sequence length 16K.

**Architecture Design.**
采用诸如SSM之类的新架构的一大障碍在于其生态系统是为Transformer量身定制的，例如用于大规模训练的硬件高效优化和并行技术。我们的框架允许使用已建立的注意力机制约定和技术来构建SSM的架构设计选择词汇表，并进一步改进它们（第7节 ）。例如，我们将多头注意力机制（MHA）中的“头”的概念引入到SSM中。我们证明Mamba架构是一种多输入SSM（MIS） ，它与多值注意力机制（MVA）类似，并比较了具有不同“头”结构的其他Mamba变体。

我们还利用这些想法对 Mamba 模块进行了一些修改，从而实现了张量并行（例如，以 Megatron 的风格 ）。主要思想包括引入分组值注意力（GVA）头部结构，并将所有数据相关的投射移至区块开始时并行发生。

改进后的并行 Mamba 模块与使用 SSD 作为内部 SSM 层相结合，构成了Mamba-2架构。我们在与 Mamba 相同的设置下研究了 Mamba-2 的 Chinchilla 扩展规律，发现其在困惑度和实际运行时间方面均优于 Mamba 和 Transformer++。此外，我们在 Pile 数据集上训练了一系列不同规模的 Mamba-2 模型，结果表明其在标准下游评估中与 Mamba 和开源 Transformer 模型性能相当甚至更优。例如，在 Pile 数据集上训练的、参数量为 27 亿、使用 3000 亿个 token 的 Mamba-2 模型，其性能优于在同一数据集上训练的 Mamba-2.8B、Pythia-2.8B 甚至 Pythia-6.9B 模型。

**系统优化。Systems Optimizations.**
SSD 框架连接了 SSM 和 Transformer，使我们能够利用为 Transformer 开发的丰富的系统优化工作（第8节 ）。
- 例如，张量并行（TP）是一种重要的模型并行技术，它通过将每一层分配到同一节点上的多个GPU来训练大型Transformer模型。我们设计的Mamba-2对TP非常友好，将每个块的同步点数量减少了一半。
- 对于激活值无法在单个设备上完整显示的超长序列，我们开发了针对注意力模块的序列并行技术。本文将介绍如何通过在设备间传递循环状态，利用序列并行技术训练一般的SSM模型以及Mamba-2模型。
- 为了使用不同长度的示例进行微调以获得最佳效率，Transformer 需要复杂的技术来去除填充标记并对可变长度序列执行注意力机制。我们展示了如何高效地训练 Mamba-2 以处理可变长度的序列，而无需填充标记。

---
# 2. 背景与概述

## 2.1 结构化状态空间模型

Structured state space sequence models (S4)是近年来兴起的一类深度学习序列模型，与循环神经网络（RNN）、卷积神经网络（CNN）和经典状态空间模型密切相关。它们的灵感来源于一个特定的连续系统。该系统将一维序列映射到一个连续状态空间$x \in R^\mathtt{T} \mapsto y \in R^\mathtt{T}$ through an implicit latent state $h \in R^{\mathtt{(T, N)}}.$  
结构化 SSM 的一般离散形式如公式 ( 1 ) 所示。
$$
\begin{array}{cc}
\begin{cases}
h_{t} = A h_{t-1} + B x_t  & \text{(1a)} \\
y_t = C^{\top} h_t & \text{(1b)}
\end{cases}
& \qquad
\begin{cases}
h_{t} = A_t h_{t-1} + B_t x_t & \text{(2a)} \\
y_t = C_t^{\top} h_t & \text{(2b)}
\end{cases}
\end{array}
$$

where $A \in R^{\mathtt{(N,N)}}, B \in R^{\mathtt{(N,1)}}, C \in R^{\mathtt{(N,1)}}$. $h_t$ is $N \times 1$  Structured SSMs are so named because the 𝐴 matrix controlling the temporal dynamics must be structured in order to compute this sequence-to-sequence transformation efficiently enough to be used in deep neural networks. The original structures introduced were diagonal plus low-rank (DPLR) and diagonal which remains the most popular structure.
在本文中，我们使用术语状态空间模型（SSM）来指代结构化的SSM。这类SSM有很多种类型，与几种主要的神经序列模型范式（例如连续时间模型、循环模型和卷积模型）有着密切的联系 

h为隐状态（Latent State）或 记忆体
物理意义：它是模型的“大脑”或“记忆库”。负责把过去所有时间步看到的信息，压缩并存储在这一步的向量里。

 y为输出（Output）
物理意义：它是模型在当前步预测或输出的信号。在语言模型中，它可以用来预测下一个单词的概率。

C为输出矩阵/观测矩阵，负责从复杂的隐状态（记忆）中提取出我们想要的输出信息。

**连续时间模型。**
最初的结构化SSM起源于函数上的连续时间映射$x(t)\in R\mapsto y(t)\in R$, 而不是直接对序列进行操作。从连续时间的角度来看，在方程（1a）中，矩阵A,B这些参数并非直接学习而来，而是由底层参数$(\mathring{A}, \mathring{B})$生成。along with a parameterized step size $\Delta$.
The continuous parameters $(\Delta, \mathring{A}, \mathring{B})$ are converted to discrete parameters $(A, B)$ through fixed formulas $A = f_A(\Delta, \mathring{A})$ and $B = f_B(\Delta, \mathring{B})$, where the pair $(f_A, f_B)$ is called a **discretization rule**.

传统的 AI 模型（如 Transformer）直接处理的是“离散序列”（例如一个接一个的单词 Token、一帧一帧的图像）。
而 SSM 的起源非常独特，它借鉴了现代控制理论，把输入视为一条平滑、连续的信号 
因为计算机无法直接处理连续的无限微积分，只能处理离散的跳跃数字。因此，必须有一个“翻译官”$f$把连续的参数翻译成离散的矩阵。数学上通常使用零阶保持器 ZOH 或双线性变换
这样做最大的好处是让模型具备了“自适应的视野”：由于步长是可以随着输入数据动态调整的，当模型遇到非常重要、细致的信息时，它可以把 $\Delta$ 变小“放慢动作”仔细观察；当遇到不重要的背景噪点时，它可以把Delta变大，“快进”跳过。这种连续时间的设计，赋予了 SSM 极强的长文本记忆能力和捕获宏观规律的能力。

**循环模型。**
方程（1）和（2）的形式为输入项x的线性递推关系式，结构化SSM可以被视为一种循环神经网络（RNN），其线性特性赋予了它们额外的属性，并使其能够避免传统RNN的顺序计算。反之，尽管进行了这种简化，SSM仍然能够作为序列变换（在通用逼近的意义上）充分表达 。

**卷积模型。**
当SSM的动态特性如公式（1）所示随时间保持不变时，该模型被称为线性时不变（LTI）模型。在这种情况下，它们等价于卷积。因此，SSM也可以被视为一种CNN，但其中（i）卷积核是通过SSM参数$(A,B,C)$隐式参数化的。(ii) 卷积核通常是全局的而非局部的。反之，根据经典信号​​处理理论，所有足够良好的卷积都可以表示为SSM。

通常，先前的 LTI SSM 会使用卷积模式进行高效的并行训练（其中整个输入序列可以预先看到），并切换到循环模式（1）进行高效的自回归推理（其中输入可以一步一步地看到）。

**选择性状态空间模型。**
The form (2) where the parameters $(A, B, C)$ can also vary in time was introduced in Mamba as the $\textbf{selective SSM}$. 该模型可以随时间变化。与标准线性时不变（LTI）模型（1）相比，该模型可以在每个时间步选择性地关注或忽略输入。结果表明，在信息密集型数据（例如语言数据）上，该模型的性能远优于 LTI SSM，尤其是在状态大小较小时。这种方法可以增加信息容量。然而，它只能以循环模式而非卷积模式进行计算，并且需要精心设计以硬件为导向的实现才能高效运行。即便如此，它的效率仍然低于像 CNN 和 Transformer 这样对硬件友好的模型，因为它没有利用矩阵乘法单元，而现代加速器（例如 GPU 和 TPU）正是专门处理矩阵乘法的。

虽然时不变状态空间模型（SSM）与连续序列模型、循环序列模型和卷积序列模型密切相关，但它们与注意力机制并无直接关联。本文揭示了选择性SSM与注意力机制之间更深层次的联系，并利用这种联系显著提升了SSM的训练速度，同时允许更大的状态空间。

**Structured SSMs as Sequence Transformations.**

**定义 2.1 (Sequence Transformation).** 我们使用术语 **sequence transformation** 来指代一个作用在序列上的参数化映射 $Y = f_{\theta}(X)$，其中 $X, Y \in \mathbb{R}^{\mathtt{(T,P)}}$，且 $\theta$ 是任意参数集合。$\mathtt{T}$ 代表序列或时间轴；下标用于对第一维进行索引，例如 $X_t, Y_t \in \mathbb{R}^\mathtt{P}$。

Sequence transformations (e.g. SSMs, or self-attention) 
是深度序列模型的基石，它们被集成到神经网络架构（例如Transformer）中式( 1 )或( 2 )中的序列形状记忆模型是一种$\mathtt{P}=1$的序列变换。it can be generalized to $\mathtt{P} > 1$ by simply broadcasting across this dimension (in other words, viewing the input as $\mathtt{P}$ independent sequences and applying the SSM to each). One can think of $\mathtt{P}$ as a $\textbf{head dimension}$ .

**Definition 2.2** We define the $\textbf{SSM operator}$ $\mathsf{SSM}(A, B, C) = \mathsf{SSM}(A_{0:T}, B_{0:T}, C_{0:T})$ as the sequence transformation $X \in R^{\mathtt{(T,P)}} \mapsto Y \in R^{\mathtt{(T,P)}}$defined by equation (2).

In SSMs, the $\mathtt{N}$ dimension is a free parameter called the $\textbf{state size}$ or state dimension.
We also call it the $\textbf{state expansion factor}$, because it expands the size of the input/output by a factor of $N$, with implications for the computational efficiency of these models.
最后，我们指出，许多类型的序列变换（例如注意力）可以表示为沿序列维度的单个矩阵乘法。

**Definition 2.3** We call a sequence transformation $Y = f_\theta(X)$ a $\textbf{matrix transformation}$ if it can be written in the form $Y = M_\theta X$ where $M$ is a matrix depending on the parameters $\theta$.
We identify the sequence transformation with the matrix $M$, and often drop the dependence on $\theta$ when clear from context.

## 2.2 Attention

注意力机制广义上是指一种计算方法，它为序列中的每一对位置分配分数，使每个元素能够“关注”其他元素。迄今为止，注意力机制最常见且最重要的变体是softmax自注意力机制，它可以定义为：
$$  Y = \operatorname*{softmax}(QK^{\top}) \cdot V
$$
for $Q, K, V \in R^{\mathtt{(T,P)}}$.The mechanism of pairwise comparisons (induced by materializing $QK^{\top}$) leads to the characteristic quadratic training cost of attention.
人们提出了许多注意力机制的变体，但所有变体都共享这些注意力分数的基本核心，只是近似程度有所不同. 这项工作最重要的变体是**线性注意力机制** 粗略地说，这类方法通过将softmax折叠到核特征图中来丢弃softmax，并利用矩阵乘法的结合律来重写$(QK^{\top}) \cdot V = Q \cdot (K^{\top} V)$.
Moreover, in the important case of causal (autoregressive) attention, they show that when the causal mask is incorporated into the left-hand side as
$(L \circ QK^{\top}) \cdot V$, where $L$ is the lower-triangular 1's matrix, then the right-hand side can be expanded as a recurrence递推形式.

下三角全 1 矩阵。在深度学习中，它通常被用作因果掩码（Causal Mask）。当他与注意力分数矩阵QK^T 逐元素相乘时，会强行把未来的信息抹除为0

## 2.3结构化矩阵
General matrices $M \in R^{\mathtt{(T,T)}}$ require $\mathtt{T}^2$ parameters to represent and $O(\mathtt{T}^2)$ time to perform basic operations such as matrix-vector multiplication.$\textbf{Structured matrices}$ are those that
-  can be represented in subquadratic (ideally linear) parameters through a compressed representation, and
- have fast algorithms (most importantly matrix multiplication) by operating directly on this compressed representation.
结构化矩阵中最典型的族群或许是稀疏矩阵和低秩矩阵。然而，还有许多其他族群，例如托普利茨矩阵、柯西矩阵、范德蒙矩阵和蝴蝶矩阵，它们都已被用于机器学习中构建高效模型.结构化矩阵是一种强大的抽象方法，可用于高效的表示和算法。本文将证明 SSM 等价于另一类此前未在深度学习中使用的结构化矩阵，并利用这种联系推导出高效的方法和算法。

## 2.4 概述：结构化状态空间对偶性
虽然本文构建了一个更丰富的 SSM、注意力机制和结构化矩阵之间的联系框架，但我们在此简要概述主要方法，该方法实际上相当独立且算法简单。

**递归（线性）形式。Recurrent (Linear) Form.**
状态空间对偶（SSD）层可以定义为选择性状态空间模型（SSM）的一个特例（2）。SSM 的标准计算方法可以采用递归（或并行扫描）的方式，其复杂度与序列长度呈线性关系。与 Mamba 中使用的版本相比，SSD 有两个细微差别：
- The structure on $A$ is further simplified from diagonal to $\textbf{scalar times identity}$ structure.Each $A_t$ can also be identified with just a scalar in this case.
- We use a larger head dimension $\mathtt{P}$, compared to $\mathtt{P}=1$ used in Mamba. Typically $\mathtt{P}=\{64,128\}$ is chosen which is similar to conventions for modern Transformers.
与原始的选择性SSM相比，这些改变可以看作是在略微降低表达能力的同时，显著提高了训练效率。尤其值得一提的是，我们的新算法将允许在现代加速器上使用矩阵乘法单元。

**对偶（二次）形式。Dual (Quadratic) Form.**
SSD 的对偶形式是一种与注意力密切相关的二次计算，定义为：
$$\begin{align*}
  (L \circ QK^{\top}) \cdot V \qquad
  L_{ij} = \begin{cases}
    a_i \times \dots \times a_{j+1} & i \ge j \\
    0 & i < j
  \end{cases}
\end{align*}
$$
where $a_i$ are input-dependent scalars bounded in $[0, 1]$.

与标准softmax attention相比，主要有两点不同
- softmax is dropped 
- The attention matrix is multiplied elementwise-wise by an additional mask matrix  $L$ 
这两个变化都可以看作是为了解决普通注意力机制中的问题。例如，最近有研究发现 softmax 函数会导致注意力得分出现问题，例如“注意力陷阱”现象“Efficient Streaming Language Models with Attention Sinks”更重要的是，掩模矩阵$L$ 可以看作是用不同的数据相关的位置掩码替换 Transformer 的启发式位置嵌入，该掩码控制有多少信息随时间传递。

**矩阵形式和SSD算法。**

The various forms of SSD are connected through a unified matrix representation,
by showing that SSMs have a matrix transformation form
$Y = MX$ for a matrix $M_\theta \in R^{\mathtt{(T,T)}}$ that depends on $\theta = (A, B, C)$.
In particular, the dual form of SSD is equivalent to naive (quadratic-time) multiplication by the matrix $M$,
and the recurrent form递归形式 is a particular efficient (linear-time) algorithm that leverages the structure in $M$.

Going beyond these, any algorithm for multiplication by $M$ can be applied.
Our proposed hardware-efficient SSD algorithm  is a new structured matrix multiplication method that involves block decompositions of $M$, which obtains better efficiency tradeoffs than either the pure linear or quadratic forms.
It is relatively simple and easy-to-implement compared to general selective SSMs.
### Listing 1 Full PyTorch example of the state space dual (SSD) model.

```python
def segsum(x):
    """Naive segment sum calculation. exp(segsum(A)) produces a 1-SS matrix,
    which is equivalent to a scalar SSM."""
    T = x.size(-1) #获取输入序列的时间步数（序列长度）
    x_cumsum = torch.cumsum(x, dim=-1)
    #计算 x 的沿着最后一个维度的前缀和（Prefix Sum/Cumulative Sum）
    #如果 x = [a, b, c]，那么 x_cumsum = [a, a+b, a+b+c]
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    #x_cumsum[..., :, None]：在最后插入一个新维度，形状变为（T，1）
    #x_cumsum[..., None, :]：倒数第二个位置插入一个新维度，形状变为（1，T），两者相减，利用广播机制扩展为(T,T)
    # ...代表前面所有维度保持不变。
    #:代表选取当前维度的所有元素（不做切片截取）。
    # None（np.newaxis）在对应位置增加一个新的维度，长度为 1
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    # torch.ones(T,T,device=x.device,dtype=bool)生成一个T \times T全为True的布尔矩阵
    # torch.tril(..., diagonal=0)triangular lower下三角，保留主对角线diagonal=0,以及左下元素为true右上严格上三角为false 
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    #~mask 取反 masked_fill(~mask, -torch.inf)把矩阵中~mask为True的部分取-\infty
    return x_segsum

def ssd(X, A, B, C, block_len=64, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0
	#在SSD输入张量中，形状为(batch,length,n_heads,d_head)，第一个维度为序列长度，Mamba-2为了在GPU上高效并行，把整条序列分为长度为block_len的小块，
    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
	#rearrange来自einops库，利用模式字符串对张量维度进行重排与拆分
	# "b (c l) ..."表示输入的第一个维度为batch，第二个维度是整条序列的长度T=c \times l, ...表示后面的所有其他维度。
	# b c l ...将序列切分成两个新维度，块数量c与块内长度l
    A = rearrange(A, "b c l h -> b h c l")
    # 将多头维度h调换到c之前，方便后续把每个头和每个块作为独立的并行批处理维度，l为块内的时间步，长度为64
    A_cumsum = torch.cumsum(A, dim=-1)
	#沿着最后维度l计算块内的前缀和
    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    #输入A的形状(b h c l)输出为(b h c l l)
    #torch.exp取指数后，上三角的-infty变为0，下三角变为累乘衰减银子exp(\sum(A))
    #L代表在一个 block/chunk 内部，位置s 到位置 l 之间的连续状态衰减权重矩阵
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    #einsum规则，在左边出现，右边消失的字母，会被自动沿该维度求和消去，该einsum做了n和s的求和
    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    #A_cumsum[:,:,:,-1:]取最后一个时间步(b,h,c,1)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
        #states[:,:1]切片，只取第0个chunk，b1hpn
        #torch.zeros_like()生成一个形状，设备，数据类型与内部完全一致的全0张量
    states = torch.cat([initial_states, states], dim=1)
    #dim=1沿着chunk维度拼接
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    #取出每个chunk内部最后一个时间步的前缀和A_cumsum[:,:,:,-1]
    #F.pad(...,(1,0)),在最后一个维度的左侧填充1个0
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    #SSD块间计算，直接构造C+1 \times C+1的衰减矩阵，通过一次einsum，一次并行算完所有Chunk在考虑了前面所有历史Chunk传递衰减后的最终真实状态。
    states, final_state = new_states[:, :-1], new_states[:, -1]
    # [:-1]表示从0到倒数第二个索引，不包含最后一个-1
    # final_state=new_states[:,-1]表示直接取最后一个

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
```    
Parallelization: larger $\mathtt{M}, \mathtt{N}, \mathtt{K}$ terms can leverage specialized matrix multiplication units on modern accelerators.

## 2.5 Notation

**矩阵和向量。**
我们通常用小写字母表示向量（即单轴张量），用大写字母表示矩阵（即多轴张量）​​。本文中矩阵不加粗。有时，如果矩阵沿某一轴重复或相连（因此也可以视为向量），则可以使用大写或小写字母In this work, this happens only with the $A$ parameter of SSMs. $\cdot$ denotes scalar or matrix multiplication while $\circ$ denotes Hadamard (elementwise) multiplication.

**Indexing.**
We use Python-style indexing, e.g. $i:j$ refers to the range $(i, i+1, \dots, j-1)$ when $i < j$ and $(i, i-1, \dots, j+1)$ when $i > j$.
For example, for any symbol $v$ we let $v_{j:i}$ for $j \ge i$ denote the sequence $(v_j, \dots, v_{i+1})$.
$[i]$ is equivalent to $0:i = (0, \dots, i-1)$.
For shorthand, we also let $v_{j:i}^\times$ denote the product $v_j \times \dots \times v_{i+1}$.
In some contexts, it is always clear that the notation $a_{i:j}$ or $A_{i:j}$ means $a_{i:j}^\times$, and the superscript is omitted.

**Dimensions.**
To distinguish from matrices and tensors, we often use capital letters in typewriter fonts (e.g. $\mathtt{D}, \mathtt{N}, \mathtt{T})$ to denote dimensions and tensor shapes.
Instead of the traditional notation $M \in \mathbb{R}^{T \times T}$ we frequently use $M \in \mathbb{R}^{\mathtt{(T,T)}}$ to reflect tensor shapes in code.

**Tensor Contractions.张量缩并**
We will heavily rely on $\textbf{tensor contraction}$ or $\textbf{einsum}$ notation 爱因斯坦求和标记both for clarity and as a central tool in stating and proving our results.
We assume the reader to be familiar with this notation,
which is commonly used in modern tensor libraries such as $\texttt{numpy}$.
For example, we can use $\mathsf{contract}(\mathtt{MN,NK \to MK})$ to denote the matrix-matrix multiplication operator, and in our notation $\mathsf{contract}(\mathtt{MN,NK \to MK})(X, Y)$ (which is equivalent to $X \cdot Y$) can be translated to code as $\mathtt{numpy.einsum('mn,nk\to mk', X, Y)}$.


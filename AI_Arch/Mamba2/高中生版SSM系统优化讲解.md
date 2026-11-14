本文针对SSM（空间结构模型）及其Mamba-2架构，提出了几种系统优化方案，以实现大规模高效训练和推理。我们重点关注用于大规模训练的张量并行和序列并行，以及用于高效微调和推理的变长序列。

# 8.1 Tensor Parallel
张量并行（TP）是一种模型并行化技术，它将每一层（例如，注意力机制、多层感知器）拆分到多个加速器（例如 GPU）上运行。这种技术广泛用于训练大多数大型模型 。在GPU集群上，每个节点通常配备4-8个GPU，并采用NVLink等高速网络。TP最初是为Transformer架构开发的，将其适配到其他架构并非易事。我们首先展示了在Mamba架构上使用TP所面临的挑战，然后阐述了Mamba-2架构如何设计以提高TP的效率。

Recall the Mamba architecture, with a single input $u \in \mathbb{R}^{L \times d}$ (no batching for simplicity), input projection matrices $W^{(x)}, W^{(z)} \in \mathbb{R}^{d \times ed}$ where $e$ is the expansion factor (typically 2), and output projection matrix $W^{(o)} \in \mathbb{R}^{ed \times d}$:$$\begin{align*}
   x &= u {W^{(x)}}^\top \in \mathbb{R}^{L \times ed} \\
   z &= u {W^{(z)}}^\top \in \mathbb{R}^{L \times ed} \\
   x_c &= \mathrm{conv1d}(x) \in \mathbb{R}^{L \times ed} \quad \text{(depthwise, independent along $d$)} \\
   \Delta, B, C &= \text{low-rank projection}(x_c) \\
   y &= SSM_{A, B, C, \Delta}(x_c) \in \mathbb{R}^{L \times ed} \quad \text{(independent along $d$)} \\
   y_g &= y \cdot \phi(z)  \quad \text{(gating, e.g., with $\phi$ being SiLU)} \\
  \mathrm{out} &= y_g {W^{(o)}}^\top \in \mathbb{R}^{L \times d}.
\end{align*}$$
---

 **Mamba-1 / Selective State Space Model (SSM) 架构的前向传播计算流程（Forward Pass）**。
### 1. 变量与维度定义

- **$u \in \mathbb{R}^{L \times d}$**：模型的输入张量。其中 $L$ 为序列长度（Sequence Length），$d$ 为隐藏层模型维度（Model Dimension / $d_{\text{model}}$）。为了简化表达，忽略了 Batch 维度。
- **$W^{(x)}, W^{(z)} \in \mathbb{R}^{d \times ed}$**：输入投影权重矩阵。其中 $e$ 为扩展因子（Expansion Factor，通常设为 2），即把通道维度从 $d$ 映射扩大到 $ed$（例如从 768 拓展到 1536）。
- **$W^{(o)} \in \mathbb{R}^{ed \times d}$**：最终的输出投影矩阵，将特征维度降回 $d$。
### 2. 逐步计算流程分解

1. **输入特征双路映射（Linear Projections）**$$\begin{aligned} x &= u W^{(x)\top} \in \mathbb{R}^{L \times ed} \\ z &= u W^{(z)\top} \in \mathbb{R}^{L \times ed} \end{aligned}$$
    - 将输入 $u$ 通过两个独立线性层映射到更高维的空间（维度为 $ed$）。
    - **$x$** 进入主分支（SSM 处理路径）；
    - **$z$** 进入残差/门控分支（Gating 路径）。

2. **深度一维卷积（Depthwise 1D Convolution）**$$x_c = \text{conv1d}(x) \in \mathbb{R}^{L \times ed}$$
    - 对 $x$ 在时间/序列长度 $L$ 维度上应用局部 1D 卷积（通常 kernel size 为 4），增强局部上下文信息。

    - 注意注脚：**(depthwise, independent along $d$)** 指出此卷积是逐通道（Depthwise）进行的，各个通道之间互不干涉。

3. **动态选择参数生成（Selective Mechanism）**$$\Delta, B, C = \text{low-rank projection}(x_c)$$
    - 这是 Mamba 的核心创举（Selective SSM）：SSM 的控制参数 $\Delta$（步长/离散化参数）、$B$（输入权重矩阵）和 $C$（输出权重矩阵）不再是固定不动的常数，而是**由当前输入 $x_c$ 通过低秩线性投影动态生成**的。这使得模型能够根据输入内容实现选择性记忆与过滤。
4. **核心 SSM 模块计算（State Space Model Core）**$$y = \text{SSM}_{A, B, C, \Delta}(x_c) \in \mathbb{R}^{L \times ed}$$
    - 将经过卷积后的输入 $x_c$ 以及动态生成的 $\Delta, B, C$（结合预先定义的矩阵 $A$）送入连续状态空间方程的离散化版本计算。
    - **(independent along $d$)**：强调 SSM 在特征维度 $ed$ 的各个通道上是独立平行计算的（各通道间通过对角/块对角结构进行解耦）。
5. **门控激活相乘（Gated Activation）$$y_g = y \cdot \phi(z)$$
    - 类似 SwiGLU / Gated Linear Unit (GLU) 结构。将 SSM 的输出 $y$ 与门控分支经过激活函数 $\phi$（如 SiLU / Swish）处理后的结果进行逐元素按位相乘（Element-wise Multiplication）。
6. **输出映射（Output Projection）**$$\text{out} = y_g W^{(o)\top} \in \mathbb{R}^{L \times d}$$
    - 最后经过线性层 $W^{(o)}$，将特征维度从扩展后的 $ed$ 投影恢复回原始的模型维度 $d$，完成当前 Mamba 块的计算。
    
Mamba 模块的典型数据流：**双路分支结构（SSM 主路径 + SiLU 门控路径）** $\rightarrow$ **Depthwise Conv 局部特征提取** $\rightarrow$ **基于输入的选择性 SSM 序列建模** $\rightarrow$ **门控融合与输出投影**。

---
With TP, suppose that we want to split the computation along 2 GPUs. It is easy to split the input projection matrices $W^{(x)}$ and $W^{(z)}$ into two partitions each of size $d \times \frac{ed}{2}$.
Then each GPU would hold half of $x_c$ of size $L \times \frac{ed}{2}$.
However, we see that since $\Delta, B, C$ are functions of $x_c$, so we would need an extra all-reduce between the GPUs to get the whole of $x_c$ before computing $\Delta, B, C$.
After that the two GPUs can compute the SSM in parallel since they are independent along $d$.

At the end, we can split the output projection matrices $W^{(o)}$ into two partitions each of size $\frac{ed}{2} \times d$, and do an all-reduce at the end.
Compared to Transformers, we would incur招致 two all-reduces instead of one, doubling the time spent in communication. For large-scale Transformers training, communication might already take a significant fraction of time (e.g. 10-20\%), **and doubling communication would make Mamba not as efficient for large-scale training.**

**With Mamba-2,** our goal is to have only one all-reduce per block, similar to attention or MLP blocks in Transformers.

As a result, we have the projection to get $\Delta, B, C$ directly from $u$ instead of from $x_c$, allowing us to split these projection matrices.
This implies that we have different sets of $\Delta, B, C$ on different GPUs, which is equivalent to having several groups of $\Delta, B, C$ on a larger logical GPU.

Moreover, we use GroupNorm within each block, with number of groups divisible by the TP degree, so that the GPUs in a TP group do not have a communicate within the block:$$\begin{align*}
   x &= u {W^{(x)}}^\top \in \mathbb{R}^{L \times ed} \\
   z &= u {W^{(z)}}^\top \in \mathbb{R}^{L \times ed} \\
   \Delta, B, C &= \text{projection}(u) \quad \text{(one or more groups of $\Delta, B, C$ per GPU)} \\
   x_c &= \mathrm{conv1d}(x) \in \mathbb{R}^{L \times ed} \quad \text{(depthwise, independent along $d$)} \\
   y &= SSM_{A, B, C, \Delta}(x_c) \in \mathbb{R}^{L \times ed} \quad \text{(independent along $d$)} \\
   y_g &= y \cdot \phi(z)  \quad \text{(gating, e.g., with $\phi$ being SiLU)} \\
   y_n &= \mathrm{groupnorm}(y_g) \quad \text{(number of groups divisible by degree of tensor parallel)} \\
   \mathrm{out} &= y_g {W^{(o)}}^\top \in \mathbb{R}^{L \times d}.
\end{align*}$$
We see that we only need to split the input projection matrices, and the output projection matrices, and only need to do all-reduce at the end of the block. This is similar to the design of TP for attention and MLP layers.

In particular, if we have TP degree 2, we would split $W^{(x)} = [W^{(x)}_1, W^{(x)}_2]$ with $W^{(x)}_i \in \mathbb{R}^{d \times ed/2}$,
$W^{(z)} = [W^{(z)}_1, W^{(z)}_2]$ with $W^{(z)}_i \in \mathbb{R}^{d \times ed/2}$,
and $W^{(o)} = \begin{bmatrix} W^{(o)}_1 \\ W^{(o)}_2 \end{bmatrix}$ with $W^{(o)}_i \in \mathbb{R}^{ed/2 \times d}$.
For $i = 1, 2$, the TP Mamba-2 layer can be written as:$$\begin{align*}
   x^{(i)} &= u {W^{(x)}_i}^\top \in \mathbb{R}^{L \times ed / 2} \\
   z^{(i)} &= u {W^{(z)}_i}^\top \in \mathbb{R}^{L \times ed / 2} \\
   \Delta^{(i)}, B^{(i)}, C^{(i)} &= \text{projection}(u) \quad \text{(one or more groups of $\Delta, B, C$ per GPU)} \\
   x_c^{(i)} &= \mathrm{conv1d}(x^{(i)}) \in \mathbb{R}^{L \times ed / 2} \\
   y^{(i)} &= SSM_{A, B, C, \Delta}(x_c^{(i)}) \in \mathbb{R}^{L \times ed/2}  \\
   y_g^{(i)} &= y^{(i)} \cdot \phi(z^{(i)})  \\
   y_n^{(i)} &= \mathrm{groupnorm}(y_g^{(i)}) \quad \text{(number of groups divisible by degree of tensor parallel)} \\
   \mathrm{out}^{(i)} &= y_g^{(i)} {W^{(o)}_i}^\top \in \mathbb{R}^{L \times d / 2} \\
   \mathrm{out} &= \sum_i \mathrm{out}^{(i)}. \quad \text{(summing outputs from all GPUs with an all-reduce)}
\end{align*}$$
![[Mamba2_figure7.png]]

---

# 8.2 Sequence Parallelism

For very long sequences, we might need to split the input and activation to different GPUs along the sequence length dimension.
There are two main techniques:
- Sequence parallelism (SP) for the residual and normalization operations: **this technique decomposes the all-reduce in TP as reduce-scatter and all-gather**. 
- Noticing that the residual and normalization operations are repeated on the same input for all GPUs in the same TP group, SP splits the activations along the sequence length dimension by performing: reduce-scatter, residual and normalization, then all-gather.**Since the Mamba-2 architecture uses the same residual and normalization structure, SP applies without modification.**
- Sequence parallelism for the token-mixing operations (attention or SSM), also known as **context parallelism (CP)**. **Several techniques have been developed for attention layer** (e.g., **Ring attention**), with sophisticated **load-balancing technique**.
- The difficulty with sequence parallelism in attention is that we can split queries and keys into block, **but each query block needs to interact with key blocks, leading to communication bandwidth quadratic in the number of workers.** 
- **With SSMs, we can split the sequence in a simple manner: each worker takes an initial state, compute the SSM with respect to their inputs, return the final state, and pass that final state to the next worker.** The communication bandwidth is **linear in the number of workers**. This decomposition is exactly the same as the block-decomposition in the SSD algorithm  to split into blocks / chunks.
---
# 8.3 Variable Length
预训练通常使用相同长度的序列作为批次，但在微调或推理过程中，模型可能需要处理不同长度的输入序列。一种简单的处理方法是将批次中的所有序列右填充到最大长度，但如果序列长度差异很大，这种方法效率会很低。对于Transformer模型，已经开发出一些复杂的技术来避免填充，并在GPU之间进行负载均衡, 或者将多个序列打包到同一批次中并调整注意力掩码 。特别是对于 SSM 和 Mamba，我们可以通过将整个批次视为一个长序列来处理可变序列长度，从而避免在各个序列之间传递状态. This is equivalent to simply setting $A_t = 0$ for tokens $t$ at the end of one sequence to prevent it from passing information to the token $t + 1$, which belongs to a different sequence.



在 SSM / Mamba 架构中，隐状态（Hidden State）是按照状态转移方程连续累加传递的：
$$h_t = A_t h_{t-1} + B_t x_t$$

当为了避免 Padding（填充 padding token），把多个不同长度的序列首尾拼接（Concatenate/Packing）成**一个长的 Batch 序列**时，会遇到一个问题：**前一个序列末尾的隐状态 $h_t$，不能泄露（污染）给下一个序列开头的 token $t+1$。**

  

这句话解释了 SSM 在处理打包序列（Sequence Packing / Variable Length Batching）时的极简机制：
1. **状态重置（State Reset）：** 在前一个序列的最后一个位置 $t$，强行将隐状态转移矩阵（或离散化后的状态衰减参数 $\bar{A}_t$）置为 $0$（即 $A_t = 0$）。
2. **切断历史信息：** 这样在计算下一个位置 $t+1$ 的隐状态时：$$h_{t+1} = A_{t+1} h_t + B_{t+1} x_{t+1}$$
    因为前一步传递过来的项包含了 $A_t = 0$，上一个序列积累的所有历史记忆 $h_t$ 会被**瞬间清零（Reset）**。
3. **效果：** 属于新序列的 $t+1$ 位置就只会接收它自己的输入 $x_{t+1}$，从头重新积累状态，实现了无需显式 Padding 或复杂 Mask 运算的硬隔离。
# A Hardware-Efficient Algorithm for SSD Models

在结构化矩阵（SSM）、注意力机制和结构化矩阵之间构建理论SSD框架的优势在于，可以利用这些联系来改进模型和算法。本节将展示如何从各种结构化矩阵乘法算法中高效地推导出各种SSD模型计算算法。

我们的主要计算成果是一种用于计算SSD模型的算法，该算法结合了线性（循环）模式和二次（注意力）模式。该算法的计算效率与SSM相当（序列长度呈线性增长），并且与注意力机制一样对硬件友好（主要使用矩阵乘法）。

**Theorem 6.1**   Consider an SSD model with state expansion factor $\mathtt{N}$ and head dimension $\mathtt{P}=\mathtt{N}$.
There exists an algorithm for computing the model on any input $X \in R^{\mathtt{(T, P)}}$
which only requires $O(\mathtt{TN}^2)$ training FLOPs, $O(\mathtt{TN})$ inference FLOPs, $O(\mathtt{N}^2)$ inference memory, and whose work is dominated by matrix multiplications.

Note that all of these bounds are tight, because
a state space model with state expansion $\mathtt{N}$ operating on a head size of $\mathtt{N}$ has total state size $\mathtt{N}^2$ (yielding the lower bounds for training and inference FLOPs of $O(\mathtt{TN}^2)$ and $O(\mathtt{N}^2)$ respectively).
Furthermore the input $X$ itself has $\mathtt{TN}$ elements, yielding the memory lower bound.

---
## 一、 符号定义

- $T$：序列长度（Sequence Length）。
- $N$：状态扩展因子（State Expansion Factor）。
- $P$：Head 维度（Head Dimension），定理中假设 $P = N$。
- $X \in \mathbb{R}^{(T, P)}$：输入张量（在 $P=N$ 时尺寸为 $T \times N$）。

## 二、 定理主要指标拆解 (Theorem 6.1)

1. **训练计算量 (Training FLOPs)：$\mathcal{O}(TN^2)$**  
    - 通过 SSD 的分块对偶算法（Chunked Dual Algorithm），训练过程将 SSM 动态映射为类似 Attention 的分块矩阵乘法，可以在 $\mathcal{O}(TN^2)$ 计算复杂度内完成全局并行计算（传统无对偶优化时的朴素实现容易引入更高复杂度）。 
2. **推理计算量 (Inference FLOPs)：$\mathcal{O}(TN)$**  
    - 在自回归生成阶段（单步步进），利用 SSM 的循环展开（Recurrent form），每生成一个 token 仅需 $\mathcal{O}(N^2)$ 计算，计算 $T$ 个 token 累积复杂度为 $\mathcal{O}(TN)$。
3. **推理内存 (Inference Memory)：$\mathcal{O}(N^2)$**
    - 自回归生成时，模型只需要维护一个固定大小的隐状态（Hidden State），其尺寸为 $N \times P = N^2$，内存占用不随序列长度 $T$ 增长（克服了标准 Attention $\mathcal{O}(T)$ KV Cache 随长文本膨胀的问题）。
4. **计算模式：由矩阵乘法为主导 (Matrix Multiplications-dominated)**
    - 算法将状态空间更新的大部分计算转换为 GEMM（通用矩阵乘法），能完美适配 GPU/TPU 等现代硬件的 Tensor Core 加速。  
## 三、 紧界证明与物理意义 (Tight Bounds Note)

段落后半部分解释了为什么这些上界是**渐进紧界（Tight Lower Bounds）**（即不可能再低）：
- **状态空间规模**：每个 head 的输入维度为 $N$，状态扩展因子为 $N$，故完整的状态矩阵维度为 $N \times N$，即隐状态总共有 $N^2$ 个元素。 
    - **推理内存下界**：要维护 $N^2$ 的状态，内存至少为 $\mathcal{O}(N^2)$。
    - **训练 FLOPs 下界**：对 $T$ 个时间步更新 $N^2$ 规模的状态，总算力下界必然是 $\mathcal{O}(T \cdot N^2)$。
- **输入维度下界**：
    - 输入张量 $X$ 包含 $T \times N$ 个元素，仅读取/写入输入数据本身就需要 $\mathcal{O}(TN)$ 的资源，因此推理算力下界为 $\mathcal{O}(TN)$。
    
---
定理 6.1的核心思想仍然是将状态空间模型的计算问题视为半可分矩阵乘法问题，但以一种新的方式利用其结构。我们不再像以往那样在递归模式或注意力模式下计算整个矩阵，而是对矩阵进行分块分解。对角块可以使用双重注意力模式计算，这可以通过矩阵乘法高效完成；而非对角块则可以利用半可分矩阵的秩结构进行分解，并简化为更小的递归式。我们强调，第6.2节 提供了 SSD 算法的独立实现。与一般的选择性 SSM 相比，这种实现方式要简单得多，即使在原生 PyTorch 中也相对高效，无需特殊的底层内核。

To begin, we partition the matrix $M$ into a $\frac{\mathtt{T}}{\mathtt{Q}} \times \frac{\mathtt{T}}{\mathtt{Q}}$ grid of submatrices of size $\mathtt{Q} \times \mathtt{Q}$, for some block size $\mathtt{Q}$.
Note that the off-diagonal blocks are low-rank by the defining property of semiseparable matrices (Def.3.1).

*Note that the block decomposition is valid even with partitions of varying size, e.g.\ if $\mathtt{Q} \not\mid \mathtt{T}$, but we assume even divisibility for simplicity.*

$$\begin{align*}%
  \text{(Block Decomposition)} \quad M &=
  \begin{bmatrix}
    M^{(0,0)} \\
    M^{(1,0)} & M^{(1,1)} \\
    \vdots & \vdots & \ddots \\
    M^{(\mathtt{T}/\mathtt{Q}-1,0)} & M^{(\mathtt{T}/\mathtt{Q}-1,1)} & \dots & M^{(\mathtt{T}/\mathtt{Q}-1,\mathtt{T}/\mathtt{Q}-1)} \\
  \end{bmatrix}
  \\
  \text{(Diagonal Block)} \quad M^{(j,j)} &= \mathsf{SSM}(A_{j\mathtt{Q}:(j+1)\mathtt{Q}}, B_{j\mathtt{Q}:(j+1)\mathtt{Q}}, C_{j\mathtt{Q}:(j+1)\mathtt{Q}}) \\
  \text{(Low-Rank Block)} \quad M^{(j,i)} &= 
  \begin{bmatrix}C_{j\mathtt{Q}}^{\top} A_{j\mathtt{Q}:j\mathtt{Q}-1} \\[1pt] \vdots \\[1pt] C_{(j+1)\mathtt{Q}-1}^{\top} A_{(j+1)\mathtt{Q}-1:j\mathtt{Q}-1}\end{bmatrix}
  A_{j\mathtt{Q}-1:(i+1)\mathtt{Q}-1}
  \begin{bmatrix}B_{i\mathtt{Q}}^{\top} A_{(i+1)\mathtt{Q}-1:i\mathtt{Q}} \\[1pt] \vdots \\[1pt] B_{(i+1)\mathtt{Q}-1}^{\top} A_{(i+1)\mathtt{Q}-1:(i+1)\mathtt{Q}-1}\end{bmatrix}^{\top}
\end{align*}$$

This is easiest illustrated through an example, e.g.\ for $\mathtt{T}=9$ and decomposing into chunks of length $\mathtt{Q}=3$.
The shaded cells are low-rank factorizations of the off-diagonal blocks of the semiseparable matrix.

$$\begin{align}
M &=
\left[
\begin{array}{ccc|ccc|ccc}
C_0^{\top} A_{0:0} B_0 & & & & & & & & \\
C_1^{\top} A_{1:0} B_0 & C_1^{\top} A_{1:1} B_1 & & & & & & & \\
C_2^{\top} A_{2:0} B_0 & C_2^{\top} A_{2:1} B_1 & C_2^{\top} A_{2:2} B_2 & & & & & & \\
\hline
C_3^{\top} A_{3:0} B_0 & C_3^{\top} A_{3:1} B_1 & C_3^{\top} A_{3:2} B_2 & C_3^{\top} A_{3:3} B_3 & & & & & \\
C_4^{\top} A_{4:0} B_0 & C_4^{\top} A_{4:1} B_1 & C_4^{\top} A_{4:2} B_2 & C_4^{\top} A_{4:3} B_3 & C_4^{\top} A_{4:4} B_4 & & & & \\
C_5^{\top} A_{5:0} B_0 & C_5^{\top} A_{5:1} B_1 & C_5^{\top} A_{5:2} B_2 & C_5^{\top} A_{5:3} B_3 & C_5^{\top} A_{5:4} B_4 & C_5^{\top} A_{5:5} B_5 & & & \\
\hline
C_6^{\top} A_{6:0} B_0 & C_6^{\top} A_{6:1} B_1 & C_6^{\top} A_{6:2} B_2 & C_6^{\top} A_{6:3} B_3 & C_6^{\top} A_{6:4} B_4 & C_6^{\top} A_{6:5} B_5 & C_6^{\top} A_{6:6} B_6 & & \\
C_7^{\top} A_{7:0} B_0 & C_7^{\top} A_{7:1} B_1 & C_7^{\top} A_{7:2} B_2 & C_7^{\top} A_{7:3} B_3 & C_7^{\top} A_{7:4} B_4 & C_7^{\top} A_{7:5} B_5 & C_7^{\top} A_{7:6} B_6 & C_7^{\top} A_{7:7} B_7 & \\
C_8^{\top} A_{8:0} B_0 & C_8^{\top} A_{8:1} B_1 & C_8^{\top} A_{8:2} B_2 & C_8^{\top} A_{8:3} B_3 & C_8^{\top} A_{8:4} B_4 & C_8^{\top} A_{8:5} B_5 & C_8^{\top} A_{8:6} B_6 & C_8^{\top} A_{8:7} B_7 & C_8^{\top} A_{8:8} B_8 \\
\end{array}
\right] \\[15pt]

&= 
\left[
\begin{array}{c|c|c}
\begin{matrix}
C_0^{\top} A_{0:0} B_0 & & \\
C_1^{\top} A_{1:0} B_0 & C_1^{\top} A_{1:1} B_1 & \\
C_2^{\top} A_{2:0} B_0 & C_2^{\top} A_{2:1} B_1 & C_2^{\top} A_{2:2} B_2 
\end{matrix} & & \\
\hline
\begin{bmatrix} C_3^{\top} A_{3:2} \\[1pt] C_4^{\top} A_{4:2} \\[1pt] C_5^{\top} A_{5:2} \end{bmatrix} A_{2:2} 
\begin{bmatrix} B_0^{\top} A_{2:0} \\ B_1^{\top} A_{2:1} \\ B_2^{\top} A_{2:2} \end{bmatrix}^{\top} 
& 
\begin{matrix}
C_3^{\top} A_{3:3} B_3 & & \\
C_4^{\top} A_{4:3} B_3 & C_4^{\top} A_{4:4} B_4 & \\
C_5^{\top} A_{5:3} B_3 & C_5^{\top} A_{5:4} B_4 & C_5^{\top} A_{5:5} B_5 
\end{matrix} & \\
\hline
\begin{bmatrix} C_6^{\top} A_{6:5} \\[1pt] C_7^{\top} A_{7:5} \\[1pt] C_8^{\top} A_{8:5} \end{bmatrix} A_{5:2} 
\begin{bmatrix} B_0^{\top} A_{2:0} \\ B_1^{\top} A_{2:1} \\ B_2^{\top} A_{2:2} \end{bmatrix}^{\top} 
& 
\begin{bmatrix} C_6^{\top} A_{6:5} \\[1pt] C_7^{\top} A_{7:5} \\[1pt] C_8^{\top} A_{8:5} \end{bmatrix} A_{5:5} 
\begin{bmatrix} B_3^{\top} A_{5:3} \\ B_4^{\top} A_{5:4} \\ B_5^{\top} A_{5:5} \end{bmatrix}^{\top} 
& 
\begin{matrix}
C_6^{\top} A_{6:6} B_6 & & \\
C_7^{\top} A_{7:6} B_6 & C_7^{\top} A_{7:7} B_7 & \\
C_8^{\top} A_{8:6} B_6 & C_8^{\top} A_{8:7} B_7 & C_8^{\top} A_{8:8} B_8 
\end{matrix} \\
\end{array}
\right]
\end{align}
$$

From here we can reduce the problem into these two parts.
These can also be interpreted as dividing the output of a chunk $y_{j\mathtt{Q}:(j+1)\mathtt{Q}}$ into two components: the effect of inputs within the chunk $x_{j\mathtt{Q}:(j+1)\mathtt{Q}}$, and the effect of inputs before the chunk $x_{0:j\mathtt{Q}}$.

---

## 2. 跨块的 $j > i$ 指的是什么？

这里的 $j > i$ 针对的是**块索引（Chunk Index）**，而不是块内部的 Token 索引：
- **对角块 $M^{(j,j)}$（即 $j = i$）**：处理的是“第 $j$ 个块内部的 Token 互相作用”（例如第 0~2 个 Token 之间的影响，设 $Q=3$）。
- **非对角块 $M^{(j,i)}$（即 $j > i$）**：处理的是“第 $i$ 个块（过去）整体对第 $j$ 个块（未来）整体的影响”（例如第 0~2 个 Token 产生的累积隐状态，如何衰减并传递给第 3~5 个 Token）。

---
## 一、 从 $y = M x$ 的矩阵乘法看“行”与“列”

在 SSD/SSM 或 Transformer 中，输入序列向量为 $X = [x_0, x_1, \dots, x_{T-1}]^\top$，输出序列向量为 $Y = [y_0, y_1, \dots, y_{T-1}]^\top$。它们满足：
$$Y = M X$$
展开来看第 $t_{\text{row}}$ 行的输出 $y_{t_{\text{row}}}$ 是如何计算出来的：
$$y_{t_{\text{row}}} = \sum_{t_{\text{col}}=0}^{T-1} M_{t_{\text{row}}, t_{\text{col}}} \cdot x_{t_{\text{col}}}$$
- **列索引 $t_{\text{col}}$**：对应**输入时间步**（即信息产生的时刻）。
- **行索引 $t_{\text{row}}$**：对应**输出时间步**（即信息被接收/计算的时刻）。
- **矩阵元素 $M_{t_{\text{row}}, t_{\text{col}}}$**：代表“在 $t_{\text{col}}$ 时刻输入的信号 $x_{t_{\text{col}}}$，对 $t_{\text{row}}$ 时刻的输出 $y_{t_{\text{row}}}$ 产生了多大的贡献/影响”。
### 二、 映照到分块矩阵 $M^{(j, i)}$

当我们将全局矩阵 $M$ 切分成 $\frac{T}{Q} \times \frac{T}{Q}$ 个大小为 $Q \times Q$ 的子块时，每一个块 $M^{(j,i)}$ 的两个下标分别继承了行与列的含义：
- **行块索引 $j$**：代表**输出的第 $j$ 个 Chunk**（时间范围为 $[jQ, (j+1)Q - 1]$）。
- **列块索引 $i$**：代表**输入的第 $i$ 个 Chunk**（时间范围为 $[iQ, (i+1)Q - 1]$）。
由于时间的单向流动（因果律）：
1. **当 $j > i$ 时**：行块的时间段段落在列块的时间段**之后**。
    - 例如 $i=0, j=1$，输入属于第 0 块（过去），输出属于第 1 块（未来）。
    - 此时 $M^{(j,i)}$ 填补的就是“**第 $i$ 个块产生的历史信息，在传播到了第 $j$ 个块的时刻后，对第 $j$ 个块产生的效应**”。这就是为什么它被称为“过去对未来的影响”（跨块历史传递）。
2. **当 $j < i$ 时**：代表未来的输入影响过去的输出，因为因果律限制，这些右上角区域全为 $0$。
3. **当 $j = i$ 时**：输入和输出在同一个时间块内，因此处理的是 Chunk 内部的局部依赖。
### 三、 结合物理衰减 $A$ 形象直观理解

在 SSD 公式中，非对角块 $M^{(j,i)}$ 被拆解为：
$$M^{(j,i)} = \text{Readout}_j \cdot \left( \prod_{\text{中间块}} A_{\text{decay}} \right) \cdot \text{State\_Build}_i$$
- **$\text{State\_Build}_i$（右侧项）**：输入在过去的第 $i$ 块内被吸收，合成了一个隐状态 $h$。
- **$\prod A_{\text{decay}}$（中间项）**：这个隐状态随时间跨越 $i+1, i+2, \dots$ 累积衰减并顺着时间流向第 $j$ 块。
- **$\text{Readout}_j$（左侧项）**：在未来的第 $j$ 块，读取这个已经过衰减的过去隐状态并作用到当前的输出上。
这就从数学和物理两个层面完全统一了：**列块索引 $i$ 代表历史源头，行块索引 $j$ 代表未来接收方，下三角非对角块 $j > i$ 自然就代表了过去对未来的影响**。

---
在状态空间模型（SSM）及 SSD（Mamba-2）框架中，$B$ 和 $C$ 分别是**控制隐状态“写入”与“读取”的映射向量/矩阵**。
结合离散状态空间方程可以非常直接地理解它们的数学定义与物理角色：

$$h_t = A_t h_{t-1} + B_t x_t$$

$$y_t = C_t^\top h_t$$
## 一、 详细定义与物理意义

1. **矩阵/向量 $B_t$ （State Input Matrix / Projection Vector）**

    - **物理意义**：**状态写入器（State Input / Write）**。
  
    - **作用**：在时间步 $t$，输入信号为 $x_t$（标量或向量），$B_t$ 的职责是将这个输入映射并“注入”到更高维度的隐状态空间 $h_t$ 中。

    - **维度**：若隐状态空间维度（State Expansion Factor）为 $N$，则 $B_t \in \mathbb{R}^N$。
2. **矩阵/向量 $C_t$ （State Output Matrix / Projection Vector）**

    - **物理意义**：**状态读取器（State Readout / Read）**。
    - **作用**：在时间步 $t$，从积累了历史信息的隐状态 $h_t$ 中，通过转置向量 $C_t^\top$ 提取/投影出当前的输出值 $y_t$。
    - **维度**：$C_t \in \mathbb{R}^N$（故 $C_t^\top \in \mathbb{R}^{1 \times N}$）。
## 二、 在公式矩阵 $M$ 中的物理图景

图中矩阵的每一个元素 $M_{t, s} = C_t^\top A_{t:s} B_s$（其中 $t \ge s$），实际上完整记录了信号从**产生**到**提取**的全过程：
- **$B_s$**：在过去的 $s$ 时刻，将输入 $x_s$ **写入**隐状态。
- **$A_{t:s}$**：隐状态在时间轴上从 $s$ 衰减传播到 $t$ 时刻。
- **$C_t^\top$**：在未来的 $t$ 时刻，将衰减后的隐状态**读取**出来作为输出。

---

## 一、 隐状态的加法去哪了？（从递推到矩阵乘法）

标准的离散状态空间方程为：

$$h_t = A_t h_{t-1} + B_t x_t$$
假定初始状态 $h_{-1} = 0$，将时间步 $t=0, 1, 2$ 的递推展开：

- **$t=0$**： $h_0 = B_0 x_0$

- **$t=1$**： $h_1 = A_1 h_0 + B_1 x_1 = A_{1:0} B_0 x_0 + B_1 x_1$

- **$t=2$**（即第 0 块的最后一个时间步）：
$$h_2 = A_2 h_1 + B_2 x_2 = \mathbf{A_{2:0} B_0 x_0 + A_{2:1} B_1 x_1 + A_{2:2} B_2 x_2}$$

这里有非常清晰的**加法累加（Summation）**。
现在我们看图中矩阵里的这一项转置：

$$\left[ \begin{matrix} B_0^\top A_{2:0} \\ B_1^\top A_{2:1} \\ B_2^\top A_{2:2} \end{matrix} \right]^\top = \begin{bmatrix} (A_{2:0} B_0)^\top & (A_{2:1} B_1)^\top & (A_{2:2} B_2)^\top \end{bmatrix}$$
当这个行向量与输入的序列向量向量 $\mathbf{x}_{\text{chunk 0}} = \begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix}$ 相乘时：

$$\begin{bmatrix} (A_{2:0} B_0)^\top & (A_{2:1} B_1)^\top & (A_{2:2} B_2)^\top \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \\ x_2 \end{bmatrix} = A_{2:0} B_0 x_0 + A_{2:1} B_1 x_1 + A_{2:2} B_2 x_2 = \mathbf{h_2}$$

**矩阵乘法的内积规则自动完成了这三个历史输入项的“求和加法”**，直接计算出了第 0 块末尾的完整隐状态 $h_2$。

转置公式
$$ \begin{bmatrix} X_1 \\ X_2 \\ X_3 \end{bmatrix}^\top = \begin{bmatrix} X_1^\top & X_2^\top & X_3^\top \end{bmatrix} $$
对称矩阵Symmetric Matrix $A=A^\top$ 

---
**$A$ 在 SSD 中是一个标量（Scalar）或对角矩阵（Diagonal Matrix）**

在转置运算中，矩阵乘法的分配律为 $(U V)^\top = V^\top U^\top$。
对于单个项 $B_s^\top A_{t:s}$，将其整体转置：
$$(B_s^\top A_{t:s})^\top = A_{t:s}^\top (B_s^\top)^\top = A_{t:s}^\top B_s$$

在一般矩阵乘法中，如果 $A$ 是普通方阵，那么 $(B_s^\top A_{t:s})^\top = A_{t:s}^\top B_s \ne A_{t:s} B_s$。
但是，在 **Mamba-2 / SSD 的设计中**：
- 衰减因子 $A_k$ 永远是**标量（Scalar）**（或者对角阵 $\text{diag}(a_1, \dots, a_N)$）。
- 对于标量或对角阵，永远有 **$A_{t:s}^\top = A_{t:s}$**。
因此：
$$(B_s^\top A_{t:s})^\top = A_{t:s}^\top B_s = A_{t:s} B_s$$
### 2. 标量/对角阵 $A$ 带来的乘法交换律

除了 $A^\top = A$ 之外，标量/对角阵还满足与向量乘法的**可交换性**：
- $B_s \in \mathbb{R}^{N \times 1}$ 是列向量，$A_{t:s}$ 是标量（或对角缩放）。
- $B_s^\top A_{t:s}$ 是一个行向量（尺寸为 $1 \times N$）。
- 将其转置后，得到尺寸为 $N \times 1$ 的列向量：
$$(B_s^\top A_{t:s})^\top = (A_{t:s} B_s^\top)^\top = B_s A_{t:s} = A_{t:s} B_s$$

---

## 二、 这个矩阵表达与离散 SSM 的物理对应

图中非对角块形如：

$$M^{(j,i)} = \mathbf{L} \cdot \mathbf{A_{decay}} \cdot \mathbf{R}^\top$$

它的三项与 SSM 离散方程的物理组件一一对应：

1. **右因子 $\mathbf{R}^\top$（隐状态构造矩阵 / State Construction）**：
    - **公式**：$\left[ \begin{matrix} B_0^\top A_{2:0} \\ B_1^\top A_{2:1} \\ B_2^\top A_{2:2} \end{matrix} \right]^\top$
    - **物理含义**：负责收集过去第 $i$ 块内所有的输入 $x_s$（乘上 $B_s$），并按时间衰减 $A$ 加和，**在块末尾生成一个汇总的隐状态 $h_{\text{end}}$**。
2. **中间因子 $\mathbf{A_{decay}}$（跨块传递 / Inter-chunk Decay）**：

    - **公式**：图中高亮的 $A_{2:2}, A_{5:2}$ 等

    - **物理含义**：把第 $i$ 块末尾的隐状态 $h_{\text{end}}$ 沿时间轴乘上连续衰减矩阵，**顺推传递**给未来第 $j$ 块的开头。

3. **左因子 $\mathbf{L}$（状态读取矩阵 / State Readout）**：
    - **公式**：$\begin{bmatrix} C_3^\top A_{3:2} \\ C_4^\top A_{4:2} \\ C_5^\top A_{5:2} \end{bmatrix}$

    - **物理含义**：在未来第 $j$ 块的各个时间步，把传递过来的隐状态**按 $C_t^\top$ 提取**成输出 $y_t$。

---
# 6.1 Diagonal Blocks
The diagonal blocks are easy to handle, because they are simply self-similar problems of a smaller size.
The $j$-th block represents computing the answer $\mathsf{SSM}(A_R, B_R, C_R)(x_R)$ for the range $R = j\mathtt{Q}:(j+1)\mathtt{Q} = (j\mathtt{Q}, j\mathtt{Q}+1, \dots, j\mathtt{Q}+\mathtt{Q}-1)$.
The key is that this block can be computed using any desired method.
In particular, for small chunk lengths $\mathtt{Q}$, 比如q=64， this problem is computed more efficiently using the dual quadratic SMA结构化掩码注意力 form.
Additionally, the chunks can be computed in parallel.
These subproblems can be interpreted as: what is the output per chunk supposing that the initial state (to the chunk) is $0$.
In other words for chunk $j$, this computes the correct outputs taking into account only the chunk inputs $x_{j\mathtt{Q}:(j+1)\mathtt{Q}}$.

---
各个对角块（Diagonal Block $M^{(j,j)}$）在初始时假设其隐状态为 $0$，**并不是因为在真实的全局模型中历史信息不存在，而是一种为了实现大规模并行计算（Parallelization）所采取的“数学解耦”策略与算法设计**。
### 一、 算子线性可加性（Linear Linearity）的数学支持

状态空间模型（SSM）本质上是一个**线性时变/时不变系统**。对于线性系统，**叠加原理（Superposition Principle）** 永远成立。

对第 $j$ 个 Chunk 而言，它的输出 $Y_{\text{chunk } j}$ 可以严格拆分为两个独立部分的线性叠加：

$$Y_{\text{chunk } j} = \text{Output}_{\text{Zero-State}} + \text{Output}_{\text{Zero-Input}}$$

1. **Zero-State Output（零状态响应）**：**假设初始状态 $h_{\text{init}} = 0$**，仅由当前 Chunk 内部的输入 $X_j$ 产生的输出。这正好就是对角块 $M^{(j,j)} X_j$ 的计算。

2. **Zero-Input Output（零输入响应）**：**假设当前 Chunk 输入为 $0$**，仅由前一个 Chunk 传递过来的末尾隐状态 $h_{\text{prev}}$ 经过时间衰减后产生的输出。这恰好对应非对角块的跨块补偿 $\sum_{i < j} M^{(j,i)} X_i$。

因为这两种响应完全可以独立计算再相加，所以我们可以**先放心地把初始隐状态设为 $0$** 算局部输出，后续再加上历史补偿。
### 二、 打断时间串行依赖，解锁 GPU 并行度

在传统的 SSM 或 RNN 递推中，计算存在极强的顺序依赖（Sequential Dependency）： 必须先算出 $h_0 \to h_1 \to \dots \to h_{Q-1}$，拿到第 0 块末尾的 $h_{Q-1}$，才能开始计算第 1 块。这导致序列长度 $T$ 很大时，GPU 的并行计算能力被严重束缚。

通过强制假设每个 Chunk 的 $h_{\text{init}} = 0$：

- **所有 Chunk 彻底解耦**：第 0 块、第 1 块……第 $\frac{T}{Q}-1$ 块之间不再有任何先后的等待关系。

- **矩阵乘法极速并行**：GPU 可以同时将所有 Chunk 的 $Q \times Q$ 小掩码矩阵与各自的输入向量并行相乘（利用二次方的对偶 Attention 形式与 Tensor Core 加速），大幅提升训练吞吐量。

### 三、 减法/清零简化了局部对偶注意力（SMA）的计算

在把块内视作结构化掩码注意力（SMA）时，$M^{(j,j)}$ 矩阵表达的是 **Chunk 内部 Token 之间的相对因果作用**。

如果保留上一个 Chunk 留下的 $h_{\text{init}}$：

- 块内的每一个 Token 不仅要计算与同块内过去 Token 的交互，还要时刻挂载一个全局递推向量。

- 这会让块内原本形式优雅的二次方矩阵运算（类似 $Q K^\top$ 的局域 Attention 乘法）变得杂乱，无法直接套用 FlashAttention 类的硬件优化 Kernel。

把 $h_{\text{init}}$ 置 0 后，块内计算变成了一个标准的、自包含的 $Q \times Q$ 局部注意力问题，计算模式非常干净。

---

### 一、 概念的严格定义对比

在系统理论中，对于任意时间块 $j$（包含时间步区间 $t \in [jQ, (j+1)Q-1]$）：
1. **“零状态响应”（Zero-State Response）**  
    - **前提条件**：进入该块时刻的**初始隐状态为零**（即 $h_{jQ-1} = 0$）。
    - **物理意义**：当前块内产生的输出 **完全且仅由该块内部的输入 $x_{jQ:(j+1)Q-1}$ 引起**。

2. **“零输入响应”（Zero-Input Response）**

    - **前提条件**：该块内部的**所有输入全为零**（即该块内 $x = 0$）。

    - **物理意义**：当前块内产生的输出 **完全由从过去历史传递过来的初始隐状态 $h_{jQ-1}$ 引起**。
### 二、 为什么零状态响应“刚好”落进对角块？

我们来看对角块 $M^{(j,j)}$ 对应的输入和输出区间：

- **输出（行索引）**：第 $j$ 块的输出向量 $Y_{\text{chunk } j}$，时间步为 $t \in [jQ, (j+1)Q-1]$。

- **输入（列索引）**：第 $j$ 块的输入向量 $X_{\text{chunk } j}$，时间步为 $s \in [jQ, (j+1)Q-1]$。

当计算矩阵乘法 $Y_{\text{chunk } j} = M^{(j,j)} X_{\text{chunk } j}$ 时：

- 输入数据**只有 $X_{\text{chunk } j}$**（没有任何来自第 $i < j$ 块的过去的输入 $X_i$）。

- 隐状态更新完全从 $t = jQ$ 开始从零递推。

这在数学上完全符合“初始状态 $h = 0$ 且仅输入当前块信号”的零状态响应定义。

### 三、 如果包含“非零初始状态”，矩阵会变成什么样？

我们可以对比一下，包含历史隐状态 $h_{jQ-1}$ 时，第 $j$ 块内部第 $m$ 个 Token（本地索引 $0 \le m < Q$）的输出是如何计算的：

$$y_{jQ+m} = \underbrace{C_{jQ+m}^\top A_{jQ+m : jQ-1} \cdot \mathbf{h_{jQ-1}}}_{\text{非零初始状态的影响（由过去的块决定）}} + \underbrace{\sum_{n=0}^{m} C_{jQ+m}^\top A_{jQ+m : jQ+n} B_{jQ+n} \cdot x_{jQ+n}}_{\text{零状态响应（仅由当前块内的输入决定）}}$$

仔细看后半部分（零状态响应项）：


- 它只包含了当前块内部从 $n=0$ 到 $n=m$ 的输入 $x_{jQ+n}$。

- 将这部分独立提取出来写成矩阵，它的行 index 和列 index 都在范围 $[0, Q-1]$ 内，对应构成了一个大小为 $Q \times Q$ 的下三角矩阵。

**这个下三角矩阵就是对角块 $M^{(j,j)}$**。
而前半部分（包含 $h_{jQ-1}$ 的项），由于 $h_{jQ-1}$ 是由过去所有块的输入累积形成的（即 $h_{jQ-1} = \sum_{i < j} \dots X_i$），它本质上是把过去那些块的输入通过非对角块 $M^{(j,i)}$ 映射过来的。

---

# 6.2 Low-Rank Blocks
The low-rank factorizations consist of 3 terms, and there are correspondingly three pieces of the computation.
In this factorization, we will use the terminology
- The terms like $\begin{bmatrix}B_0^{\top} A_{2:0} \\[1pt] B_1^{\top}A_{2:1} \\[1pt] B_2^{\top}A_{2:2}\end{bmatrix}^{\top}$ are called the right factors or $B$-block-factors.
- The terms like $A_{5:2}$ are called the center factors or $A$-block-factors.
- The terms like $\begin{bmatrix}C_6^{\top} A_{6:5} \\[1pt] C_7^{\top} A_{7:5} \\[1pt] C_8^{\top} A_{8:5}\end{bmatrix}$ are called the left factors or $C$-block-factors.

**Right Factors.** This step computes the multiplication by the right $B$-block-factors of the low-rank factorization.
Note that for each chunk, this is a $(\mathtt{N}, \mathtt{Q})$ by $(\mathtt{Q}, \mathtt{P})$ matrix multiplication, where $\mathtt{N}$ is the state dimension and $P$ is the head dimension.
The result is a $(\mathtt{N}, \mathtt{P})$ tensor for each chunk, which has the same dimensionality as the expanded hidden state $h$.
**This can be interpreted as: what is the final state per chunk supposing that the initial state (to the chunk) is $0$.**
**In other words this computes $h_{j\mathtt{Q}+\mathtt{Q}-1}$ assuming that $x_{0:j\mathtt{Q}} = 0$.**

**Center Factors** This step computes the effect of the center $A$-block-factors terms in the low-rank factorization.
In the previous step, the final states per chunk have total shape $\mathtt{(\mathtt{T}/\mathtt{Q},\mathtt{N},\mathtt{P})}$.
This is now multiplied by a 1-SS matrix generated by $A_{2\mathtt{Q}-1:\mathtt{Q}-1}^\times, A_{3\mathtt{Q}-1:2\mathtt{Q}-1}^\times, \dots, A_{\mathtt{T}-1:\mathtt{T}-\mathtt{Q}-1}^\times$.
chunk0是从0到Q-1 1是从Q到2Q-1，依次类推
This step can be computed by any algorithm for computing 1-SS multiplication (also known as the scalar SSM scan or $\texttt{cumprodsum}$ operator).

This can be interpreted as: what is the actual final state per chunk taking into account all previous inputs;
in other words, this computes the true hidden state $h_{j\mathtt{Q}}$ taking into account all of $x_{0:(j+1)\mathtt{Q}}$.

**Left Factors.** This step computes the multiplication by the left $C$-block-factors of the low-rank factorization.
For each chunk, this can be represented by a matrix multiplication
$\mathsf{contract}(\mathtt{QN,NP \to QP})$.

This can be interpreted as: what is the output per chunk taking into account the correct initial state $h_{j\mathtt{Q}-1}$, and supposing the inputs $x_{j\mathtt{Q}:(j+1)\mathtt{Q}}$ are $0$.
In other words for chunk $j$, this computes the correct outputs taking into account only the prior inputs $x_{0:j\mathtt{Q}}$.

----
# 6.3 Computational Cost
We define the notation $\mathsf{BMM}(\mathtt{B}, \mathtt{M}, \mathtt{N}, \mathtt{K})$ to define a batched matrix multiplication $\mathsf{contract}(\mathtt{\mathtt{MK},\mathtt{KN} \to \mathtt{MN}})$ with batch dimension $\mathtt{B}$.

From this notation we can infer three aspects of the efficiency:
- Computation cost: total of $O(\mathtt{BMNK})$ FLOPs.
- Memory cost: total of $O(\mathtt{B}(\mathtt{MK}+\mathtt{KN}+\mathtt{MN}))$ space.
- Parallelization: larger $\mathtt{M}, \mathtt{N}, \mathtt{K}$ terms can leverage specialized matrix multiplication units on modern accelerators.
**Center Blocks.**
The cost of the quadratic SMA computation consists of three steps 
- Computing the kernel matrix $C^{\top} B$, which has cost $\mathsf{BMM}(\mathtt{T}/\mathtt{Q}, \mathtt{Q}, \mathtt{Q}, \mathtt{N})$.
- **Multiplying by the mask matrix, which is an elementwise operation on tensors of shape $(\mathtt{T}/\mathtt{Q}, \mathtt{Q}, \mathtt{Q})$.**说明是标量
- Multiplying by the $X$ values, which has cost $\mathsf{BMM}(\mathtt{T}/\mathtt{Q}, \mathtt{Q}, \mathtt{P}, \mathtt{N})$
**Low-Rank Blocks: Right Factors.**
This step is a **single matrix multiplication** with cost $\mathsf{BMM}(\mathtt{T}/\mathtt{Q}, \mathtt{N}, \mathtt{P}, \mathtt{Q})$.

**Low-Rank Blocks: Center Factors.**

This step is a scalar SSM scan (or 1-SS multiplication) of length $\mathtt{T}/\mathtt{Q}$ on $(\mathtt{N}, \mathtt{P})$ independent channels.
The work of this scan is $\mathtt{TNP}/\mathtt{Q}$, which is negligible compared to the other factors.

Note that because of the blocking which reduces the length of the sequence from $\mathtt{T}$ to $\mathtt{T}/\mathtt{Q}$,
this scan has $\mathtt{Q}$ times smaller cost than a pure SSM scan (e.g. the selective scan of Mamba).

Thus we observe that on most problem lengths,
other algorithms  may be more efficient or much easier to implement without a significant slowdown.

**For example, a naive implementation of this via 1-SS matrix multiplication has cost $\mathsf{BMM}(1, \mathtt{T}/\mathtt{Q}, \mathtt{NP}, \mathtt{T}/\mathtt{Q})$,**
**which is much easier to implement and can be more efficient than a naive recurrence/scan implementation.**

Low-Rank Blocks: Left Factors.
This step is a single matrix multiplication with cost $\mathsf{BMM}(\mathtt{T}/\mathtt{Q}, \mathtt{Q}, \mathtt{P}, \mathtt{N})$.

**Total Cost.**
If we set $\mathtt{N} = \mathtt{P} = \mathtt{Q}$ (in other words the state dimension, head dimension, and chunk length are equal),
then all BMM terms above become $\mathsf{BMM}(\mathtt{T}/\mathtt{N}, \mathtt{N}, \mathtt{N}, \mathtt{N})$.
The computational chacteristics of this are:
- Total FLOP count of $O(\mathtt{TN}^2)$.
- Total memory of $O(\mathtt{TN})$.
- The work consists primarily of matrix multiplications on matrices of shape $(\mathtt{N}, \mathtt{N})$.
Notice that the memory consumption is tight; the inputs and outputs $x, y$ have shape $(\mathtt{T}, \mathtt{P}) = (\mathtt{T}, \mathtt{N})$.
Meanwhile the flop count reflects an extra factor of $\mathtt{N}$, which is cost incurred发生（费用、损失等） by the autoregressive state size and is common to all models.

Aside from the matmuls, there is a scalar SSM scan on $\mathtt{NP} = \mathtt{N}^2$ features and sequence length $\mathtt{T}/\mathtt{Q}$.

This has cost $O(\mathtt{T}/\mathtt{Q} \mathtt{N}^2)$ FLOPs and $O(\log(\mathtt{T}/\mathtt{Q}))$ depth.
Although it does not use matrix multiplications, it is still parallelizable and the total work done is negligible compared to the other steps; this has a negligible 微不足道 cost in our GPU implementation.

**Comparison to Pure SSM and Attention Models.**

二次注意力机制仅利用矩阵乘法，因此硬件效率非常高，但它也存在一些问题。but has $\mathtt{T}^2 N$ total FLOPs.其在训练和推理阶段较慢的计算速度可以直接看作是其状态规模较大的结果——标准注意力机制的状态规模随序列长度T而变大。因为它会缓存历史记录，并且不会压缩状态。

Linear SSMs have $\mathtt{TNP} = \mathtt{TN}^2$ total FLOPs, which is the same as SSD.
However, a naive implementation requires a state expansion eq.15a that materializes extra memory,这会产生额外的内存
and a scalar operation eq.15b that does not leverage matrix multiplications.这不利用矩阵乘法。

$$
\begin{array}{llll}
\hline
& \text{Attention} & \text{SSM} & \textbf{SSD} \\
\hline
\text{State size} & \mathtt{T} & \mathbf{N} & \mathbf{N} \\
\text{Training FLOPs} & \mathtt{T}^2\mathtt{N} & \mathbf{T}\mathbf{N}^2 & \mathbf{T}\mathbf{N}^2 \\
\text{Inference FLOPs} & \mathtt{T}\mathtt{N} & \mathbf{N}^2 & \mathbf{N}^2 \\
\hline
\text{(Naive) memory} & \mathtt{T}^2 & \mathtt{T}\mathbf{N}^2 & \mathbf{T}\mathbf{N} \\
\text{Matrix multiplication} & \checkmark & & \checkmark \\
\hline
\end{array}
$$

**我们注意到，还有许多其他的矩阵分解方式（例如，参见附录 B，其中汇总了通过不同结构化矩阵分解实现 1-SS 乘法的算法），这可能会衍生出更多适用于其他特定场景的 SSD 算法。更广泛地说，我们注意到，半可分矩阵的研究文献非常丰富，除了我们使用的 SSS 形式（定义 3.2）之外，还有许多其他表示形式，甚至可能存在更高效的算法。**
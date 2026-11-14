# 3. State Space Models are Structured Matrices
## 3.1状态空间模型的矩阵变换形式

Our theoretical framework starts by simply writing this transformation as a matrix multiplication mapping the vectors $x \in R^\mathtt{T} \mapsto y \in R^\mathtt{T}$.
By definition, $h_0 = B_0 x_0$. 隐状态
By induction,数学归纳法
$$\begin{align*}
  h_t &= A_t \dots A_1 B_0 x_0 + A_t \dots A_2 B_1 x_1 + \dots + A_t A_{t-1} B_{t-2} x_{t-2} + A_t B_{t-1} x_{t-1} + B_t x_t
    \\&= \sum_{s=0}^t A_{t:s}^\times B_s x_s
    .
\end{align*}$$
输出是隐状态通过输出投影矩阵$C_t^\top$ 得到的结果，将整条序列的输入向量和输出向量向量化，整个线性系统映射可以表达为一个整体的矩阵乘法，矩阵M是下三角矩阵，因果掩码，表示时间步i的输入对时间步j的输出的影响。
Multiplying by $C_t$ to produce $y_t$ and vectorizing the equation over $t \in [\mathtt{T}]$, we derive the matrix transformation form of SSMs. 公式3
  $$\begin{aligned}
    y_t &= \sum_{s=0}^t C_t^{\top} A_{t:s}^\times B_s x_s
    \\
    y &= \mathsf{SSM}(A, B, C)(x) = Mx
    \\
    M_{ji} &\coloneqq C_j^{\top} A_{j} \cdots A_{i+1} B_{i}
  \end{aligned}$$
  $$M_{ji} = \begin{cases} 
C_j^\top A_{j:i}^\times B_i = C_j^\top A_j A_{j-1} \dots A_{i+1} B_i & \text{if } j > i \\ 
C_i^\top B_i & \text{if } j = i \\ 
0 & \text{if } j < i 
\end{cases}$$
$$M = \begin{bmatrix} 
C_0^\top B_0 & 0 & 0 & \dots & 0 \\ 
C_1^\top A_1 B_0 & C_1^\top B_1 & 0 & \dots & 0 \\ 
C_2^\top A_2 A_1 B_0 & C_2^\top A_2 B_1 & C_2^\top B_2 & \dots & 0 \\ 
\vdots & \vdots & \vdots & \ddots & \vdots \\ 
C_{T-1}^\top A_{T-1:0}^\times B_0 & \dots & \dots & \dots & C_{T-1}^\top B_{T-1} 
\end{bmatrix}$$
## 3.2 Semiseparable Matrices
$M$ in equation 3 is a particular representation of a class of matrices known as semiseparable matrices.

**Definition 3.1**  A (lower triangular) matrix $M$ is $\mathtt{N}$-semiseparable if every submatrix contained in the lower triangular portion (i.e.\ on or below the diagonal) has rank秩 at most $\mathtt{N}$. We call $\mathtt{N}$ the order or rank of the semiseparable matrix.
*矩阵的秩（Rank）是矩阵中线性无关Linear Independence的行数或列数的极大值。*

定义 3.1以及其他相关的“可分离”结构形式（例如拟可分离矩阵和其他半可分离矩阵的定义）有时被称为结构化秩矩阵（或秩结构矩阵），因为它们的特征在于其子矩阵的秩条件。半可分离矩阵有许多结构化表示，包括层次半可分离 (HSS)、顺序半可分离 (SSS) 和 Bruhat 形式，我们将主要使用SSS。

结构化秩矩阵（Rank-Structured Matrices）特定位置的子矩阵（小方块）的秩被限制在一个很小的常数以内（低秩）。在计算机里，处理一个普通的NN大矩阵需要O(N^3)的计算量，但如果它是“结构化秩”的，我们就可以利用它子矩阵低秩的特性，把计算量降低到O(N)

可分离、拟可分离、半可分离（Semi-separable, Quasiseparable） 这些是结构化秩矩阵的具体家族成员。它们主要的区别在于“对角线以外的部分如何低秩”。半可分离矩阵： 粗略地说，是指矩阵的主对角线以上（右上角）或以下（左下角）任意切出一个子矩阵，这个子矩阵的秩最大只有 1（或者一个很小的常数 r)

HSS、SSS、Bruhat形式（结构化表示法)
虽然我们知道这些矩阵的子矩阵秩很低，但怎么在计算机里用最省内存、最快的方式把它们“表达”出来呢？科学家们发明了不同的存储和组织数据的方法（可以理解为不同的数据结构）：

HSS (Hierarchical Semi-Separable，层次半可分离)： 像树状图（分层）一样去切分和表示矩阵。

SSS (Sequentially Semi-Separable，顺序半可分离)： 像一条流水线（顺序）一样，用一系列小矩阵前后相连地去表示大矩阵。

## 3.2.1 The Sequentially Semiseparable (SSS) Representation

**Definition 3.2** A lower triangular matrix $M \in R^{\mathtt{(T,T)}}$ has a $\mathtt{N}$-$\textbf{sequentially semiseparable (SSS)}$ representation if it can be written in the form 公式4
$$\begin{equation}
    M_{ji} = C_j^{\top} A_{j} \cdots A_{i+1} B_{i}
  \end{equation}$$
for vectors $B_{0}, \dots, B_{\mathtt{T}-1}, C_{0}, \dots, C_{\mathtt{T}-1} \in R^{\mathtt{N}}$ and matrices $A_{0}, \dots, A_{\mathtt{T}-1} \in R^{\mathtt{(N,N)}}$.
We define the operator $\mathsf{SSS}$ so that $M = \mathsf{SSS}(A_{0:\mathtt{T}}, B_{0:\mathtt{T}}, C_{0:\mathtt{T}})$.
半可分矩阵的一个基本结果是，它们与具有 SSS 表示的矩阵完全等价。一个方向可以通过一个简单的构造性证明来推导。

**Lemma 3.3** An $\mathtt{N}$-SSS matrix $M$ with representation 4 is $\mathtt{N}$-semiseparable. 那么他在主对角线下方的任意子块的秩最多为N，低秩

*Poof* Consider any off-diagonal block非对角 $M_{j:j', i':i}$ where $j' > j \ge i > i'$. This has an explicit rank-$\mathtt{N}$ factorization as 公式5低秩因子分解，
$$  \begin{equation}
    \label{eq:sss-rank-factor}
    \begin{bmatrix}
      C_j^{\top} A_{j:i'}^\times B_{i'}         & \dots & C_j^{\top} A_{j:i-1}^\times B_{i-1}   \\
      \vdots                                    &       &       \vdots      \\
      C_{j'-1}^{\top} A_{j'-1:i'}^\times B_{i'} & \dots & C_{j'-1}^{\top} A_{j'-1:i-1}^\times B_{i-1} \\
    \end{bmatrix}
    =
    \begin{bmatrix} C_j^{\top} A_{j:j}^\times \\ \vdots \\ C_{j'-1}^{\top} A_{j'-1:j}^\times \end{bmatrix}
    A_{j:i-1}^\times
    \begin{bmatrix} A_{i-1:i'}^\times B_{i'} & \cdots & A_{i-1:i-1}^\times B_{i-1} \end{bmatrix}
    .
  \end{equation}$$
证毕 公式（5）将在推导序列模型的快速算法时被广泛使用。另一方面，关于半可分矩阵的研究文献中已有充分的研究成果。

非对角块（Off-diagonal Block） 是指将一个大矩阵按行和列划分为多个小矩阵（即“子块”/“块矩阵”）后，不在主对角线位置上的那些子块。

---
## Remark1: Lemma推导

**这里把时间索引拆分和矩阵累乘符号混在了一起**我们可以从**物理直觉**和**符号展开**两个层面逐层剥开，看清它是怎么断开的。

---
 **1.先看物理直觉**
假设我们在处理一个长序列，时间轴从左到右延伸。

* **列索引 $c$**：属于“过去”的输入时间段（范围在 $i'$ 到 $i-1$ 之间）。
* **行索引 $r$**：属于“未来”的输出时间段（范围在 $j$ 到 $j'$ 之间）。
* **索引顺序**：满足 $c \le i-1 < j \le r$。
从时间 $c$ 演化到时间 $r$，状态转移矩阵 $A$ 需要经历三个阶段：
$$\underbrace{\text{时间 } c}_{\text{过去的某点}} \xrightarrow{\quad \text{阶段 1: } A_{i-1:c}^{\times} \quad} \underbrace{\text{时间 } i-1}_{\text{过去块的终点}} \xrightarrow{\quad \text{阶段 2: } A_{j:i-1}^{\times} \quad} \underbrace{\text{时间 } j}_{\text{未来块的起点}} \xrightarrow{\quad \text{阶段 3: } A_{r:j}^{\times} \quad} \underbrace{\text{时间 } r}_{\text{未来的某点}}$$

* **阶段 1 ($A_{i-1:c}^{\times}$)**：从输入点 $c$ 步进演化到过去块的末尾 $i-1$。
* **阶段 2 ($A_{j:i-1}^{\times}$)**：**关键桥梁**，从过去块的末尾 $i-1$ 一口气跨越到未来块的开头 $j$。**注意：这个阶段与具体的 $r$ 和 $c$ 无关！**
* **阶段 3 ($A_{r:j}^{\times}$)**：从未来块的开头 $j$ 步进演化到输出点 $r$。
---

**2.再看具体的矩阵乘积展开**

根据之前的定义，$A_{t:s}^{\times}$ 代表从时间步 $s+1$ 到时间步 $t$ 的矩阵连乘：
$$A_{t:s}^{\times} = A_t A_{t-1} A_{t-2} \dots A_{s+1}$$
现在我们要展开从 $c$ 到 $r$ 的连乘矩阵 $A_{r:c}^{\times}$：
$$A_{r:c}^{\times} = A_r A_{r-1} \dots A_1 A_0 \quad (\text{其中包含了所有从 } c+1 \text{ 到 } r \text{ 的 } A \text{ 矩阵})$$
由于矩阵乘法满足**结合律**，我们在时间轴上插入两个固定节点 $j$ 和 $i-1$：
$$A_{r:c}^{\times} = \underbrace{(A_r A_{r-1} \dots A_{j+1})}_{\text{第一段：从 } j \text{ 到 } r} \cdot \underbrace{(A_j A_{j-1} \dots A_i)}_{\text{第二段：从 } i-1 \text{ 到 } j} \cdot \underbrace{(A_{i-1} A_{i-2} \dots A_{c+1})}_{\text{第三段：从 } c \text{ 到 } i-1}$$

利用累乘符号把这三段分别打包写回：
1. 第一段 $(A_r \dots A_{j+1}) = A_{r:j}^{\times}$
2. 第二段 $(A_j \dots A_i) = A_{j:i-1}^{\times}$
3. 第三段 $(A_{i-1} \dots A_{c+1}) = A_{i-1:c}^{\times}$
三者相乘，就得到了图片中的等式：
$$A_{r:c}^{\times} = A_{r:j}^{\times} A_{j:i-1}^{\times} A_{i-1:c}^{\times}$$
---
**3.为什么要费尽心思这么“断开”？**

注意看等式右边的中间项：**$A_{j:i-1}^{\times}$**。

在这个子块中，不管你取哪一行（变动 $r$）或者哪一列（变动 $c$），**中间这一项 $A_{j:i-1}^{\times}$ 都是完全相同的固定矩阵**（维度为 $N \times N$）。

这就实现了**变量分离**：

* 含有 $r$ 的部分（行变量）：提到了左边 $A_{r:j}^{\times}$

* 含有 $c$ 的部分（列变量）：提到了右边 $A_{i-1:c}^{\times}$

* 中间夹着一个维数为 $N \times N$ 的固定桥梁矩阵 $A_{j:i-1}^{\times}$

当对整个子块里的所有元素同时做这个拆分时，整个子块矩阵就可以写成：
$$\text{子块} = (\text{只跟行 } r \text{ 有关的纵向矩阵}) \times \mathbf{A_{j:i-1}^{\times}} \times (\text{只跟列 } c \text{ 有关的横向矩阵})$$

由于中间项的维数只有 $N \times N$（隐藏状态维度，如 $N=64$），因此不论这个子块有多大（即使 $T$ 达到几万），这个子块的**秩（Rank）绝对不会超过 $N$**。这就是 Lemma 3.3 证明低秩分解的关键所在！

---
为了彻底搞清楚**矩阵连乘**是如何把行变量 $r$（未来时间步）和列变量 $c$（过去时间步）解耦并联系起来的，我们需要看一看**矩阵乘法的定义（元素级乘法）以及 矩阵的分块构造。
我们可以分两步，从**单个元素的计算推导到整个子块矩阵的乘法表示。
### 一、 先看单个元素的计算过程 ($M_{r,c}$)
在非对角块中，第 $r$ 行、第 $c$ 列的元素 $M_{r,c}$ 的标量定义为：
$$M_{r,c} = C_r^{\top} A_{r:c}^{\times} B_c$$
将上一问中的连乘拆分 $A_{r:c}^{\times} = A_{r:j}^{\times} A_{j:i-1}^{\times} A_{i-1:c}^{\times}$ 代入进来：

$$M_{r,c} = C_r^{\top} \left( A_{r:j}^{\times} A_{j:i-1}^{\times} A_{i-1:c}^{\times} \right) B_c$$

注意看这里的结合顺序。我们利用矩阵乘法的结合律，把这个标量重新组合：
$$M_{r,c} = \underbrace{\left( C_r^{\top} A_{r:j}^{\times} \right)}_{1 \times N \text{ 行向量}} \cdot \underbrace{\left( A_{j:i-1}^{\times} \right)}_{N \times N \text{ 核心矩阵}} \cdot \underbrace{\left( A_{i-1:c}^{\times} B_c \right)}_{N \times 1 \text{ 列向量}}$$
在这三个部分中：
1. **左侧行向量**：$\left(C_r^{\top} A_{r:j}^{\times}\right) \in \mathbb{R}^{1 \times N}$ —— **只与行索引 $r$ 有关**（表示在时间步 $r$ 的观测输出投影）。
2. **中间矩阵**：$A_{j:i-1}^{\times} \in \mathbb{R}^{N \times N}$ —— **固定常数矩阵**（与 $r$ 和 $c$ 都无关，仅依赖于两个块之间的边界跨度）。
3. **右侧列向量**：$\left(A_{i-1:c}^{\times} B_c\right) \in \mathbb{R}^{N \times 1}$ —— **只与列索引 $c$ 有关**（表示在时间步 $c$ 的控制输入演化）。
### 二、 从单个元素组合成整个子块矩阵 ($M_{j':j, i':i}$)

现在，我们要把这个非对角块里的所有元素（$r$ 从 $j$ 到 $j'$，$c$ 从 $i'$ 到 $i-1$）排成一个完整的矩阵。
假设行范围包含 $h$ 个时间步，列范围包含 $w$ 个时间步。通过将上述所有 $r$ 的行向量**纵向堆叠**，所有 $c$ 的列向量**横向拼合**，我们可以将整个子块显式地写成**三个矩阵相乘**的形式：
$$M_{j':j, \, i':i} = \mathbf{L} \cdot A_{j:i-1}^{\times} \cdot \mathbf{R}$$

其中：
1.左因子矩阵 $\mathbf{L}$（只与行变量 $r$ 有关）
把所有可能行 $r \in [j, j']$ 的左侧向量按行堆叠在一起：
$$\mathbf{L} = \begin{bmatrix} C_j^{\top} A_{j:j}^{\times} \\ C_{j+1}^{\top} A_{j+1:j}^{\times} \\ \vdots \\ C_{j'}^{\top} A_{j':j}^{\times} \end{bmatrix} \quad \in \mathbb{R}^{h \times N}$$
- **形状**：$(h \times N)$，即 行数 $\times$ 状态维度 $N$。
- **特点**：每一行只包含了对应的行索引 $r$。
2.中间衰减矩阵 $A_{j:i-1}^{\times}$（跨块状态转移核）

$$A_{j:i-1}^{\times} \quad \in \mathbb{R}^{N \times N}$$
- **形状**：$(N \times N)$。
- **特点**：一个完全固定的方阵，它不随子块内部的具体行 $r$ 或列 $c$ 改变。
3.右因子矩阵 $\mathbf{R}$（只与列变量 $c$ 有关）
把所有可能列 $c \in [i', i-1]$ 的右侧向量按列横向排列在一起：
$$\mathbf{R} = \begin{bmatrix} A_{i-1:i'}^{\times} B_{i'} & A_{i-1:i'+1}^{\times} B_{i'+1} & \dots & A_{i-1:i-1}^{\times} B_{i-1} \end{bmatrix} \quad \in \mathbb{R}^{N \times w}$$
- **形状**：$(N \times w)$，即 状态维度 $N \times$ 列数。
- **特点**：每一列只包含了对应的列索引 $c$。
### 三、 总结：矩阵尺寸与低秩几何含义
将这三个矩阵的维度拼接在一起看：
$$\underbrace{M_{j':j, \, i':i}}_{(h \times w)} = \underbrace{\mathbf{L}}_{(h \times N)} \times \underbrace{A_{j:i-1}^{\times}}_{(N \times N)} \times \underbrace{\mathbf{R}}_{(N \times w)}$$

1. **维度匹配**：根据矩阵乘法规则，$(h \times N) \times (N \times N) \times (N \times w) = (h \times w)$，正好填满大小为 $h \times w$ 的非对角子块。
2. **秩的限制（Rank Restriction）**：因为整个乘积中间穿过了一个维度仅仅为 $N$ 的“狭窄通道”（$N \times N$ 矩阵），无论子块尺寸 $h$ 和 $w$ 有多大（即使序列长达数万），这个矩阵乘积的**最高秩（Rank）绝对不会超过 $N$**。多个矩阵相乘后，最终结果的“秩”（即有效信息的维度/线性无关的行或列的最大数量）绝对不会超过其中任何一个中间矩阵的“秩”。即便输出空间的大小是 h但由于中间所有的数据都必须完全挤过一个只有 N维的狭窄通道，最终生成的h维向量在几何空间中，只能分布在一个最多只有N维的子空间中
3. **计算优化（SSD / Mamba-2 的底层核心）**： 在做矩阵向量乘法 $y = M x$ 时，我们**不需要**算成完整的 $h \times w$ 大矩阵再乘以 $x$，而是利用这种分解按顺序乘：
    $$\text{Block Output} = \mathbf{L} \left( A_{j:i-1}^{\times} \left( \mathbf{R} x_{\text{block}} \right) \right)$$
    计算复杂度从直接相乘的 $\mathcal{O}(h \cdot w)$ 降到了 $\mathcal{O}((h + w) \cdot N)$。
---
## Proposition 3.4 Every $\mathtt{N}$-semiseparable matrix has a $\mathtt{N}$-SSS representation.

Furthermore, note that although D3.2 involves $O(\mathtt{N}^2\mathtt{T})$ parameters for the representation (in particular to store the $A$ matrices),
it can actually be compressed down to $O(\mathtt{NT})$ parameters, which is asymptotically tight渐进紧的“Exact computations with quasiseparable matrices” asymptotically代表理论极限
Therefore in the rest of this paper we will conflate 合并 the structured matrix class (D3.1) and a particular representation of it (D3.2); we will always use this representation instead of other candidates.
In turn we will use $\mathtt{N}$-SS to refer to an $\mathtt{N}$-semiseparable matrix in SSS form.

半可分矩阵是一种基本的矩阵结构，具有许多重要的性质。它们与大尺度递推关系密切相关，并且可以通过多种刻画方式进行定义（例如定义 3.1和 3.2 ），这些刻画方式揭示了不同的联系以及相应的有效算法。

备注：半可分性的概念非常广泛，文献中出现了许多相似但略有不同的定义；我们的定义可能与其他约定略有不同。首先，由于本文主要关注因果或自回归设置，我们将半可分性的定义限制在三角情形； 定义 3.1更正式地可以称为more formally might be called $(\mathtt{N},0)$-semiseparability by some authors. Some authors may also instead refer to it as a form of quasiseparability “On a new class of structured matrices” “Computing with Quasiseparable Matrices” “A bibliography on semiseparable matrices”

---

## 3.2.2 1-Semiseparable Matrices: the Scalar SSM Recurrence

We will single out the special case of $1$-SS matrices. Note that in this case, the $C_j$ and $B_i$ are scalars, and can be factored out of the SSS representation 4
(we also use lower-case to emphasize that the parameters are scalars in this case)
$$\begin{align*}
  \mathsf{SSS}(a, b, c) = \mathsf{diag}(c) \cdot M \cdot \mathsf{diag}(b) \qquad \text{where} \qquad M_{ji} = a_{j:i}^\times
  .
\end{align*}$$
Since diagonal matrices are easy to handle (e.g.\ multiplication by a diagonal matrix is the same as elementwise scalar multiplication),
we can ignore these terms.
Thus our basic representation of a 1-SS matrix is $M_{ji} = a_{j:i}$ or 公式6
$$\begin{equation}%
  \label{eq:1ss}
  M =
  \mathsf{1SS}(a_{0:T}) \coloneqq
  \begin{bmatrix}
    1 & \\
    a_1 & 1 & \\
    a_2a_1 & a_2 & 1 \\
    \vdots & \vdots & \ddots & \ddots \\
    a_{T-1}\dots a_1 & a_{T-1}\dots a_2 & \dots & a_{T-1} & 1 \\
  \end{bmatrix}
  .
\end{equation}
$$
The importance of 1-SS matrices lies in their equivalence to the minimal form of a scalar recurrence -- the case of a degenerate SSM with state dimension $\mathtt{N}=1$ and no $(B, C)$ projections.
Note that multiplication $y = Mx$ can be computed by the recurrence公式7
$$\begin{equation}
  \label{eq:1ss-recurrence}
  \begin{aligned}
    y_t &= a_{t:0}x_0 + \dots + a_{t:t}x_t \\
        &= a_t \left(a_{t-1:0}x_0 + \dots + a_{t-1:t-1}x_{t-1}\right) + a_{t:t}x_t \\
        &= a_t y_{t-1} + x_t
        .
  \end{aligned}
\end{equation}$$
We thus also refer to matrix multiplication by $1$-SS matrices as the $\textbf{scalar SSM recurrence}$ or the $\texttt{cumprodsum}$(cumulative product sum; a generalization of cumulative product and cumulative sum) operator.
As the fundamental form of recurrence, multiplication by 1-SS matrices is important as a building block for our main algorithms.

本文的核心主题之一是，序列模型上的许多算法可以简化为结构化矩阵乘法算法。1-SS 矩阵就是一个很好的例子：有很多快速算法可以计算原始标量递推或累积和算子，而所有这些算法都等价于 1-SS 矩阵的不同结构化分解。附录 B专门介绍这些 1-SS 矩阵乘法算法。

---
## Remark2: 1-SS与标量SSM递推的等价推导

这部分的核心逻辑是：将前面的状态维度 $N$ 简化为最极端的情况 **$N=1$（标量状态）**。当状态空间维度缩减为标量后，矩阵算法会退化为极其简洁的**累乘累加算子（`cumprodsum`）**，这正是 Mamba-2 (SSD) 算法在 GPU 底层高效实现的核心基石。

---
### 1. 标量退化：提离对角因子

在一般的 $N$-SSS 矩阵中，$B_i, C_j$ 是 $N$ 维向量，$A_i$ 是 $N \times N$ 矩阵。
当 **$N=1$** 时：
* $B_i, C_j, A_i$ 全部退化为**标量（Scalar）**，图中改用小写字母 $b_i, c_j, a_i$ 表示。
原本的非对角元素 $M_{ji} = c_j^{\top} a_{j:i}^{\times} b_i$ 可以直接拆分为三个标量乘积：
$$M_{ji} = c_j \cdot a_{j:i}^{\times} \cdot b_i$$
将所有的 $c_j$ 和 $b_i$ 提到矩阵外面，整个矩阵就可以表示为：

$$\text{SSS}(a, b, c) = \text{diag}(c) \cdot M \cdot \text{diag}(b)$$
* **物理直觉**：$\text{diag}(c)$ 和 $\text{diag}(b)$ 只是对输入和输出做逐元素的标量缩放（Elementwise Multiplication），在 GPU 上计算代价极小。因此，计算的核心彻底简化为了中间这个纯粹由 $a_i$ 构成的核心矩阵 $M$。

---
### 2. 1-SS 核心矩阵 $M$ 的结构（公式 6）

消去 $b, c$ 后，核心矩阵 $M = \text{1SS}(a_{0:T})$ 变成了一个纯粹由衰减系数 $a_i$ 累乘构成的下三角矩阵：
$$M = \begin{bmatrix} 1 & 0 & 0 & \dots & 0 \\  a_1 & 1 & 0 & \dots & 0 \\  a_2 a_1 & a_2 & 1 & \dots & 0 \\  \vdots & \vdots & \ddots & \ddots & \vdots \\  a_{T-1} \dots a_1 & a_{T-1} \dots a_2 & \dots & a_{T-1} & 1 \end{bmatrix}$$
矩阵元素 $M_{ji} = a_{j:i}^{\times} = a_j a_{j-1} \dots a_{i+1}$（主对角线上全为 1）。

---
### 3. 与一阶标量递推的等价性（公式 7）
计算 $y = M x$ 的矩阵乘法，写成第 $t$ 个元素的显式展开：
$$y_t = a_{t:0}^{\times} x_0 + a_{t:1}^{\times} x_1 + \dots + a_{t:t-1}^{\times} x_{t-1} + x_t$$
提取公因数 $a_t$：
$$y_t = a_t \underbrace{\left( a_{t-1:0}^{\times} x_0 + \dots + a_{t-1:t-1}^{\times} x_{t-1} \right)}_{y_{t-1}} + x_t$$
直接推导出极为简单的一阶一维递推公式：
$$y_t = a_t y_{t-1} + x_t$$
---

### 4. 核心术语：`cumprodsum` 算子及其重要性

文中将 $y = M x$ 的计算称为 **`cumprodsum` (Cumulative Product Sum，累乘前缀和)** 算子：

* **`cumsum`（累加和）**：$y_t = y_{t-1} + x_t$（对应 $a_t = 1$ 的情况）。
* **`cumprod`（累乘积）**：$y_t = a_t \cdot y_{t-1}$。
* **`cumprodsum`**：两者的结合（先衰减乘 $a_t$，再叠加新输入 $x_t$）。
#### **为什么这个结论在论文中至关重要？**

1. **分治与算法构建基块**：任意高维的 SSM（$N > 1$）或者复杂的非对角块分解，在底层都可以拆解为多组独立的一维 `cumprodsum` 计算。

2. **GPU 硬件极速并行（Associative Scan）**：
一阶递推 $y_t = a_t y_{t-1} + x_t$ 满足**结合律（Associativity）**。这意味着它不需要像传统 RNN 那样一步一步串行算，而是可以通过 **Parallel Scan（并行前缀和算法）** 在 GPU 上以 $\mathcal{O}(\log T)$ 的并行时间复杂度极速完成计算。

---
## 3.3 State Space Models are Semiseparable Matrices

Recall that our definition of an SSM is defined as a parameterized map
defined through D2.1
The connection between SSMs and semiseparable matrices follows from simply writing this transformation as a matrix multiplication mapping the vectors $x \mapsto y \in R^\mathtt{T}$.

Equation 3 directly establishes the link between state space models and the sequentially semiseparable representation, which in turn are equivalent to semiseparable matrices in general (Lemma3.3 Proposition3.4).

**Theorem 3.5** The state space model transformation $y = \mathsf{SSM}(A, B, C)(x)$ with state size $\mathtt{N}$ is identical to matrix multiplication by an $\mathtt{N}$-SS matrix in sequentially semiseparable representation $y = \mathsf{SSS}(A, B, C) \cdot x$.

In other words the sequence transformation operator $\mathsf{SSM}$ (D2.2) coincides with the matrix construction operator $\mathsf{SSS}$ (D3.2),
and we use them interchangeably可互换的 (or sometimes $\mathsf{SS}$ as shorthand).
Furthermore---by a twist of fate---structured state space models and sequentially semiseparable matrices have the same acronyms, underscoring their equivalence!

Conveniently we can use any of these acronyms SSM (state space model or semiseparable matrix), SSS (structured state space or sequentially semiseparable), or SS (state space or semiseparable) interchangeably to unambiguously refer to either concept.

However, we will generally use the convention that SSM refers to state space model, SS refers to semiseparable, and SSS refers to sequentially semiseparable.

---
## 3.4 Computing State Space Models through Structured Matrix Algorithms

The reason Theorem 3.5 is important is that it will allow us to reduce the problem of efficient computation of SSMs (and other sequence models) into efficient algorithms for structured matrix multiplication.

As previously defined, semiseparable matrices (i.e.\ rank-structured matrices) are a classical type of structured matrix:
- They have compressed representations such as the SSS form which has only $O(\mathtt{T})$ instead of $O(\mathtt{T}^2)$ parameters.
- They have fast algorithms operating directly on the compressed representation.
Furthermore, the parameterization and matrix multiplication cost can be tight in the semiseparable order.

---
## Remark：大O

要理解这里的**计算复杂度**和**非零参数个数**，我们需要将**普通稠密下三角矩阵**与 **SSS 结构化表示**放在一起做精细的数学对比。
设序列长度为 $T$，状态维度为 $N$（在实际模型中，通常 $T \gg N$，比如 $T = 32000, N = 64$）。
### 一、 普通下三角矩阵（Unstructured Lower Triangular Matrix）

一个普通的 $T \times T$ 下三角矩阵 $M$ 形式如下：

$$M = \begin{bmatrix} m_{0,0} & 0 & 0 & \dots & 0 \\ m_{1,0} & m_{1,1} & 0 & \dots & 0 \\ m_{2,0} & m_{2,1} & m_{2,2} & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ m_{T-1,0} & m_{T-1,1} & m_{T-1,2} & \dots & m_{T-1,T-1} \end{bmatrix}$$
#### 1. 零参数个数计算

- 第 $0$ 行有 1 个非零元素；
- 第 $T-1$ 行有 $T$ 个非零元素。
将所有行的非零元素个数相加（等差数列求和）：
$$\text{参数个数} = \sum_{t=0}^{T-1} (t+1) = \frac{T(T+1)}{2} = \frac{1}{2}T^2 + \frac{1}{2}T$$

在大 $\mathcal{O}$ 记号中，忽略低阶项 $\frac{1}{2}T$ 和常数系数 $\frac{1}{2}$，其空间复杂度为 **$\mathcal{O}(T^2)$*。
#### 2. 计算/时间复杂度（矩阵向量乘法 $y = M x$）

计算输出向量 $y$ 的第 $t$ 个元素 $y_t = \sum_{s=0}^{t} m_{t,s} x_s$：
- 计算 $y_t$ 需要做 $t+1$ 次乘法和 $t$ 次加法（约 $2(t+1)$ 次浮点运算 FLOPs）。
- 整个向量 $y$ 的总计算量为：
    $$\text{FLOPs} = \sum_{t=0}^{T-1} 2(t+1) = T(T+1) = T^2 + T$$
在大 $\mathcal{O}$ 记号中，计算复杂度为 **$\mathcal{O}(T^2)$**。
### 二、 SSS 结构化表示（Sequentially Semiseparable Representation）

在 SSS 表示中，矩阵 $M$ 的每一个元素并不是独立的自由参数，而是由参数序列隐式生成的：
$$M_{ji} = C_j^\top A_j A_{j-1} \dots A_{i+1} B_i \quad (j \ge i)$$
整个矩阵由三个参数序列完整确定：
- $A = \{A_0, A_1, \dots, A_{T-1}\}$，其中每个 $A_t \in \mathbb{R}^{N \times N}$；
- $B = \{B_0, B_1, \dots, B_{T-1}\}$，其中每个 $B_t \in \mathbb{R}^{N \times 1}$；
- $C = \{C_0, C_1, \dots, C_{T-1}\}$，其中每个 $C_t \in \mathbb{R}^{N \times 1}$。
#### 1. 非零参数个数计算

我们**不需要**把 $T \times T$ 的大矩阵铺开存储，只需要在内存中保存这三个序列：
- $A$ 序列：$T$ 个 $N \times N$ 矩阵 $\implies T \cdot N^2$ 个参数；
- $B$ 序列：$T$ 个 $N \times 1$ 向量 $\implies T \cdot N$ 个参数；
- $C$ 序列：$T$ 个 $N \times 1$ 向量 $\implies T \cdot N$ 个参数    

$$\text{总参数个数} = T \cdot (N^2 + 2N)$$

在状态维度 $N$ 固定为常数（如 $N=64$）的前提下，$N^2 + 2N$ 是一个常数。因此参数量与序列长度 $T$ 呈**纯线性关系**，大 $\mathcal{O}$ 记号下为 **$\mathcal{O}(T)$**。
#### 2. 计算/时间复杂度（利用递推关系算 $y = M x$）

计算 $y = M x$ 并不需要展开成大矩阵乘法，而是利用其等价的线性状态空间递推关系：
对每个时间步 $t = 0, 1, \dots, T-1$：
1. **隐状态更新**：$h_t = A_t h_{t-1} + B_t x_t$
    - $A_t h_{t-1}$（矩阵-向量乘法）：$N^2$ 次乘加；
    - $B_t x_t$（标量-向量乘法）：$N$ 次乘法；
    - 向量相加：$N$ 次加法。
2. **输出计算**：$y_t = C_t^\top h_t$
    - 点积运算：$N$ 次乘法和加法。
每个时间步的总计算量仅为 $\mathcal{O}(N^2 + N)$ 次操作。
对于整个长度为 $T$ 的序列：
$$\text{总 FLOPs} = T \cdot \mathcal{O}(N^2 + N)$$
当 $N$ 视为常数时，整个过程的计算复杂度与 $T$ 呈**线性关系**，即 **$\mathcal{O}(T)$**。

---

## Proposition3.6 “Exact computations with quasiseparable matrices”
  
An $\mathtt{N}$-SS matrix of size $\mathtt{T}$ can be represented in $O(\mathtt{NT})$ parameters and has matrix-vector multiplication in time and space $O(\mathtt{NT})$.

For example, 1-SS matrices illustrate the essence of this connection.
The matrix $M = \mathsf{1SS}(a)$ is defined by exactly $\mathtt{T}-1$ parameters $a_{0:\mathtt{T}-1} = a_1, \dots, a_{\mathtt{T}-1}$,
and can be computed in $O(\mathtt{T})$ time by following the scalar recurrence 7.

---
## 3.4.1 The Linear (Recurrent) Mode

 Proposition3.6 can be easily seen in the case of diagonal structured SSMs (S4D),
simply by leveraging the state space model formulation 2 and unrolling the recurrence.
We provide the formal tensor-contraction algorithm in 8, where the dimension $\mathtt{S}$ is equal to $\mathtt{T}$ 公式8
$$
  \begin{align}
    Z &= \mathsf{contract}(\mathtt{SP},\mathtt{SN} \to \mathtt{SPN})(X, B) & \mathtt{(S,P,N)} \\
    H &= \mathsf{contract}(\mathtt{TSN},\mathtt{SPN} \to \mathtt{TPN})(L, Z) & \mathtt{(T,P,N)} \\
    Y &= \mathsf{contract}(\mathtt{TN},\mathtt{TPN} \to \mathtt{TP})(C, H) & \mathtt{(T,P)}
  \end{align}
$$
Here, $L \in R^{(\mathtt{T},\mathtt{T})}$ is defined as $\mathsf{1SS}(A)$, or in other words $L_{0:\mathtt{T},0:\mathtt{T}} = \mathsf{1SS}(A_{0:\mathtt{T}})$ for $i \in [\mathtt{N}]$. This algorithm involves three steps corresponding to 2:
- expanding the input $X$ by the input matrix $B$ 8a,
- unrolling independent scalar SSM recurrences 8b
- contracting the hidden state $H$ by the output matrix $C$ 8c
Note that we have used the equivalence between scalar SSMs and 1-SS matrices in step 8b
  
We note that 8 is a special case of the Mamba (S6) model.
however, a naive implementation is slow because of the expanded tensors $Z$ and $H$ of size $\mathtt{(T,P,N)}$; “Mamba: Linear-Time Sequence Modeling with Selective State Spaces” introduced a hardware-aware implementation to avoid materializing these tensors.
令人惊讶的是，定理 3.5和命题 Proposition3.6立即表明所有 SSM 都具有与算法 ( 8 ) 相同的渐近效率。

**Theorem3.7** Any state space model (D2.2) of state size $\mathtt{N}$ on sequence length $\mathtt{T}$ can be computed in time $O(\mathtt{TN})$ (not accounting for potential preprocessing).

We note that this result is new to the structured SSM literature.
In particular, given dense unstructured $A_t$ matrices, the total representation alone seems to be of size $O(\mathtt{TN}^2)$.
Thus  Theorem3.7 states the non-trivial result that with a pre-processing step, even an unstructured SSM can be computed optimally efficiently,
with upper bound matching the lower bound $O(\mathtt{TN})$ given by the size of $B$ and $C$.

Theorem3.7 is perhaps not too surprising in light of the fact that almost all dense matrices over $R^{\mathtt{(N,N)}}$ are diagonalizable over $\mathbb{C}$,
leading to the result that almost all dense real SSMs are equivalent to a diagonal complex SSM.

This fact underlies the reason why diagonal SSMs are the most popular form of structured SSM.

However, Theorem3.7 implies the much stronger result for all real SSMs (not just the diagonalizable ones), as well as dense SSMs over other fields (including $\mathbb{C}$ itself).

---
## Remark3:复数集
这里的 $\mathbb{C}$ 指的是**复数集（Complex Numbers）**。
在数学中，$\mathbb{R}$ 代表实数集（Real Numbers），$\mathbb{C}$ 代表复数集（Complex Numbers）。

1. **矩阵对角化（Diagonalizable）**：
对于任意一个在实数域上的 $N \times N$ 密集矩阵 $A \in \mathbb{R}^{N \times N}$，如果不加特殊约束（比如对称性），它的特征值（Eigenvalues）通常会有虚部。因此，虽然矩阵 $A$ 本身的所有元素都是实数，但它**无法在实数域 $\mathbb{R}$ 内对角化**，必须扩展到**复数域 $\mathbb{C}$** 才能被对角化为 $P^{-1} A P = \Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$（其中特征值 $\lambda_i \in \mathbb{C}$ 是复数）。
2. **“Almost all”（几乎所有）**：
在测度论和线性代数中，“几乎所有”意味着不可对角化的矩阵（比如有重特征值且几何重数不足的矩阵）在所有 $N \times N$ 矩阵构成的空间中测度为零（即概率为 0）。所以绝大多数随机初始化的实数 SSM 矩阵 $A$ 都是可以复对角化的。
3. **对 SSM 的物理/算法影响**：
* 原本的实数 SSM 状态转移矩阵 $A \in \mathbb{R}^{N \times N}$ 是一个满秩的稠密实数矩阵，状态之间互相耦合；
* 通过在复数域 $\mathbb{C}$ 上做对角化，可以将这个复杂的实数 SSM **等价变换为一个对角复数 SSM（Diagonal Complex SSM，如 S4D）**；
* 对角化后，不同维度之间的状态完全解耦，每个维度只对应一个复数标量演化，这也是上一节能够把复杂的 $N$ 维状态分解为独立的标量递推/张量缩约（Tensor Contraction）的数学前提。
---
## 3.4.2 The Quadratic (Naive) Mode

We note that there is another way to compute an SSM exposed by our new matrix point of view.
A naive computation of the matrix SSM representation 3 involves simply materializing the sequence transformation matrix $M=\mathsf{SSS}(A, B, C)$.

This is a $\mathtt{(T,T)}$ matrix, and therefore this naive algorithm will scale quadratically in sequence length.

**However, when the sequence length $\mathtt{T}$ is short, this can actually be more efficient than the linear algorithm due to constant factors and the hardware-friendliness of the computation pattern (e.g. leveraging matrix-matrix multiplications).**

**In fact, for a particular case of structured SSMs, this looks very similar to a quadratic attention computation.**

朴素算法之所以会在序列长度 $T$ 上呈**二次方级增长（Quadratically Scaling, 即 $\mathcal{O}(T^2)$）**，核心原因在于它选择在显存中直接显式构造（Materialize）整个大小为 $(T, T)$ 的序列变换矩阵 $M = \text{SSS}(A, B, C)$。
### 1. 显存空间开销：$\mathcal{O}(T^2)$

- 矩阵 $M$ 的维度是 $(T, T)$。即使它是因果下三角矩阵（主对角线上方全为 0），下三角区域内仍有 $\frac{T(T+1)}{2} = \frac{1}{2}T^2 + \frac{1}{2}T$ 个不同的元素需要计算并存储。
- **结果**：空间复杂度为 $\mathcal{O}(T^2)$。当序列长度 $T$ 增加 2 倍时，该矩阵占用的显存空间会膨胀 **4 倍**。
### 2. 计算时间开销（FLOPs）：$\mathcal{O}(T^2)$

朴素算法需要完成两步计算：
1. **构造矩阵 $M$**： 计算 $M$ 中第 $j$ 行第 $i$ 列的每一个元素 $M_{ji} = C_j^\top A_j \dots A_{i+1} B_i$（其中 $j \ge i$）。 由于下三角区域共有约 $\frac{1}{2}T^2$ 个位置，哪怕利用前缀累乘算子填充每一个 $M_{ji}$，填满整个矩阵最少也需要 $\mathcal{O}(T^2)$ 次标量/矩阵乘法操作。
2. **矩阵-向量乘法 $y = Mx$**： 将这个大小为 $(T, T)$ 的密集/下三角矩阵 $M$ 乘以长为 $T$ 的输入向量 $x$：  
    $$y_t = \sum_{i=0}^{t} M_{ti} x_i$$
    计算完整序列 $y$ 的点积累加操作次数同样为 $\frac{T(T+1)}{2} \sim \mathcal{O}(T^2)$。
    
- **结果**：计算复杂度（FLOPs）同样为 $\mathcal{O}(T^2)$。

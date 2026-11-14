# 7.1 Block Design

**平行参数投影**
In Mamba-2, the SSD layer is viewed as a map from $A, X, B, C \mapsto Y$.
It therefore makes sense to produce $A, X, B, C$ in parallel with a single projection at the beginning of the block.
Note the analogy 类比 to standard attention architectures,
where $X, B, C$ correspond to the $Q, K, V$ projections that are created in parallel.

Note that adopting parallel projections for the $A, B, C, X$ inputs to the SSM slightly reduces parameters and more importantly is more amenable顺从的，乐于接受的 to tensor parallelism for larger models, by using standard Megatron sharding patterns.碎片模式

**额外归一化**
在初步实验中，**我们发现大型模型容易出现不稳定性**。我们通过在最终输出投影之前的模块中添加一个额外的归一化层（例如 LayerNorm、GroupNorm 或 RMSNorm）来缓解这个问题。这种归一化的使用与 NormFormer 架构最为直接相关 “NormFormer: Improved Transformer Pretraining with Extra Normalization”，并在 MLP 和 MHA 模块的末尾添加了归一化层。

我们还注意到，这种变化与其他近期基于线性注意力机制的 Mamba-2 模型类似。原始的线性注意力公式通过一个分母项进行归一化，该分母项模拟了标准注意力机制中 softmax 函数的归一化。TransNormerLLM “TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer”和 RetNet “Retentive network: A successor to transformer for large language models”
我们发现这种归一化方法不稳定，因此在线性注意力层之后添加了一个额外的 LayerNorm 或 GroupNorm 层。我们添加的额外归一化层与这些方法略有不同，它位于乘法门分支之后，而不是之前。

---

# 7.2 用于序列转换的多头模式

Recall that SSMs are defined as a sequence transformation
- $A, B, C$ parameters have a state dimension $\mathtt{N}$.
- They define a sequence transformation $R^\mathtt{T} \to R^\mathtt{T}$, which for example can be represented as a matrix $M \in R^{\mathtt{(T, T)}}$.
- This transformation operates over an input sequence $X \in R^{\mathtt{(T, P)}}$, independently over the $\mathtt{P}$ axis.
One can view this as defining one head of the sequence transformation.

![[Mamba2_Architecture.png]]

---

## 1. 序列变换 $R^T \to R^T$ 与变换矩阵 $M \in \mathbb{R}^{(T, T)}$

- **含义**：它定义了一个**作用在时间序列（长度为 $T$）上的线性映射**。

- **矩阵表示**：这个变换可以被写成一个 $T \times T$ 的下三角/结构化矩阵 $M$（在 SSD 中，$M$ 属于 semiseparable 矩阵类，比如 causal mask 矩阵或由 SSM 参数生成的衰减矩阵 $M_{i,j} = C_i A^{i-j} B_j$）。

- **作用机制**：$M$ 的第 $(i, j)$ 个元素决定了第 $j$ 个时刻的输入对第 $i$ 个时刻的输出贡献了多少。
## 3. “One Head of the Sequence Transformation”（单个 Head 的定义）

-  在多头架构（Multi-Head Attention 或 Multi-Head SSM）中，**单头（Head）专精于时间/空间维度的信息混合（Time Mixing）**，而通道维度的混合（Channel Mixing）则留给头外侧的线性投影层（如 $W_{out}$）去完成。
-  无论是 Self-Attention 的 Causal Mask $M = \text{Softmax}(Q K^\top)$，还是 Mamba/SSM 的选择性衰减矩阵 $M = \text{SSM}(A, B, C)$，在一个 Head 内部，核心计算都可以写成 $Y = MX$ 的形式，即对每一个通道 $p$ 独立应用 $T \times T$ 的时间变换矩阵 $M$。
---

## 为什么混合“时间”必须左乘？

根据矩阵乘法的规则 $(m \times n) \times (n \times k) = (m \times k)$：

#### **左乘：混合行（时间/位置维度 $T$）**

如果我们有一个大小为 $T \times T$ 的变换矩阵 $M$，将它**左乘** $X$：

$$Y = M_{T \times T} \, X_{T \times P}$$

- **乘法过程**：输出 $Y$ 的第 $i$ 行（即第 $i$ 个时刻的输出 vector）是 $M$ 的第 $i$ 行与 $X$ 的所有行做线性组合：$$Y[i, :] = \sum_{j=1}^T M[i, j] \cdot X[j, :]$$
- **物理意义**：它在**沿着 $T$ 轴（时间轴）打乱和融合信息**。例如 $M[i, j]$ 代表第 $j$ 个时刻的输入对第 $i$ 个时刻的输出有多大贡献（在 Attention 中就是注意力权重，在 SSM 中就是选择性衰减权重）。因为这一步完全是在跨时间步混合，所以被称为 **Sequence Transformation（序列变换）**。
#### **右乘：混合列（特征/通道维度 $P$）**

作为对比，如果我们有一个大小为 $P \times P$ 的矩阵 $W$，将它**右乘** $X$：

$$Z = X_{T \times P} \, W_{P \times P}$$

- **乘法过程**：输出 $Z$ 的每一个时刻 $i$，其特征向量 $Z[i, :]$ 都是 $X[i, :]$ 乘以 $W$。
- **物理意义**：它在**沿着 $P$ 轴（特征通道）做映射和混合**，即 Channel Mixing（比如 Linear 投影层 $X W_W$）。在此过程中，不同时刻之间（行与行之间）互不干扰。
### 3. “在 $P$ 轴上独立”具体表现为啥？

当对 $X$ 执行左乘 $Y = MX$ 时，如果我们把矩阵 $X$ 按列拆开，看第 $p$ 个特征通道：
$$X = \begin{bmatrix} \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(P)} \end{bmatrix}$$

其中每个 $\mathbf{x}^{(p)}$ 都是一个长为 $T$ 的列向量。
左乘 $M$ 相当于对每一列单独做变换：
$$Y = \begin{bmatrix} M\mathbf{x}^{(1)} & M\mathbf{x}^{(2)} & \cdots & M\mathbf{x}^{(P)} \end{bmatrix}$$
这意味着：

1. 第 $1$ 个通道的序列 $\mathbf{x}^{(1)}$ 通过矩阵 $M$ 演化成了 $M\mathbf{x}^{(1)}$。
2. 第 $p$ 个通道在做时间混合时，**完全用的是同一套矩阵 $M$，且不需要借用第 $q$ 个通道的数据**。
这就是所谓的“序列变换由左乘 $M$ 体现，且在 $P$ 轴（特征轴）上独立”。

---

**Definition 7.1**  A multihead sequence transformation consists of $\mathtt{H}$ independent heads, for a total model dimension of $\mathtt{D}=\mathtt{d\_model}$. **The parameters may be tied across heads**, leading to a $\textbf{head pattern}$.

The state size $\mathtt{N}$ and head dimension $\mathtt{P}$ are analogous to the $QK$ head dimension and $V$ head dimension of attention, respectively.
Just as in modern Transformer, in Mamba-2 we generally choose these to be constants around $64$ or $128$; when the model dimension $\mathtt{D}$ increases, we increase the number of heads while keeping the head dimensions $\mathtt{N}$ and $\mathtt{P}$ fixed.

In order to describe how to do this, we can transfer and generalize ideas from multihead attention to define similar patterns for SSMs, or any general sequence transformation.为了描述如何做到这一点，我们可以将多头注意力机制中的思想迁移和概括，从而为 SSM 或任何一般的序列变换定义类似的模式。
$$
\begin{array}{cccc}
% 第一列: Multi-head SSM
\begin{aligned}
  & \textbf{Multi-head SSM} \\
  & (\text{Multi-head Attn.}) \\
  X & \quad \mathtt{(T, H, P)} \\
  A & \quad \mathtt{(T, H)} \\
  B & \quad \mathtt{(T, H, N)} \\
  C & \quad \mathtt{(T, H, N)}
\end{aligned}
&
% 第二列: Multi-contract SSM
\begin{aligned}
  & \textbf{Multi-contract SSM} \\
  & (\text{Multi-query Attn.}) \\
  X & \quad \mathtt{(T, 1, P)} \\
  A & \quad \mathtt{(T, H)} \\
  B & \quad \mathtt{(T, 1, N)} \\
  C & \quad \mathtt{(T, H, N)}
\end{aligned}
&
% 第三列: Multi-expand SSM
\begin{aligned}
  & \textbf{Multi-expand SSM} \\
  & (\text{Multi-key Attn.}) \\
  X & \quad \mathtt{(T, 1, P)} \\
  A & \quad \mathtt{(T, H)} \\
  B & \quad \mathtt{(T, H, N)} \\
  C & \quad \mathtt{(T, 1, N)}
\end{aligned}
&
% 第四列: Multi-input SSM
\begin{aligned}
  & \textbf{Multi-input SSM} \\
  & (\text{Multi-value Attn.}) \\
  X & \quad \mathtt{(T, H, P)} \\
  A & \quad \mathtt{(T, H)} \\
  B & \quad \mathtt{(T, 1, N)} \\
  C & \quad \mathtt{(T, 1, N)}
\end{aligned}
\end{array}
$$

**Multihead SSM (MHS) / Multihead Attention (MHA) Pattern.**
The classic MHA pattern assumes that the head dimension $\mathtt{P}$ divides the model dimension $\mathtt{D}$.
The number of heads is defined as $\mathtt{H} = \mathtt{D} / \mathtt{P}$.
Then, $\mathtt{H}$ copies of the core sequence transformation are created by creating $\mathtt{H}$ independent copies of each parameter.
Note that while the MHA pattern was first described for the attention sequence transformation, where the SSD algorithm is broadcasted over the $\mathtt{H} = \mathtt{n\_heads}$ dimension.

**Multi-contract SSM (MCS) / Multi-query Attention (MQA) Pattern.**
Multi-query attention is a clever optimization for attention that can dramatically improve the speed of autoregressive inference,
which relies on caching the $K$ and $V$ tensors.
This technique simply avoids giving $K$ and $V$ the extra head dimension, or in other words broadcasts a single head of $(K, V)$ across all the heads of $Q$. Using the state space duality, we can define an equivalent SSM version of MQA. Here, $X$ and $B$ (the SSM analogs of attention's $V$ and $K$) are shared across the $\mathtt{H}$ heads.
We also call this the multi-contract SSM (MCS) head pattern, because the $C$ parameter which controls the SSM state contraction 收缩 has independent copies per head.

We can similarly define a multi-key attention (MKA) or multi-expand SSM (MES) head pattern,
where $B$ (which controls the SSM expansion) is independent per head while $C$ and $X$ are shared across heads.

**多输入SSM（MIS）/多值注意力（MVA）模式**。虽然 MQA 由于其键值缓存机制而适用于注意力机制，但它并非 SSM 的自然选择。在 Mamba 中，instead, $X$ is viewed as the main input to the SSM,
and therefore $B$ and $C$ are parameters that are shared across the input channels.

We define a new multi-value attention (MVA) of multi-input SSM (MIS) pattern, which can again be applied to any sequence transformation such as SSD.

Armed with this vocabulary, we can characterize the original Mamba architecture more precisely.

**Proposition 7.2**  The selective SSM (S6) layer of the Mamba architecture can be viewed as having
- Head dimension $P=1$: every channel has independent SSM dynamics $A$.
- Multi-input SSM  (MIS) or multi-value attention (MVA) head structure: the $B, C$ matrices (corresponding to $K, Q$ in the attention duality) are shared across all channels of the input $X$ (corresponding to $V$ in attention).
**We empirically find that the MVA pattern as originally used in Mamba performs best.**

**Grouped Head Patterns.**
The ideas of multi-query attention can be extended to **grouped-query attention**: instead of $1$ K and V head, one can create $\mathtt{G}$ independent K and V heads, where $1 < \mathtt{G}$ and $\mathtt{G}$ divides $\mathtt{H}$.
This is motivated both by bridging the performance difference between multi-query and multi-head attention,
and enabling more efficient tensor parallelism by setting $\mathtt{G}$ to be a multiple of the number of shards .碎片
Similarly, the multi-input SSM head pattern used in Mamba-2 can be easily extended to $\textbf{grouped-input SSM (GIS)}$, or synonymously同义地 $\textbf{grouped-value attention (GVA)}$.

---

# 7.3 Other SSD Extensions from Linear Attention

**核注意​​力机制对 Softmax 注意力机制的近似。**
Many variants of linear attention or kernel attention are motivated by viewing the attention scores $\mathsf{softmax}(QK^{\top})$ as composed of
- An exponential kernel $Z = \exp(QK^{\top})$, which can be approximated by $Z = \psi(Q)\psi(K)^{\top}$ for some kernel feature map.
- Normalizing the kernel so that rows sum to $1$ via $M = G / G \texttt{1} \texttt{1}^{\top}$, where the division happens elementwise and $\texttt{1}$ is the all 1's vector.逐元素相除

---

公式 $M = G \oslash (G \mathbf{1} \mathbf{1}^\top)$（其中 $\oslash$ 表示逐元素相除）是用纯矩阵代数语言表达“将矩阵 $G$ 的每一行进行归一化，使其行和为 1”的紧凑写法。
## 1. 逐步拆解矩阵运算

假设未归一化的核矩阵（Kernel/Affinity Matrix）为 $G \in \mathbb{R}^{T \times T}$，全 1 列向量为 $\mathbf{1} = [1, 1, \dots, 1]^\top \in \mathbb{R}^{T \times 1}$。
 **第一步：计算每行的求和向量 $G \mathbf{1}$**

$G \mathbf{1}$ 是一个大小为 $T \times 1$ 的列向量，第 $i$ 个元素为矩阵 $G$ 第 $i$ 行所有元素的和：
$$(G \mathbf{1})_i = \sum_{j=1}^T G_{i, j}$$
**第二步：构造广播矩阵 $(G \mathbf{1}) \mathbf{1}^\top$**

$G \mathbf{1}$（$T \times 1$）与 $\mathbf{1}^\top$（$1 \times T$）做外积，生成一个大小为 $T \times T$ 的矩阵：$$(G \mathbf{1} \mathbf{1}^\top)_{i, j} = \sum_{k=1}^T G_{i, k}$$这个矩阵的特点是：**第 $i$ 行的所有元素完全相同，都等于 $G$ 第 $i$ 行元素的总和**。

**第三步：逐元素相除 $M = G \oslash (G \mathbf{1} \mathbf{1}^\top)$**

对两矩阵对应位置相除：
$$M_{i, j} = \frac{G_{i, j}}{\sum_{k=1}^T G_{i, k}}$$
#### **验证：行和是否为 1**

计算归一化后矩阵 $M$ 第 $i$ 行的元素和：

$$\sum_{j=1}^T M_{i, j} = \sum_{j=1}^T \frac{G_{i, j}}{\sum_{k=1}^T G_{i, k}} = \frac{\sum_{j=1}^T G_{i, j}}{\sum_{k=1}^T G_{i, k}} = 1$$

### 2. 代码对比与物理意义

在代码实现（如 PyTorch）中，这一操作等价于设置 `keepdim=True` 的按行求和与广播除法：

Python

```
# 逻辑完全等价
row_sums = G.sum(dim=-1, keepdim=True)  # 对应 G 1
M = G / row_sums                         # 对应逐元素相除 (广播机制)
```

在纯线性代数论文中，由于没有代码中的“自动广播（Broadcasting）”语法，必须借助全 1 向量的外积 $(G \mathbf{1})\mathbf{1}^\top$ 将行和向量拉伸为同尺寸矩阵，从而在严谨的数学形式上完成逐元素相除。
### 3. 在 Attention 与 Linear Attention / SSD 中的作用

- **概率加权（Row-stochastic Matrix）**：未归一化的内核 $G_{i,j}$ 表示时刻 $j$ 的 Token 对时刻 $i$ 的关联强度。归一化后的 $M$ 变成行随机矩阵（每行构成一个概率分布），确保输出 $Y = MX$ 是对输入特征 $X$ 的**凸组合（Convex Combination / 纯加权平均）**，防止数值因为序列长度 $T$ 增长而发散。
- **对应 Standard Attention 的 Softmax**：在标准 Transformer 中，Softmax 自带分母归一化（$\sum_j \exp(Q_i K_j^\top)$）；而在 Linear Attention 或 Mamba-2 (SSD) 的核矩阵视能中，通常先显式算出关联内核 $G$，再利用上述公式完成类似 Softmax 分母的全局归一化。

---

**Exponential Kernel Feature Maps.**
In Mamba-2, we incorporate  包含 a flexible kernel feature map, and apply it to the $B$ and $C$ branches (corresponding to the $K$ and $V$ branches in attention).
The feature map can also be optionally applied to the $X$ ($V$) branch, for simplicity and symmetry.
By default, we simply choose $\psi$ to be an elementwise Swish / SiLU function.

**Incorporating a Normalization (Denominator) Term.**

To find the denominator分母 term, we simply have to compute $M \texttt{1}$.
But recall that the final output of the model is just $Y = MX$ .
So the normalization terms can be found simply by augmenting增强 $X$ with an extra column $\texttt{1}$, resulting in a tensor of shape $\mathtt{(T, P+1)}$.

Note that in this case, the kernel feature map $\psi$ must be positive so that the sum is positive.

---

## 第二段：Incorporating a Normalization (Denominator) Term（如何引入归一化/分母项）

这一段展示了一个非常精彩且高效的**代数 Trick**，用来计算上一问中提及的行归一化（即分母 $M\mathbf{1}$）：

1. **目标**： 要实现核矩阵的行归一化，需要计算每个时刻的行和向量 $M\mathbf{1}$。

2. **巧妙解法（拼接全 1 列）**：
  
    - 模型的最终输出形式是 $Y = M X$，其中 $X \in \mathbb{R}^{T \times P}$。

    - 如果我们**在 $X$ 的特征维度（列）最后直接拼接一列全 1 的向量 $\mathbf{1}$**，得到扩展后的张量 $\tilde{X} = [X \mid \mathbf{1}] \in \mathbb{R}^{T \times (P+1)}$；

    - 那么直接对扩展后的 $\tilde{X}$ 执行标准的 SSD / Mamba 序列变换：
$$\tilde{Y} = M \tilde{X} = M [X \mid \mathbf{1}] = [M X \mid M \mathbf{1}]$$
    - 结果中：
  
        - 前 $P$ 列就是**未归一化的输出 $Y = MX$**；

        - 最后一列（第 $P+1$ 列）正好就是**归一化分母 $M\mathbf{1}$**！

3. **后处理**： 拿到 $\tilde{Y}$ 后，只需要把前 $P$ 列除以最后一列，就一次性完成了全局归一化，**完全不需要显式构造 $T \times T$ 的大矩阵 $M$ 或额外增加计算图分支**。

4. **正值限制**： 这种归一化要求核特征映射 $\psi$ 的输出必须是非负的（positive），从而确保累加出来的分母 $M\mathbf{1}$ 大于 0（防止除以零或产生负权重）。
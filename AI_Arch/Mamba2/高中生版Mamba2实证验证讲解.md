我们对 Mamba-2 进行了实证评估，评估对象包括对循环模型而言具有挑战性的合成召回任务以及标准语言建模的预训练和下游评估。我们验证了我们的 SSD 算法比 Mamba-1 效率更高，并且在中等序列长度下与优化注意力机制相当。最后，我们分析了 Mamba-2 架构中的各种设计选择。

# 9.1 Synthetics: Associative Recall

合成联想回忆任务一直以来都被广泛用于测试语言模型在上下文中查找信息的能力。总的来说，这类任务包括向自回归模型输入键值对关联信息，然后提示模型在看到先前见过的键时给出正确的补全。多查询联想回忆（MQAR）任务是该任务的一种特殊形式，它要求模型记住多个关联。最初的 Mamba 论文报告了相关合成任务的结果，特别是选择性复制和感应头。这可以看作是更容易的联想回忆任务。MQAR 任务也与“电话簿查找”任务密切相关，后者已被证明对 SSM 等循环模型来说具有挑战性，因为它们的状态容量有限 。

我们使用更难的任务、更长的序列和更小的模型。我们的基线模型包括标准的多头softmax注意力机制，以及结合了卷积、局部注意力机制和线性注意力机制变体的Based架构。

结果如图 8所示。Mamba-1 在这项任务中表现不佳，而 Mamba-2 在所有设置下都表现良好。令人惊讶的是，即使控制状态大小，它的表现也明显优于 Mamba-1（N=16（我们尚不确定架构的哪个方面是主要因素，这仍需在未来的工作中探讨。））此外，这项任务验证了状态规模的重要性：从N=16到N=256持续提高 MQAR 的性能，因为更大的状态允许记忆更多信息（键值对）。

![[Mamba-2_fig8.png]]

---

#  9.2 语言建模

遵循语言学习模型（LLM）的标准协议，我们使用标准的自回归语言模型训练并评估 Mamba-2 架构，并将其与其他架构进行比较。我们比较了预训练指标（困惑度）和零样本评估结果。模型规模（深度和宽度）遵循 GPT3 规范，从 1.25 亿到 27 亿不等。我们使用 Pile 数据集，并遵循文中描述的训练方案。

# D.1 MQAR详情

其中，非查询/键/值对的标记会被替换为随机标记。此外，与先前工作中使用的 MQAR 变体相比，我们使用了更多的键值对、更长的序列和更小的模型规模，所有这些都增加了任务的难度。
For each sequence length $T \in \{256, 512, 1024\}$, we use $T/4$ key-value pairs. The total vocab size is 8192.

We use a form of **curriculum training** where training cycles through datasets using $(T/32, T/16, T/8, T/4)$ key-value pairs, where each dataset has $2^{18} \approx 250000$ examples, for a total of $8$ epochs through each dataset (total of $2^{28} \approx 270M$ examples).
The total batch size is $2^{18} \approx 0.25M$ tokens (e.g. for $T=1024$, the batch size is $256$).

All methods use $2$ layers with default settings; the attention baseline additionally receives positional embeddings.
For each method, we sweep over model dimensions $\mathtt{D} = \{32, 64, 128, 256\}$ and **learning rates** $\{10^{-3.5}, 10^{-2}, 10^{-2.5}\}$.
**We use a linear decay schedule that drops on every epoch** (e.g. the last epoch would have a learning rate $1/8$ of the maximum/starting learning rate).以 Epoch 为最小单位进行更新步长（Step）的线性衰减，整个 32 Epochs 共享一条统一的学习率衰减曲线。

---
采用了课程学习（Curriculum Training）策略（即由易到难）：训练时按键值对数量逐步递增，每个数据集包含约 25 万个样本，每个数据集训练 8 个 epoch，总计训练约 2.7 亿样本。
### 1. MQAR 到底是一个什么任务？训练目标是什么？

**MQAR (Multiple Query Association Recall，多查询关联检索)** 是近年来在现代 RNN、SSM（如 Based、Mamba 等）以及线性注意力（Linear Attention）研究中，广泛使用的**合成基准测试（Synthetic Benchmark）**。
- **简单直观的比喻**： 你可以把 MQAR 看作一个“查字典/查电话簿”的记忆与检索测试。
    - 输入一串文本，里面包含了多对“键-值”（Key-Value, 比如 `电话号码A: 张三`, `电话号码B: 李四`...）。
    - 文本中间填充大量的无意义干扰 Token（padding/random tokens）。
    - 文本的末尾会提出一个或多个 Query（比如 `询问: 电话号码A 对应的是谁？`）。

- **模型的训练目标**： 模型需要像语言模型（Next-token Prediction）一样，根据前面的上下文，在看到 Query 的瞬间**精准 recall（检索/召回）出之前与之绑定的那个 Value**。

- **为什么要做这个任务？** 传统的 Softmax Attention（如 Transformer）因为有显式的 $O(T^2)$ KV-Cache 交互矩阵，做这种精准检索非常容易。但对于 SSM 或 Linear Attention（试图将复杂度降为 $O(T)$ 的模型），由于隐藏状态 $h_t$ 受到压缩限制，在面对长序列和大量 KV 对时非常容易丢失关联信息。因此 MQAR 被专门用来**考查模型在长序列中“精确记忆与关联检索”的能力极限**。
### 2. 为什么会有“多个数据集”？（课程学习 Curriculum Training）

段落中提到使用了 $(T/32, T/16, T/8, T/4)$ 四组 key-value 数量的数据集。
- **为什么不直接用最难的数据集训练？** 如果一上来就给模型喂满 $T/4$ 个 KV 对（比如序列长 1024，里面有 256 个键值对需要记忆），由于难度太高，从零开始训练的模型梯度很容易找不到正确的检索方向，导致模型“学塌了”（无法收敛，或者陷入随机猜词的局部最优）。
- **多数据集的作用（循序渐进）**： 这就是课程学习（Curriculum Training）的思想——“先教 1+1，再教微积分”。
    - **数据集 1**：极少 KV 对（$T/32$，比如 1024 长度里只有 32 个 KV 对，检索非常简单）。模型先学会“看到 Query 就去寻找前面对应的 Key/Value”的机制。
    - **数据集 2**：较少 KV 对（$T/16$）。
    - **数据集 3**：中等 KV 对（$T/8$）。
    - **数据集 4**：最终目标难度（$T/4$）。模型在已经掌握检索机制的基础上，挑战高密度的记忆容量。
这里所谓的“4 个数据集”，本质上是**生成规则相同，但按“KV 密集度/难度等级”划分的 4 级阶梯数据包**。
### 3. Epoch 和数据集、训练流程是怎么对应的？

```
[课程训练总流程]
数据集 1 (KV = T/32) ──> 训练 8 Epoch
     ↓
数据集 2 (KV = T/16) ──> 训练 8 Epoch
     ↓
数据集 3 (KV = T/8)  ──> 训练 8 Epoch
     ↓
数据集 4 (KV = T/4)  ──> 训练 8 Epoch
-------------------------------------
总计：4 个阶段 × 8 Epoch = 32 Epochs 的整体训练周期
```

- **每个 Epoch 的含义**： 每个数据集包含 $2^{18} \approx 250,000$ 个样本。 遍历完这 25 万个样本一次，就代表在该难度等级下完成了一个 **Epoch**。
- **每个数据集训练 8 Epoch**： 这意味着在每一个难度级别上，模型都会看到 $8 \times 25万 = 200万$ 个训练样本，充分巩固当前难度的记忆能力，然后再过渡到下一个更难的数据集。
- **整体训练量**： 4 个难度数据集共 $4 \times 8 = 32$ 个 Epoch，总计遍历约 $2^{28} \approx 2.7$ 亿个样本。
- **学习率衰减的匹配**： 文中提到的“每个 epoch 线性衰减”，就是指在整个包含 32 个 Epoch 的长线训练中，学习率从初始的最高值逐渐平滑下降，到了最后一个 Epoch 时降为初始值的 $1/8$，确保模型在最难的 $T/4$ 数据集上收敛得更平稳。

---
要理解这里的 Batch Size 为什么看起来“高达 25 万”，我们需要先区分两个概念：**样本级 Batch Size（Sequence/Example Count）** 和 **Token 级 Batch Size（Token Count）**。

在语言模型或序列模型训练中，经常使用 **Tokens 数量** 来度量批次大小。

## 1. 什么是 Batch Size？

在机器学习训练中，**Batch Size（批次大小）** 指的是模型在**进行一次梯度更新（更新一次参数）前，同时处理的数据量**。
- **如果按“样本数”（Sequence Count）算**：Batch Size 是指这一批包含多少条文本/序列。
- **如果按“Token 数量”（Token Count）算**：Batch Size 是指这一批中所有文本包含的 Token（字/词/符号）总数。  

**公式关系**：
$$\text{Total Tokens Batch Size} = \text{样本级 Batch Size (Sequences)} \times \text{序列长度 (T)}$$

## 2. 这里的 Batch Size 到底有多大？

文中提到的 $2^{18} \approx 250,000$（25万）指的是 **Total Token Batch Size**，也就是单次梯度更新所包含的**总 Token 数量**。如果换算回**样本数量（Sequences）**，其实并不大：

- 当序列长度 **$T = 256$** 时：$$\text{样本级 Batch Size} = \frac{250,000}{256} \approx 1024 \text{ 条样本}$$
- 当序列长度 **$T = 512$** 时：$$\text{样本级 Batch Size} = \frac{250,000}{512} \approx 512 \text{ 条样本}$$
- 当序列长度 **$T = 1024$** 时：$$\text{样本级 Batch Size} = \frac{250,000}{1024} = 256 \text{ 条样本}$$

从样本数来看（256 ~ 1024 条/批），这在深度学习训练中是非常标准的配置。
### 3. 为什么现代大模型/SSM 训练喜欢用这么大的 Token 级 Batch Size？

保持约 25 万 Tokens 的 Batch Size 主要有以下几点考虑：
1. **保证梯度估计的稳定性（降低梯度方差）** MQAR 是合成的联想记忆任务，模型需要学习在长序列的干扰 Token 中精准定位 KV 映射。如果 Batch Size 太小，单次采样的噪声极大，梯度容易剧烈震荡导致收敛缓慢甚至发散。25 万 Token 能提供极度平滑、稳定的梯度估计。
2. **充分发挥 GPU / TPU 的并行计算吞吐（Hardware Throughput）** 现代显卡（如 A100/H100）具有极高的并行张量计算能力。如果每次只喂几千个 Token，GPU 的 CUDA Core 大半时间都在等待数据搬运（Memory-bound）。将 Batch 填满到几十万 Token，可以实现近乎满载的矩阵乘法并行（Compute-bound），大幅提升训练速度。
3. **消除序列长度 $T$ 改变带来的影响（公平对比）** 在实验中，作者对比了 $T=256, 512, 1024$ 多种序列长度。如果固定“样本数量”（比如都用 256 条），那么 $T=1024$ 时单批总 Token 数就是 $T=256$ 时的 4 倍，这会导致不同序列长度下模型每步接收到的信息量不同。固定总 Token 数为 $2^{18}$ 保证了无论序列怎么变，模型每一步梯度更新看过的“总信息量”是完全一致的。

---

# D.2 Scaling Law Details

All models were trained on the Pile. For the scaling law experiments, we use the GPT2 tokenizer.

**Model Sizes.** First, we changed the batch size of the 1.3B model from 1M tokens to 0.5M tokens for uniformity.
Second, we changed the number of training steps and total tokens to roughly match Chinchilla scaling laws, which specify that training tokens should increase proportionally to model size.

**Training Recipes.** All models used the AdamW optimizer with
- gradient clip value $1.0$
- weight decay $0.1$
- no dropout
- linear learning rate warmup with cosine decay

$$
\begin{array}{llllllll}
\hline
\text{Params} & \mathtt{n\_layers} & \mathtt{d\_model} & \mathtt{n\_heads} / \mathtt{d\_head} & \text{Training steps}  & \text{Learning Rate}  & \text{Batch Size} & \text{Tokens} \\
\hline
\text{125M}   & 12                   & 768                 & 12 / 64                                  & 4800            & \text{6e-4}           & \text{0.5M tokens} & \text{2.5B}       \\
\text{350M}   & 24                   & 1024                & 16 / 64                                  & 13500           & \text{3e-4}           & \text{0.5M tokens} & \text{7B}         \\
\text{760M}   & 24                   & 1536                & 16 / 96                                  & 29000           & \text{2.5e-4}         & \text{0.5M tokens} & \text{15B}       \\
\text{1.3B}   & 24                   & 2048                & 32 / 64                                  & 50000           & \text{2e-4}           & \text{0.5M tokens} & \text{26B}       \\
\hline
\end{array}
$$
Scaling Law Model Sizes Our model sizes and hyperparameters for scaling experiments. (Model dimension and number of heads applies only to Transformer models.）

By default, the peak learning rate is the GPT3 specification.

Compared to GPT3 recipe, we use an improved recipe, inspired by changes adopted by popular large language models such as PaLM and LLaMa.
These include:
- linear learning rate warmup with cosine decay to $1e-5$, with a peak value of $5\times$ the GPT3 value
- no linear bias terms
-  RMSNorm instead of LayerNorm
- AdamW hyperparameter $\beta=(.9, .95)$ (the GPT3 value) instead of the PyTorch default of $\beta=(.9, .999)$

---
# D.3 Downstream Evaluation Details

To evaluate downstream performance of fully trained, we train Mamba-2 on 300B tokens on the Pile, using the GPT-NeoX tokenizer.

We use the same hyperparameters as the scaling experiments, except with batch size 1M for the 1.3B and 2.7B model.
For the 2.7B model, we also follow GPT3 specification (32 layers, dimension 2560).

For all models, we use 5x the learning rate of the corresponding GPT3 model.

For downstream evaluation, we use the LM evaluation harness from EleutherAI, on the same tasks as Mamba with one additional one:
- LAMBADA
- HellaSwag
- PIQA
- ARC-challenge
- ARC-easy: an easy subset of ARC-challenge
- WinoGrande
- OpenBookQA

---
**以 Mamba/SSM 为代表的状态空间模型通常在数值上比传统 Softmax Attention 更加稳定，因此能够容忍并受益于更大的学习率，从而实现更快的收敛速度。**

The Pile（包含 3000 亿 / 300B Token 的数据量）

分词器（Tokenizer）使用的是 GPT-NeoX 的分词器（词表大小约为 50,257）。

测试工具：采用了行业通用的标准评测框架 EleutherAI LM Evaluation Harness。   评估任务：涵盖了 Mamba-1 原论文（Gu and Dao 2023）使用的所有经典 NLP 评测集，并额外增加了一个（ARC-easy）。

**具体测试集包含**：

- **LAMBADA**：长距离常识预测/单词补全任务。

- **HellaSwag**：常识推理任务（预测事件的合理后续）。

- **PIQA**：物理常识问答任务。

- **ARC-challenge**：AI2 科学考试挑战集（难度较高的常识与科学推理题）。

- **ARC-easy**：ARC 任务的较简单子集（这是 Mamba-2 相比 Mamba-1 论文**额外新增**的测试集）。

- **WinoGrande**：代词消歧与常识推理任务。

- **OpenBookQA**：开卷科学问答与多步推理任务。
---
# D.4 Ablation Details

**(Re)Based Details.** Our ablations considered the Based “Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff” and ReBased “Linear Transformers with Learnable Kernel Functions are Better In-Context Models” models.

Based approximates the $\exp$ kernel with a quadratic Taylor expansion $\exp(x) \approx 1 + x + x^2/2$,
which can be accomplished 完成 by the feature map
$$\psi_{\text{Taylor}}(x) = \mathsf{concatenate}(1, x, 1/\sqrt{2} x \otimes x).$$
---
这段文本介绍了消融实验中涉及的 **Based** 与 **ReBased** 两组线性注意力架构的背景细节，重点推导了 **Based 模型如何通过二阶泰勒展开来近似 Softmax Attention 中的指数核（$\exp$ kernel）**。
### 1. 论文与模型背景（Model Context）
消融实验对比了两个经典架构：
    1. **Based**（Arora et al., 2024）：提出简单线性注意力模型，在召回能力（Recall）与吞吐量（Throughput）之间取得平衡。
    2. **ReBased**：引入可学习的核函数（Learnable Kernel Functions），提升线性 Transformer 的上下文能力。
### 2. Based 的二阶泰勒近似（Taylor Approximation）

- **解析**： 在传统 Softmax Attention 中，注意力权重计算依赖点积指数项 $\exp(q_i^\top k_j)$，这需要全量保留 $O(T^2)$ 的交互矩阵。 Based 为了将其转化为 **$O(T)$ 的线性注意力**（可写成累加状态/矩阵相乘结合律），将 $\exp(x)$ 在 $x=0$ 处展开为**二阶泰勒多项式**：
$$\exp(x) \approx 1 + x + \frac{x^2}{2}$$
### 3. 特征映射（Feature Map）构造原理

为了在向量点积中还原上述标量展开式（令 $x = q^\top k$），Based 构造了一个显示特征映射 $\psi_{\text{Taylor}}(x)$。
映射公式为：
$$\psi_{\text{Taylor}}(x) = \text{concat}\left(1, x, \frac{1}{\sqrt{2}} x \otimes x\right)$$

- **映射维度拆解**（设 $x \in \mathbb{R}^d$）：
  
    1. 标量数字 **$1$**：维度为 $1$。

    2. 原始向量 **$x$**：维度为 $d$。

    3. **外积展开项** **$\frac{1}{\sqrt{2}} x \otimes x$**：将向量做**张量积展开**（$\text{vec}(x x^\top)$），维度为 $d^2$。**Outer Product** 升维，让两个向量变矩阵

- **验证特征向量的点积（对应关系）**： 当分别对 $q$ 和 $k$ 进行该特征映射后求内积：$$\langle \psi(q), \psi(k) \rangle = 1 \cdot 1 \;+\; q^\top k \;+\; \left(\frac{1}{\sqrt{2}} q \otimes q\right)^\top \left(\frac{1}{\sqrt{2}} k \otimes k\right)$$
    **利用外积点积性质** $\langle q \otimes q, k \otimes k \rangle = (q^\top k)^2$，上述内积精确等于：
$$\langle \psi(q), \psi(k) \rangle = 1 + q^\top k + \frac{(q^\top k)^2}{2} \approx \exp(q^\top k)$$
通过这种显式的低阶多项式特征展开，Based 在无需计算 $\exp$ 的前提下保留了类似 Softmax 的高阶交互能力，同时允许利用结合律将计算复杂度降低到线性 $O(T)$。

---

ReBased proposes to use the simpler feature map $\psi_{\text{Quadratic}}(x) = x \otimes x$ **corresponding to the kernel transformation** $x^2$, but also applies a layer normalization beforehand. 预先
We view the layer normalization as an alternative **non-linear activation** to our default Swish activation, and ablate combinations of these.

Note that because these expand the feature dimension, we must project to smaller $B, C$ dimensions;
in T.7, use state size $N=64$ for 130M models and $N=256$ for 380M models.
For the (Re)Based methods, we project to 8 and 16 dimensions respectively before applying their feature maps;
this results in a total state size of $8^2 = 64$ for ReBased and $1+8+8^2=73$ for Based in the 130M model case.
Because the $B$ and $C$ projections are smaller, these methods use fewer parameters, and we adjust the layer count appropriately.

---

解释了**如何公平对比 Transformer/SSM 与 Based/ReBased 线性注意力模型**。核心矛盾在于：基于特征映射（Feature Map）的线性注意力方法会导致**隐状态（State Size $N$）爆炸**，为了控制状态维度和参数量一致，作者对输入维度进行了压缩（Down-projection）。

## 1. 背景：特征展开导致的状态维度爆炸

在 Linear Attention / SSM 中，模型需要维护一个状态矩阵。如果使用上文提到的 Based 特征映射 $\psi_{\text{Taylor}}(x) = \text{concat}(1, x, \frac{1}{\sqrt{2}} x \otimes x)$，包含了一项二阶外积 $x \otimes x$。

如果原始特征维度是 $d$：
- $x$ 的维度是 $d$。

- $x \otimes x$ 的外积维度会暴涨到 $d^2$。

如果不做维度压缩，写成状态矩阵时，隐藏状态尺寸 $N$ 就会变成 $d^2$，这会导致内存开销和计算量不可接受。

## 2. 解决方案：先降维（Project to Smaller Dimensions）

为了把总隐状态大小控制在与其他模型（如 Mamba 的 $N=64$ 或 $N=256$）相同的水平，作者在应用特征映射之前，**先将输入的 $B$（Key）和 $C$（Value/Query）矩阵投影（降维）到极小的维度 $d_{proj}$**：
- 130M 参数量的模型：降维到 $d_{proj} = 8$。

- 380M 参数量的模型：降维到 $d_{proj} = 16$。

## 3. 数学计算：如何算出 $N=64$ 和 $N=73$？

以 **130M 模型（降维到 $d_{proj} = 8$）** 为例：
## ReBased 的状态尺寸：$8^2 = 64$

ReBased 在构造特征映射时只保留二次交互项（Tensor Product）：
$$\psi_{\text{ReBased}}(x) \in \mathbb{R}^{d_{proj} \times d_{proj}}$$

- 当 $d_{proj} = 8$ 时，展开维度为 $8 \times 8 = 64$。

- 这与 130M 模型的目标状态尺寸 $N=64$ **完全对齐**。
##  Based 的状态尺寸：$1 + 8 + 8^2 = 73$

Based 使用完整的二阶泰勒展开式 $\psi_{\text{Taylor}}(x) = \text{concat}(1, x, \frac{1}{\sqrt{2}} x \otimes x)$，其特征映射维度为各部分维度之和：

- 标量项 $1$：维度为 $1$。

- 一阶线性项 $x$：维度为 $d_{proj} = 8$。

- 二阶外积项 $\frac{1}{\sqrt{2}} x \otimes x$：维度为 $d_{proj}^2 = 8^2 = 64$。

- **总状态尺寸**：$1 + 8 + 64 = 73$（略高于标准 64，但在同一个量级）。
### 4. 对参数量与层数（Layer Count）的影响

- **参数量减少**：因为 $B$ 和 $C$ 降维到了很小的维度（只有 8 或 16），用来做 $B, C$ 投影的线性层权重矩阵变小了。
- **层数补偿（Adjust the layer count）**：为了保证各种对比方法在**总参数量（Total Parameter Count）一致**的前提下进行公平比较，作者通过增加 Based/ReBased 的层数（Layers）来补齐因降维节省出来的参数量。

---
# 9.2 .1 Scaling Laws

For baselines, we compare against both Mamba and its Transformer++ recipe, which is based on the PaLM and LLaMa architectures (e.g.\ rotary embedding, SwiGLU MLP, RMSNorm instead of LayerNorm, no linear bias, and higher learning rates ).

As Mamba has already demonstrated that it outperforms the standard Transformer architecture (GPT3 architecture) as well as recent subquadratic architectures (H3, Hyena, RWKV-4, RetNet), we omit those in the plot for clarity.

---

# 9.2.3 Hybrid Models: Combining SSD Layer with MLP and Attention

近期及同期工作研究表明，**结合了SSM层和注意力层的混合架构可以提升模型质量**，**优于Transformer或纯SSM模型（例如Mamba）**，尤其是在上下文学习方面。我们探索了SSD层与注意力层和MLP的不同组合方式，以了解每种方式的优势。**实验结果表明，注意力层占总层数的10%左右效果最佳。** **此外，SSD层、注意力层和MLP的组合也优于纯Transformer++或Mamba-2。**

**SSD 和注意力** 我们发现SSD层和注意力层是互补的：单独使用时（例如在Mamba-2架构与Transformer++架构中），它们的性能（以困惑度衡量）几乎相同，**但SSD层和注意力层的混合使用优于纯Mamba-2或Transformer++架构。** 我们展示了一些结果，这些结果基于一个3.5亿字的模型（48层），该模型在Pile数据集上用GPT-2分词器训练了70亿个词元（参数数量、超参数、训练集和验证集均相同）。**仅添加几个注意力层就能带来显著的性能提升，并在质量和效率之间取得最佳平衡。** **我们假设SSM层可以很好地作为通用的序列到序列映射，而注意力层则作为一种检索机制，可以快速引用序列中的先前词元，而不是强制模型将所有上下文压缩到其内存（SSM状态）中。**

**结合SSD、MLP和注意力机制的混合模型**

我们比较了 SSD 与（门控）MLP 和注意力层结合的不同方式，并在 27 亿规模（64 层）上进行评估，在 Pile 上训练到 3000 亿个标记（参数数量相同，超参数相同，训练集和验证集相同，数据顺序相同）：
- Transformer++：32 个注意力层和 32 个门控 MLP，交错排列。
- Mamba-2：64 个 SSD 层。
- Mamba-2-MLP：32 个 SSD 层和 32 个门控 MLP 层，交错排列。
- Mamba-2-Attention：58 个 SSD 层和 6 个注意力层（索引为 9、18、27、36、45、56）在小规模实验中，我们发现，只要注意力层分布均匀，不在最开始或最结束的位置，模型质量就不会很大程度上取决于注意力层的确切位置。
- Mamba-2-MLP-Attention：28 个 SSD 层和 4 个注意力层，与 32 个门控 MLP 层交错排列。

总体而言，Transformer++ 和 Mamba-2 模型的质量大致相同。**我们发现，仅添加 6 个注意力层就能显著提升模型性能**，优于纯 Mamba-2 模型（以及 Transformer++）。添加 MLP 层会降低模型质量，但可以 (i) 由于 MLP 层的简洁性和硬件效率，加快训练和推理速度；(ii) 通过将 MLP 层替换为混合专家模型，更容易将其升级为混合专家模型。

---
# 9.3 速度基准测试
我们对 SSD 算法的速度进行了基准测试，并将其与 Mamba 的扫描实现和 FlashAttention-2 进行了比较。由于 SSD 重新表述，将矩阵乘法作为子程序使用，因此可以利用 GPU 上专门的矩阵乘法 (matmul) 单元，也称为张量核心。因此，它的速度提升了 2-8 倍。比 Mamba 的融合关联扫描速度更快，后者不使用矩阵乘法单元。由于 SSD 的速度与序列长度呈线性关系，因此从2K序列长度开始，它的速度就比 FlashAttention-2 更快。

However, we note that the Mamba-2 model as a whole might not be as efficient to train as Transformer at short sequence length (e.g. at $2K$), since a Transformer with $L$ layers would have $\frac{L}{2}$ MLP layers and $\frac{L}{2}$ attention layers, while a Mamba-2 model would have $L$ SSD layers for the same number of parameters.

Generally the MLP layers are very hardware efficient since they consist of simple matrix multiplication and pointwise linearity.
 one can also combine $\frac{L}{2}$ SSD layers and $\frac{L}{2}$ MLP layers to speed up training at short sequence length.

---

# 9.4.1 Architecture Ablations-Block Design

The ablations validate **that parallel projections to create** $(A,B,C,X)$ saves parameters and performs slightly better than Mamba's sequential projections.

More importantly, this modification is amenable顺从的，乐于接受的 to tensor parallelism at larger model sizes.Additionally, **the extra normalization layer also slightly improves performance.**
**More importantly, preliminary experiments at larger scales observed that it also helps with training stability.**

# 9.4.2 Head Structure

Sec.7 describes how the dimensions of the $B, C, X$ projections can be viewed as a hyperparameter analogous to notions of multi-head attention and multi-query attention. 

Strikingly,显著地 we find **a large difference between multi-value and multi-query or multi-key head patterns,** despite seeming very similar. Note that this is not explained by the total state size, which is the same for all of them (equal to $\mathtt{HPN}$ or the product of the number of heads, head dimension, and state dimension).

We also compare to multi-head patterns where the number of $C, B, X$ (analogous to $Q, K, V$) heads is equal.

We compare against the standard multi-head pattern, as well as one with aggressive sharing where they all have only $1$ head.
Note that in the latter case, the model still has $\mathtt{H}$ different sequence mixers $M$, because each head still has a different $A$.
When parameter matched, these multi-head patterns perform similarly to each other, in between the MVA and MQA/MKA patterns.

从表面的对称性来看，MQA、MKA 和 MVA 似乎只是把多头的位置换了换，但在状态空间模型（SSM）中，**MVA 的表现远好于 MQA 和 MKA**：

1. **Mamba 的表达能力集中在通道维度（Value / $X$）**：
在 SSM 中，输入 $X \in \mathbb{R}^{T \times P}$ 包含了丰富的通道特征。MVA 模式允许模型在不同的通道维度 $P$（即每个 Head）上解耦并独立演化隐藏状态，极大地提升了模型的特征表达丰富度。


2. **$B, C$（即 $K, Q$）的作用是全局索引调制**：
在 Mamba-1 中，$B$ 和 $C$ 通常维度极小或仅包含单头。它们的作用是提供全局的时间轴门控与选择性记忆。将 $B, C$ 单头化不仅极大地节省了参数量与矩阵乘法开销，而且实验证明，把主要的“头容量（Head Capacity）”分配给 $X$（即 MVA）能带来最佳的性能收益。

---

## SSM 变量与 Attention 的类比

在 Mamba-2 的 SSD（State Space Duality，状态空间双重性）理论中，$B, C, X$ 三个投影矩阵与注意力机制的对应关系如下：
- **$C \leftrightarrow Q$**（Query / 查询）
- **$B \leftrightarrow K$**（Key / 键）
- **$X \leftrightarrow V$**（Value / 值）
在传统 Transformer 中，改变 $Q, K, V$ 的 Head 数量对应不同的注意力变体（如 Multi-Query Attention MQA、Grouped-Query Attention GQA）。Mamba-2 借用了这一思想，通过控制 $C, B, X$ 在通道维度上的切分方式（Head 数量）来探索最佳结构。

在原始 Mamba-1 中，$B$ 和 $C$ 的 Head 维度通常被设为 1（即所有通道共享相同的 $B, C$ 标量或小向量），而输入数据 $X$ 在每个通道上独立演化（即 $X$ 拥有很多 Head）。 因此，**Mamba-1 在本质上等价于“多值注意力”（Multi-Value Attention, MVA）**——即控制查询（$Q$）和键（$K$）的头数很少，但控制值（$V$）的头数很多。

## 消融实验发现——MVA 与 MQA/MKA 存在巨大性能差距

- **解析**： 作者在表 5 中测试了不同的头结构配置：
    - **MVA (Multi-Value)**：$X$（即 $V$）多头，$B, C$ 单头。
    - **MQA/MKA (Multi-Query / Multi-Key)**：$C$ 或 $B$ 多头，其余单头。 表面上看起来这些切分方式很对称，但实验发现**它们的表达能力和最终性能差异巨大**。 注意：这种性能差异并不是由总状态容量（Total State Size）引起的，因为所有配置的总状态大小都是相等的（均满足 $H \cdot P \cdot N$，即 头数 × 头维度 × 隐状态维度）。
## 对称多头（Equal Head Count）与极端共享结构
- **解析**： 作者还测试了 $C, B, X$ 的头数完全相等的配置：
    1. **标准 Multi-Head**：$C, B, X$ 拥有相同的多头数 $H$。
    2. **极端共享（Single-Head Sharing）**：$C, B, X$ 全都只有 1 个头。值得注意的细节：即使 $C, B, X$ 都只有 1 个头，模型内部依然拥有 $H$ 个不同的序列混合器 $M$，因为**每个 Head 依然拥有独立控制时间尺度衰减的对角矩阵 $A$（即 $\Delta A$）**。
-  **结论**：当在相同参数量（Parameter Matched）的前提下进行比较时，这些对称多头模式的性能相近，介于 MVA（多值）和 MQA/MKA（多查询/多键）模式之间。

这一节的核心要点在于：

1. **Mamba-2 明确了多头结构的设计空间**：改变 $B, C, X$ 的 Head 分配，本质上是在调整状态空间表达能力在 $Q, K, V$ 三者之间的侧重。
2. **MVA 依然是 SSM 的最佳天然范式**：基于 SSM 演化而来的 Multi-Value 结构（即更多注意力放在特征/通道方向的 $X$ 上）比单纯模仿 Transformer 的 MQA/MKA 表现要优异得多。

---

在 Mamba-2 中，**序列混合器 $M$（Sequence Mixer）** 就是那个替代传统 Transformer 中 $O(T^2)$ 注意力矩阵的**因果掩码注意力核矩阵（Causal Attention / SSM Kernel Matrix）**。
## 1. 什么是序列混合器 $M$？

在传统的 Self-Attention 中，序列沿着时间维度 $T$ 的交互是由注意力矩阵 $A_{attn} = \text{softmax}(QK^\top / \sqrt{d})$ 完成的。

而在 Mamba-2 的 SSD 理论中，连续 SSM 的离散化递归表达可以精确等价地改写为一种**1D 序列变换矩阵 $M$**（维度为 $T \times T$）。

- 它的每个元素 $M_{i, j}$ 表示第 $j$ 个时刻的输入对第 $i$ 个时刻的状态贡献了多少（当 $j > i$ 时，$M_{i,j}=0$，即严格因果）：$$M_{i, j} = \begin{cases} \prod_{k=j+1}^{i} A_k = \exp\left(\sum_{k=j+1}^{i} \Delta A_k\right), & i \ge j \\ 0, & i < j \end{cases}$$
- 这里的 **$A$（或者说离散化的衰减参数 $\Delta A$）控制着历史信息的遗忘率与时间衰减速度**。
- **$M$ 的作用**：它就像一个沿着时间轴 $T$ 滚动的“混合器”，决定了过去的 Token 经过怎样的衰减后与当前 Token 混合。
## 2. 为什么即使 $C, B, X$ 只有 1 个头，依然有 $H$ 个不同的 $M$？

在 Mamba-2 的结构设计中：
1. **$C, B, X$ 共享单头（Single-Head Sharing）**： 意味着整个模型在输入维度上，只把序列投影成**1 套** $B \in \mathbb{R}^{T \times N}$、$C \in \mathbb{R}^{T \times N}$ 和 $X \in \mathbb{R}^{T \times P}$。也就是说，$Q, K, V$ 在特征通道方向不再切分成 $H$ 个独立的子空间。
2. **但对角衰减矩阵 $A$ 依然是多头的（$H$ 个 Head）**： 每个 Head 对应一个独立的对角参数 $A^{(h)}$（或者说独立的学习步长/衰减率 $\Delta A^{(h)}$）。
    - **Head 1** 的 $A^{(1)}$ 衰减极慢，它的混合器 $M^{(1)}$ 能捕捉**极长距离**的上下文交互；
    - **Head 2** 的 $A^{(2)}$ 衰减极快，它的混合器 $M^{(2)}$ 只关注**局部相邻**的 Token 交互；
    - **Head $H$** 的 $A^{(H)}$ 则关注中等距离。
因为每个 Head 依然保留了独立的衰减动态 $A^{(h)}$，所以它就能生成一个独立的 $T \times T$ 维度的混合矩阵 $M^{(h)}$。
## 3. 直观对比

- **标准 Transformer 多头注意力**： 由 $Q^{(h)}, K^{(h)}$（投影权重不同）共同决定了 $H$ 个不同的注意力矩阵 $A_{attn}^{(h)}$。
- **Mamba-2 极端共享模式（Single-Head Sharing）**： $B, C, X$ 的投影权重完全单头化（共享），但由于每个 Head 拥有独立的**时间衰减率 $A^{(h)}$**，模型依然构建出了 **$H$ 个作用在时间轴 $T$ 上不同衰减粒度的序列混合器 $M^{(1)}, M^{(2)}, \dots, M^{(H)}$**。

这就解释了为什么就算大幅压缩 $B, C, X$ 的头数，Mamba-2 依然没有退化成单头模型——因为它的多头能力一部分是由**时间维度的多衰减率 $A$（即多序列混合器 $M$）** 来支撑的。

---
在 Mamba-2 的 SSD框架中，公式里出现的 $\Delta A_k$（更严格地写作 $\Delta_k A_k$ 或 $(\Delta A)_k$）表示**在第 $k$ 个时间步（Time Step）上，经过离散化步长（Step Size）$\Delta_k$ 调制后的连续时间衰减率（Decay Rate）**。

在连续时间状态空间模型（SSM）中，隐状态的微分方程为：
$$\dot{h}(t) = A h(t) + B x(t)$$
当将其离散化到离散序列（Token index $k$）时，连续矩阵 $A$ 需要通过零阶保持器（Zero-Order Hold, ZOH）或欧拉法映射到离散矩阵 $\bar{A}_k$：
$$\bar{A}_k = \exp(\Delta_k A_k)$$
因此，将指数映射带入累积连乘项中：
$$\prod_{k=j+1}^{i} \bar{A}_k = \prod_{k=j+1}^{i} \exp(\Delta_k A_k) = \exp\left( \sum_{k=j+1}^{i} \Delta_k A_k \right)$$
## 1. 拆解各变量含义
- **$A_k$（连续空间衰减率）**： 在 Mamba-2 中，$A$ 通常被限制为**对角矩阵（Diagonal Matrix）**且元素恒为负实数（例如 $A \le 0$）。它控制着系统隐状态随时间的**本征衰减速度（Intrinsic Decay Rate）**。
- **$\Delta_k$（输入依赖的离散化步长 / 时间步长 Step Size）**： 这是 Mamba 相比传统 Linear RNN 产生动态选择机制（Selective Mechanism）的核心。$\Delta_k > 0$ 是根据当前时刻的输入 $x_k$ 动态计算出来的标量（或向量）：$$\Delta_k = \text{softplus}(\text{Linear}(x_k))$$
- **$\Delta_k A_k$（离散化后的有效衰减指数）**： 两者相乘后，$\Delta_k A_k$ 作为一个负数标量/向量，直接控制当前 Step 信息的遗忘程度。
### 2. 直观物理理解：动态门控（Gating）与遗忘机制

如果你将 SSD 的序列混合器矩阵 $M$ 与传统 Transformer 的 Attention / Forget Gate 对比，$\exp\left( \sum_{k=j+1}^{i} \Delta_k A_k \right)$ 的作用完全等价于**连续时间下的动态遗忘门（Continuous Forget Gate）**：
- **当 $\Delta_k \to 0$ 时**： $\Delta_k A_k \to 0$，导致 $\exp(0) = 1$。 此时系统**完全不衰减（不遗忘）**，第 $j$ 个时刻的信息可以**无损地穿透**第 $k$ 个时刻直接传到第 $i$ 个时刻（对应模型选择性地“记住”历史信息）。
- **当 $\Delta_k$ 很大时**： $\Delta_k A_k$ 是一个很大的负数，导致 $\exp(\Delta_k A_k) \to 0$。 此时系统**剧烈衰减（迅速遗忘）**，第 $k$ 个时刻直接斩断了第 $j$ 个时刻对未来的影响，重置隐状态（对应模型选择性地“忽略/冲刷掉”无关的噪声 Token）。
在 Mamba-2 中，通过这种以对角矩阵 $A$ 和动态步长 $\Delta_k$ 构成的指数累加和 $\sum \Delta_k A_k$，模型不仅实现了严格因果的 1-Semiseparable 矩阵结构，还赋予了 $M$ 在 $O(T)$ 线性计算下实现长短期选择性记忆的能力。

---
## 1. 公式逐项拆解

- **$x_k$（当前输入 Token 的特征向量）**： 在序列的第 $k$ 个时间步，输入模型的 Token 隐藏状态（维度为 $D$）。
- **$\text{Linear}(x_k)$（线性投影 / 权重）**： 将 $x_k$ 通过一个无激活函数的线性层（矩阵乘法 $W_\Delta x_k + b_\Delta$）映射为一个标量或低维向量。这一步的作用是让模型根据**当前输入的语义内容**来决定该给多大的时间步长。
- **$\text{softplus}(\cdot)$（平滑激活函数）**： Softplus 函数定义为 $\text{softplus}(z) = \ln(1 + e^z)$。
    - **作用**：它是 ReLU 的平滑版本，确保输出值**恒为严格正数（$\Delta_k > 0$）**。
    - **为什么必须大于 0**：在状态空间模型的连续时间离散化中，时间步长 $\Delta$ 在物理上代表“采样间隔/持续时间”，几何意义上必须为正数；同时在 $\exp(\Delta_k A_k)$ 连续衰减中，$A_k < 0$ 搭配 $\Delta_k > 0$ 才能保证指数衰减不发生数值爆炸。
## 2. 物理含义与直观理解

$\Delta_k$ 决定了模型在处理当前 Token 时，**“时间向前走了多久”** 或者说 **“信息筛选的关卡开得有多大”**。
结合离散化衰减项 $\bar{A}_k = \exp(\Delta_k A_k)$（其中 $A_k < 0$）：
1. **当 $x_k$ 是无用噪声 / 填料 Token 时（如标点符号、无意义介词）**：
    - 神经网络学会将 $\text{Linear}(x_k)$ 输出为一个很大的正数，使得 $\Delta_k$ 变大。
    - 指数衰减项 $\exp(\Delta_k A_k) \to 0$（急剧衰减）。
2. **当 $x_k$ 是重要信息 / 关键 Token 时（如需要长距离记忆的 Key/Value 关键词）**：
    - 神经网络将 $\text{Linear}(x_k)$ 输出为一个极小的值，使得 $\Delta_k \to 0$。
    - 指数衰减项 $\exp(\Delta_k A_k) \to \exp(0) = 1$（几乎无衰减）。

在 Mamba 中，模型不需要存储显式的 $O(T^2)$ KV-Cache，而是通过这个简单的 **$\Delta_k = \text{softplus}(\text{Linear}(x_k))$** 实时计算当前时刻的门控响应，使得 $O(T)$ 复杂度的 SSM 能够像 Softmax Attention 一样灵活地控制信息的选择性保留与遗忘。

---
这四个概念（**MHA**、**MQA**、**MKA**、**MVA**）本质上都是在探讨**注意力/状态空间模型在处理多头结构（Multi-Head Structure）时，如何分配 $Q, K, V$（或 Mamba-2 中的 $C, B, X$）在隐空间与通道维度上的“多头数（Head Count）”**。

它们的核心差异在于**谁在独立拆分多头，谁在跨头共享参数**。以下是具体对比：

---
# 9.4.3 注意力核近似

We also ablate adding a normalization term, akin to类似于 the denominator of the softmax function in standard attention.
**We found that this introduced instabilities to most variants,** but slightly improved performance for the ReLU activation function $\psi$.

T.7 also tests more recent proposals to improve linear attention that involve expanding the feature dimension (Based and ReBased).
These linear attention extensions aim to appropriate拟合 the $\exp$ kernel with a quadratic approximation. 
ReBased also proposes to replace the QK activation function with a layer normalization;
**from an SSM-centric view we apply a normalization on top of $(B, C)$ before applying the SSM function.**
We note that this technique has been independently proposed as the QK-Norm for softmax attention and an internal normalization for Mamba.

Overall, T.6 and T.7 found that the kernel approximation methods we tried did not seem to improve over simple pointwise 逐点 non-linear activation functions for $\psi$.
Thus our default settings for Mamba-2 used $\psi(x) = \mathsf{Swish}(x)$ to follow Mamba-1,
**but we suggest that removing this activation entirely may be a simpler choice that we did not extensively test.**

We emphasize however that SSD and vanilla linear attention differ in the inclusion包容/包含 of the 1-semiseparable mask $L$, while the various linear attention methods in the literature were derived to approximate 近似 softmax attention without this term; thus, our negative results may be not unexpected.

---

### 深入物理机制：SSD 与 Vanilla Linear Attention 的本质差异

在计算形式上，两者的矩阵表达对比如下：

1. **Vanilla Linear Attention（传统线性注意力）**：
$$Y = (Q K^\top \odot \text{CausalMask}) V$$

这里只有简单的因果下三角掩码（全 1 下三角矩阵），没有距离衰减。
2. **SSD (State Space Duality, Mamba-2)**：
$$Y = (Q K^\top \odot L) V$$

这里的掩码矩阵 $L$ 是一个 **1-半可分矩阵（1-Semiseparable Matrix）**：
$$L_{i,j} = \begin{cases} \exp\left(\sum_{k=j+1}^{i} \Delta_k A_k\right), & i \ge j \\ 0, & i < j \end{cases}$$
这个掩码 $L$ 赋予了模型类似 SSM 的动态遗忘/选择性记忆能力。

正是因为多了这个矩阵 $L$，SSD 才能在保留 $O(T)$ 线性计算效率的同时，实现超越传统线性注意力的召回与表达能力。而试图直接拿不带 $L$ 假设的**现有线性注意力核函数**来硬套 SSD，效果不好是完全符合理论预期的。
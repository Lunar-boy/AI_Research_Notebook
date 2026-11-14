# 1. Abstract.

高吞吐量的大型语言模型 (LLM) 服务需要一次性处理足够多的请求。然而，现有系统难以胜任，因为每个请求的键值缓存 (KV 缓存) 内存巨大，并且会动态增长和收缩。如果管理不善，这些内存会因碎片化和冗余复制而造成大量浪费，从而限制批处理大小。为了解决这个问题，我们提出了 PagedAttention，这是一种受操作系统中经典虚拟内存和分页技术启发的注意力算法。

在此基础上，我们构建了 vLLM，一个 LLM 服务系统，它实现了 (1) KV 缓存内存的近乎零浪费，以及 (2) 在请求内部和请求之间灵活共享 KV 缓存，从而进一步降低内存使用量。我们的评估表明，vLLM 将常用 LLM 的吞吐量提高了 2-4 倍。

与 FasterTransformer 和 Orca 等最先进的系统相比，vLLM 的延迟水平相同。对于更长的序列、更大的模型和更复杂的解码算法，这种改进更为显著。vLLM 的源代码可在 https://github.com/vllm-project/vllm 公开获取。
```
@inproceedings{kwon2023efficient,
  title={Efficient memory management for large language model serving with pagedattention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph and Zhang, Hao and Stoica, Ion},
  booktitle={Proceedings of the 29th symposium on operating systems principles},
  pages={611--626},
  year={2023}
}
```

---
# 2. Intro

许多云公司都在竞相提供LLM的托管服务 https://aws.amazon.com/bedrock/ 
https://openai.com/blog/openai-api
然而，运行这些应用程序成本非常高昂，需要大量的硬件加速器，例如 GPU。根据最近的估计，处理一个 LLM 请求可能比传统关键词查询贵10倍 。鉴于这些高昂的成本，提高LLM服务系统的吞吐量（从而降低每次请求的成本）变得越来越重要。

LLM 的核心是一个自回归 Transformer 模型 ，该模型根据输入（prompt）和之前生成的输出词元序列，逐个生成单词（tokens） 。对于每个请求，这个耗时的过程都会重复进行，直到模型输出一个**终止词元** termination token 为止。这种顺序生成过程使得工作负载受限于内存，未能充分利用GPU的计算能力，从而限制了服务吞吐量。

在GPU中，会有固定一部分显存用于存放模型权重，同时会有KV Cache，会根据每个服务的请求进行分配或者释放，对于 Transformer 模型，这些状态由与注意力机制相关的键值张量组成，通常称为KV 缓存。arXiv preprint arXiv:2211.05102 这些缓存用于存储上下文信息，这些信息来自之前的词元，用于按顺序生成新的输出词元。


少量剩余内存用于存储其他数据，包括激活值——即在评估 LLM 时创建的临时张量。由于模型权重是恒定的，且激活值仅占用 GPU 内存的一小部分，因此 KV 缓存的管理方式对于确定最大批处理大小至关重要。如果管理效率低下，KV 缓存内存会显著限制批处理大小，从而降低 LLM 的吞吐量

vLLM可以平滑现有系统中KV缓存内存的快速增长曲线。
https://github.com/NVIDIA/FasterTransformer. Nvidia FasterTransformer 2023a.
Orca: A Distributed Serving System for 
{Transformer-Based} Generative Models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). 521–538.

通过将多个请求批量处理，可以提高吞吐量。然而，为了批量处理大量请求，必须高效管理每个请求的内存空间。

本文观察到，现有的LLM服务系统在有效管理 KV 缓存方面存在不足。这主要是因为它们将请求的 KV 缓存存储在连续的内存空间中，就像大多数深度学习框架一样
Pytorch: An imperative style, high-performance deep learning library.
Tensorflow arXiv preprint arXiv:1712.06139 (2017)

需要将张量存储在连续内存中。然而，与传统深度学习工作负载中的张量不同，键值缓存具有独特的特性：**它会随着模型生成新的词元而动态增长和收缩，并且其生命周期和长度事先未知**。这些特性使得现有系统的方法在两个方面效率低下：

首先，现有系统会受到内部和外部内存碎片化的影响。为了将请求的键值缓存存储在连续空间中，它们会预先分配一块与请求最大长度（例如，2048 tokens）相同的连续内存块。这会导致严重的内部碎片化，因为请求的实际长度可能远小于其最大长度。

此外，即使事先知道实际长度，预分配仍然效率低下：由于整个内存块在请求的生命周期内都被占用，其他较短的请求无法利用当前未使用的内存块。另外，外部内存碎片化也可能很严重，因为每个请求的预分配大小可能不同。事实上，我们在图 2中的分析结果表明，**现有系统中只有 20.4% 到 38.2% 的键值缓存内存用于存储实际的token状态。**

其次，现有系统无法利用内存共享的优势。LLM 服务通常使用高级解码算法decoding algorithms，例如并行采样 parallel sampling和束搜索beam search，这些算法每次请求都会生成多个输出。在这种情况下，请求包含多个序列，这些序列可以部分共享其键值缓存。然而，由于序列的键值缓存存储在独立的连续空间中，因此现有系统无法实现内存共享。

---
# Remark1: Decoding algorithms

在大型语言模型（LLM）的推理过程中，高级解码算法是指为了提高文本生成质量、多样性或效率，在模型输出概率分布后，采用更复杂的机制来选择或组合单词的策略。

相比于每次只走一步、只选择一个最优解的常规算法（如贪婪搜索），高级解码算法通常会同时探索或生成多个可能的文本路径。

## 并行采样 Parallel Sampling

当给模型一个提示词（Prompt）时，系统不是让模型只生成一个回答，而是同时（并行）进行多次独立的随机采样，一次性生成 N个不同的完整回答。

在每一步生成时，模型会根据当前语境计算出下一个词的概率分布。采样算法（如结合了 Top-p、Top-k 或 Temperature 的采样）会根据这个分布随机抽取一个词。并行采样就是把这个随机抽样的过程并发执行𝑁次。

因为这 𝑁个独立的回答都是基于完全相同的提示词（Prompt）开始的。在生成第一个词之前，提示词部分的键值（KV）缓存对这 𝑁个序列来说是完全一模一样的。

## 束搜索 (Beam Search)

传统的贪婪搜索每一步只选概率最高的那个词，但眼前最好的词不一定能带来整体最好的句子。束搜索为了解决这个问题，在每一步都会保留当前最优的 𝐵个候选序列(𝐵被称为束宽，Beam Width）

第一步：预测接下来的词，选择概率最高的 𝐵个词，形成 𝐵 个初始路径。 第二步：基于这 𝐵 个路径，分别预测下一个词。如果原本有 𝐵个路径，每个路径再衍生出词表大小的可能性，系统会计算所有组合的累积概率，并再次只保留全局概率最高的𝐵个新路径。依此类推，直到生成结束，最后从这 𝐵个完整的备选文本中选出综合得分最高的一个或几个

在束搜索的过程中，不同的候选路径往往拥有共同的祖先（前几个词是完全一样的）。这意味着，这些并行推进的序列在分叉之前，其历史文本的 KV 缓存是完全可以复用的。

---
# 3. PagedAttention

We refer to the concatenation of the prompt and output lists as **sequence.**

我们提出了PagedAttention，这是一种受操作系统解决内存碎片和共享方案（即带分页的虚拟内存）启发的注意力算法。PagedAttention 将请求的键值缓存 (KV 缓存) 划分为多个块，每个块可以包含固定数量标记的注意力键和值。在 PagedAttention 中，KV 缓存块不一定存储在连续的空间中。因此，我们可以像操作系统的虚拟内存一样，以更灵活的方式管理 KV 缓存：可以将块视为页，标记视为字节，请求视为进程。

这种设计通过使用相对较小的块并按需分配来缓解内部碎片。此外，由于所有块的大小相同，它也消除了外部碎片。最后，它实现了块级的内存共享，可以跨与同一请求关联的不同序列，甚至跨不同的请求。

vLLM 采用与 PagedAttention 协同设计的块级内存管理和抢占式请求调度机制。实现了 KV 缓存内存的近乎零浪费。vLLM 支持流行的 LLM，例如 GPT OPT 以及LLaMA（实际上更多）

甚至超过单个 GPU 内存容量。我们在各种模型和工作负载上的评估表明，与最先进的系统相比 vLLM 将 LLM 服务吞吐量提高了 2-4 倍。
且完全不影响模型精度。对于更长的序列、更大的模型和更复杂的解码算法，改进效果更为显著（§ 4.3）。

具体来说，PagedAttention 将每个**序列**的 KV 缓存划分为KV块。Each block contains the key and value vectors for a fixed number of tokens, which we denote as KV block size $(B)$. Denote the key block $K_j=(k_{(j-1)B+1},...,k_{jB})$ and value block $V_j=(v_{(j-1)B+1},...,v_{jB})$. The attention computation can be transformed into the following block wise computation:

$$A_{ij}=\frac{exp(q_i ^\top K_j / \sqrt d)}{\sum_{t=1}^{\lceil i / B \rceil}exp(q_i^\top K_t1/\sqrt d) }, \quad o_i=\sum_{j=1}^{\lceil i/B \rceil}V_jA_{ij}^\top,$$ 
where $A_{ij} =(a_{i,(j-1)B+1},...,a_{i,jB})$ is the row vector of attention score on j-th KV block.
在注意力计算过程中，PagedAttention 内核会分别识别并获取不同的键值对块。
假设每个token的特征维度为d，$q_i$是行向量, 第i个token的查询向量， 形状 $1 \times d$；$K_j$是第j个block中包含的B个token的建向量，形状$d\times B$ ；$A_{ij}$注意力分数块块，行向量，第i个query token对第j个KV block中所有B个token的注意力权重，形状$1\times B$ ；$V_j$ 矩阵，第j个Block中包含的B个Token的值向量，形状$d \times B$  

## Token

是大语言模型处理文本的最小文本单位。模型在拿到一段文本（如 "Learning AI"）时，无法直接理解字符串，需要先通过 Tokenizer（分词器） 将文本切分成 Token 序列。每个 Token 在模型的词表中都有一个唯一的数字编号（即 Token ID），例如 `learning → 1421`

## **为什么 Token 会有“特征维度”？**

模型不能直接拿数字 ID 进行数学运算（比如矩阵乘法、注意力计算），所以需要将数字 ID 转为连续的向量表达。模型包含一个巨大的嵌入表（Embedding Table）。输入 Token ID 后，通过查表操作，将其映射为一个固定长度的高维数值向量（e.g.,d=4096数组）。这个向量的长度就是特征维度。向量中的每一个维度代表了该 Token 在高维语义空间中的某一种抽象特征（如语法特征、词性、情感倾向、语义上下文等）。

## 词表和嵌入映射的关系

**词表**（Vocabulary）：是一个包含固定大小V的离散符号集合，词表中每一个Token 都有一个唯一的整数索引（Token ID）

**嵌入映射**（Embedding Matrix）：在模型中表现为一个巨大的权重矩阵E,维度为$V \times d$  d为模型的hidden size特征维度

映射在数学上等价于用一个 One-Hot 独热向量，长度为V，只有第i位是1,其余为0, 去乘以嵌入矩阵E $OneHot(i) \cdot E$ ，最终得到高维向量$1 \times d$ 

嵌入映射赋予了词语义与几何关系, E中的参数是在模型训练过程中不断学习调整的，训练完成后，语义相似或经常在相同语境下出现的 Token，它们在嵌入矩阵中所对应的向量会在高维空间中彼此靠拢。

在许多主流的大语言模型中，词表和嵌入映射还存在一个首尾呼应的镜像关系：输入端（Embedding）：用词表 ID 查表，将离散 Token 转化为连续特征向量（从V维压缩到 d 维）。输出端（LM Head / Output Projection）：在模型最后一层，用线性层将最终的特征向量投影回词表大小（从 d 维扩展到 V 维），得到每一个 Token 的 Logits。

**在很多模型架构中，输出端的投影矩阵和输入端的嵌入矩阵甚至共享同一块权重数据（称为 Weight Tying），这直接将词表与嵌入映射在模型输入和输出两端紧密连接在一起。**


## 服务引擎vLLM

vLLM 采用集中式调度器来协调分布式 GPU 工作节点的执行，在 PagedAttention 的支持下， KV 缓存管理器能够以分页的方式有效地管理 KV 缓存。具体来说，KV 缓存管理器通过集中式调度器发送的指令来管理 GPU 工作节点上的物理 KV 缓存。

## KV缓存管理器

vLLM内存管理器的核心思想类似于虚拟内存，在操作系统中，内存被划分为固定大小的页面，并将用户程序的逻辑页面映射到物理页面。连续的逻辑页面可以对应于不连续的物理内存页面，从而允许用户程序像访问连续内存一样访问内存。此外，物理内存空间无需预先完全预留，操作系统可以根据需要动态分配物理页面。vLLM 利用虚拟内存的思想来管理 LLM 服务中的键值缓存。借助 PagedAttention，我们将键值缓存组织成固定大小的键值块，类似于虚拟内存中的页面。

请求的键值缓存由一系列逻辑键值块表示，随着新令牌及其键值缓存的生成，逻辑键值块从左到右依次填充。最后一个键值块的未填充位置会保留给后续generation。

在 GPU 工作节点上，块引擎block engine会分配一块连续的 GPU DRAM 内存，并将其划分为物理键值块（CPU RAM 也采用类似的交换机制；参见 § 4.5）。键值块管理器还维护块表block tables——记录每个请求的逻辑键值块和物理键值块之间的映射关系。每个块表条目记录逻辑块对应的物理块以及已填充的位置数量。将逻辑键值块和物理键值块分离，使得 vLLM 能够动态地扩展键值缓存，而无需预先为所有位置预留内存，从而消除了现有系统中的大部分内存浪费

---
# 4. 调度和抢占 Scheduling and Preemption

vLLM 对所有请求采用先到先服务 (FCFS) 调度策略，以确保公平性并防止资源饥饿。当 vLLM 需要抢占请求时，它会确保最早到达的请求优先得到服务，而最后到达的请求优先被抢占。

LLM 服务面临着一个独特的挑战：LLM 的输入提示长度可能差异很大，而最终的输出长度事先未知，取决于输入提示和模型。随着请求及其输出数量的增长，vLLM 可能会耗尽 GPU 的物理块来存储新生成的键值缓存。在这种情况下，vLLM 需要回答两个经典问题：（1）应该驱逐哪些块？Which blocks should it evict?（2）如果需要再次访问，如何恢复被驱逐的块？How to recover evicted blocks if needed again?

通常，驱逐策略使用启发式方法来预测未来最有可能被访问的块，并驱逐该块。由于在我们的案例中，我们知道序列中的所有块都会同时被访问，因此我们采用了一种“要么全部驱逐，要么全部不驱逐”的驱逐策略all-or-nothing eviction policy。此外，一个请求中的多个序列（例如，一个束搜索请求中的束候选）会作为一个序列组进行调度。由于序列组内的各个序列可能存在内存共享，因此它们总是同时被抢占或重新调度。为了回答第二个问题，即如何恢复被驱逐的内存块，我们考虑两种技术：

**交换Swapping**。这是大多数虚拟内存实现所采用的经典技术，它将被驱逐的页面复制到磁盘上的交换空间。在我们的方案中，我们将被驱逐的块复制到 CPU 内存。除了 GPU 块分配器之外，vLLM 还包含一个 CPU 块分配器来管理交换到 CPU RAM 的物理块。当 vLLM 耗尽用于新令牌的空闲物理块时，它会选择一组序列进行驱逐，并将其键值缓存转移到 CPU。一旦 vLLM 抢占了一个序列并驱逐了其块，它就会停止接受新的请求，直到所有被抢占的序列都处理完毕。请求完成后，其块将从内存中释放，而被抢占序列的块将被重新加载以继续处理该序列。需要注意的是，在这种设计中，交换到 CPU RAM 的块数永远不会超过 GPU RAM 中的物理块总数，因此 CPU RAM 上的交换空间受限于分配给键值缓存的 GPU 内存。

**Recomputation重新计算**。在这种情况下，当被抢占的序列重新调度时，我们只需重新计算键值缓存。请注意，重新计算的延迟可以显著低于原始延迟，因为解码时生成的令牌可以与原始用户提示符连接成一个新的提示符——它们在所有位置的键值缓存都可以在一次提示符阶段迭代中生成。

交换和重新计算的性能取决于 CPU RAM 和 GPU 内存之间的带宽以及 GPU 的计算能力。我们在原始论文第7.3节中考察了交换和重新计算的速度。

---
# 5. 分布式执行 Megatron-LM

许多LLM的参数规模超过了单个GPU的容量因此，有必要将它们划分到分布式 GPU 上，并以模型并行的方式执行它们，这就需要一个能够处理分布式内存的内存管理器。vLLM 在分布式环境中非常有效，因为它支持 Transformer 上广泛使用的 Megatron-LM 风格的张量模型并行策略arXiv:1909.08053
该策略遵循 SPMD（单程序多数据）执行调度，其中线性层被划分以执行分块矩阵乘法，GPU 通过 all-reduce 操作不断同步中间结果。具体而言，注意力算子按注意力头维度进行划分，每个SPMD 进程负责多头注意力中的一个注意力头子集。

我们观察到，即使采用模型并行执行，每个模型分片仍然处理相同的输入标记集same set of input tokens，因此对相同位置的键值缓存 (KV Cache) 的需求也相同。因此，vLLM 在集中式调度器中采用单一的 KV Cache 管理器，不同的 GPU 工作节点共享该管理器以及从逻辑块到物理块的映射。这种公共映射使得 GPU 工作节点能够使用调度器为每个输入请求提供的物理块来执行模型。尽管每个 GPU 工作节点具有相同的物理块 ID，但每个工作节点仅存储其对应注意力头的 KV Cache 的一部分。

在每个步骤中，调度器scheduler首先为批次中的每个请求准备包含 input token IDs的消息，以及每个请求的块表block table。接下来，调度器将此控制消息广播给 GPU 工作节点。然后，GPU 工作节点开始使用 input token IDs执行模型。在注意力层中，GPU 工作节点根据控制消息中的块表读取 KV 缓存。在执行过程中，GPU 工作节点使用 all-reduce 通信原语同步中间结果，而无需调度器的协调，arXiv:1909.08053最后，GPU 工作进程会将本次迭代的sampled tokens发送回调度器。总之，GPU 工作进程无需同步内存管理，因为它们只需在每次解码迭代开始时接收所有内存管理信息以及步长输入即可。In summary, GPU workers do not need to synchronize on memory management as they only need to receive all the memory management information at the beginning of each decoding iteration along with the step inputs.

---
# 6. 内核级优化

由于 PagedAttention 引入了现有系统无法有效支持的内存访问模式，我们开发了几个 GPU 内核来对其进行优化。

(1)融合重塑和块写入。 Fused reshape and block write 在每个 Transformer 层中，新的 KV 缓存被分割成块，重塑为针对块读取优化的内存布局，然后保存到块表指定的位置。为了最大限度地减少内核启动开销，我们将它们融合到一个内核中。

(2)融合块读取和注意力机制。 Fusing block read and attention
我们采用了 FasterTransformer 中的注意力内核  https://github.com/NVIDIA/FasterTransformer. 为了根据块表读取键值缓存并动态执行注意力操作，我们为每个块分配一个 GPU warp 进行读取。此外，我们还支持请求批次内可变序列长度。 

(3)融合块复制。Fused block copy. 由写时复制机制发出的块复制操作可能作用于不连续的块。如果我们使用cudaMemcpyAsync API， 这可能会导致多次小数据移动调用。为了减轻这种开销，我们实现了一个内核，将不同块的复制操作批量合并到单个内核启动中。

---
# 7. Result

并行采样和束搜索，计算方法为：通过共享节省的数据块数除以未共享时的总数据块数。结果显示，并行采样可节省 6.1% 至 9.8% 的内存，而束搜索可节省 37.6% 至 55.2% 的内存。在相同的 ShareGPT 数据集实验中，并行采样可节省 16.2% 至 30.5% 的内存，而束搜索可节省 44.3% 至 66.3% 的内存。

## 内核微基准测试

PagedAttention 中的动态块映射会影响涉及存储的键值缓存的 GPU 操作的性能，例如块读写和注意力机制。与现有系统相比，我们的 GPU 内核（第5节）需要额外的开销来访问块表、执行额外的分支以及处理可变序列长度。与高度优化的 FasterTransformer 实现相比，这导致注意力内核的延迟增加了 20% 到 26%。我们认为这种开销很小，因为它只影响注意力算子，而不影响模型中的其他算子，例如线性算子。尽管存在这种开销，PagedAttention 仍然使 vLLM 在端到端性能方面显著优于 FasterTransformer

## 块尺寸的影响block size

块大小的选择会对 vLLM 的性能产生显著影响。如果块大小过小，vLLM 可能无法充分利用 GPU 的并行性来读取和处理 KV 缓存。如果块大小过大，内部碎片化会增加，共享概率会降低。

我们使用 ShareGPT 和 Alpaca 数据集，在固定请求速率下，采用基本采样，评估了不同块大小的 vLLM 性能。在 ShareGPT 数据集中，块大小从 16 到 128 均能获得最佳性能。在 Alpaca 数据集中，虽然块大小为 16 和 32 时性能良好，但更大的块大小会显著降低性能，因为序列长度会小于块大小。在实践中，我们发现块大小为 16 既足以有效利用 GPU，又足以避免大多数工作负载中出现明显的内部碎片。因此，vLLM 将其默认块大小设置为 16。

## 比较重新计算和交换

vLLM 支持重新计算和交换两种恢复机制。为了理解这两种方法的优劣，我们评估了它们的端到端性能，并对其开销进行了微基准测试，结果表明，对于较小的块大小，交换会产生过大的开销。这是因为较小的块大小通常会导致 CPU 和 GPU 之间进行大量的小数据传输，从而限制了 PCIe 的有效带宽。相比之下，重新计算的开销在不同块大小下保持不变，因为重新计算不使用 KV 块。因此，当块大小较小时，重新计算效率更高；而当块大小较大时，交换效率更高，尽管重新计算的开销从未超过交换延迟的 20%。对于 16 到 64 的中等块大小，这两种方法表现出相当的端到端性能。

---
# Remark2:带分页的虚拟内存

早期操作系统也面临着和 LLM 推理一模一样的困境：程序（请求）必须占用连续的物理内存空间。
操作系统通过引入带分页的虚拟内存（Paged Virtual Memory）完美解决了内存碎片和内存共享两大难题。

## 什么是虚拟内存与分页？

**在操作系统中，程序不能直接访问物理内存（RAM），而是看到一个连续的、虚拟的内存空间。**

- 页（Page）：操作系统将虚拟内存划分为固定大小的连续逻辑块（例如 4KB）

- 页框 / 块（Page Frame / Block）：操作系统将物理内存也划分为同样大小的实际物理块。

- **页表（Page Table）：这是最关键的纽带。它像一本字典，记录了虚拟页到物理块的映射关系。**

**虚拟出来的内存看起来是连续的，但在物理内存里，这些块可以散落在任何地方。**

## 操作系统如何解决“内存碎片”？

外部碎片（External Fragmentation）

不分页的痛点：程序 A 申请了 10MB，程序 B 申请了 10MB。后来程序 A 释放了，内存里空出了 10MB。此时程序 C 想申请 12MB 连续空间，虽然整个系统剩余内存很大，但由于没有连续的 12MB 空闲空间，程序 C 无法运行。

分页如何解决：因为物理块的大小完全相同（比如都是 4KB），操作系统可以把物理内存像积木一样切碎。程序 C 申请 12MB，操作系统只需要在物理内存里凑出 3000 个 4KB 的空闲块即可，它们不需要连续，通过页表把它们拼成虚拟的连续空间。因此，彻底消除了外部碎片。

内部碎片（Internal Fragmentation）

不分页的痛点：为了防止外部碎片，早期系统可能会给每个程序分配固定的大块内存（比如每次都给 64MB）。如果程序只需要 5MB，剩下的 59MB 就被浪费了，这叫内部碎片。

分页如何解决：分页虽然也有内部碎片（如果一个程序需要 5KB，分配了 2 个 4KB 的页，最后一个页只用了 1KB，浪费了 3KB），但因为页的粒度非常小（通常是 4KB），即使浪费，最多也只浪费最后一个页的一部分。因此，极大地缓解了内部碎片。

## 操作系统如何实现“内存共享”？

**在操作系统中，不同的进程（Process）可以通过让各自的页表指向同一个物理内存块，来实现极其高效的内存共享。**

场景 A：共享只读代码库（静态共享）

当你在电脑上同时打开 10 个 Chrome 浏览器窗口时，这 10 个进程都需要加载 Chrome 的基础核心代码。

做法：操作系统不需要在物理内存里复制 10 份同样的代码。它只在物理内存里存一份，然后让这 10 个进程的虚拟页表，全都指向这同一块物理内存。

场景 B：分叉与写时复制（Fork & Copy-on-Write，动态共享）

当一个父进程创建（Fork）一个子进程时，子进程在初始状态下和父进程几乎完全一样。

做法：操作系统不会立刻为子进程复制一份完整的内存。相反，它让子进程的页表直接指向父进程的物理块，并将这些块标记为“只读”。

触发复制：只有当子进程尝试修改某一个页的数据时，操作系统才会触发一个错误，把这一个页的数据复制到新的物理块中，并更新子进程的页表。这就叫写时复制（CoW）。

---
# A.1 基于Transformer的大型语言模型

The task of language modeling is to model the probability of a list of tokens $(x_1, \dots, x_n)$. Since language has a natural sequential ordering, it is common to factorize the joint probability over the whole sequence as the product of conditional probabilities (a.k.a. autoregressive decomposition)
由于语言具有自然的顺序性，因此通常将整个序列的联合概率分解为条件概率的乘积（也称为自回归分解 ）
A neural probabilistic language model. Yoshua Bengio

$$P(x) = P(x_1) \cdot P(x_2 \mid x_1) \cdots P(x_n \mid x_1, \dots, x_{n-1})$$
The most important component of a Transformer-based language model is its self-attention layers. For an input hidden state sequence $(x_1, \dots, x_n) \in \mathbb{R}^{n \times d}$, a self-attention layer first applies linear transformations on each position $i$ to get the query, key, and value vectors:

$$q_i = W_q x_i, \quad k_i = W_k x_i, \quad v_i = W_v x_i.$$
### 线性代数基础以及llm基础

其中n代表序列长度，d代表隐藏层维度。例如"I Love you"切分为3个token，n为3

矩阵与向量相乘的基础法则
$$W_q \in \mathbb{R}^{d_k \times d}, \quad x_i \in \mathbb{R}^{d \times 1} \implies q_i = W_q x_i \in \mathbb{R}^{d_k \times 1}$$
$$q_i = W_q x_i$$
$$(d_k \times d) \cdot (d \times 1) \longrightarrow d_k \times 1$$

Then, the self-attention layer computes the attention score $a_{ij}$ by multiplying the query vector at one position with all the key vectors before it and compute the output $o_i$ as the weighted average over the value vectors:

$$a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{t=1}^i \exp(q_i^\top k_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^i a_{ij} v_j$$
All components in the Transformer model, including the embedding layer, feed-forward layer, layer normalization ({Ba et al., 2016}), residual connection ({He et al., 2016}), output logit computation, and the query, key, and value transformation, are all applied independently position-wise in a form of $y_i = f(x_i)$

传统的RNN中当前位置i的计算必须依赖前一时刻i-1的隐状态 $y_i=f(x_i,h_i-1)$ 这导致计算无法并行。**而 Transformer 的设计哲学是：把“跨位置的信息交互”与“单位置的特征提炼/变换”彻底解耦。**
- 跨位置交互：只交给 Self-Attention 来做。例如通过$q_i^Tk_j$跨位置计算注意力
- 单位置变换： 其余所有组件在计算第i个位置输出$y_i$时，只用第i个位置输入$x_i$, 与其他位置特征无关。

e.g., Embedding Layer每一个 token ID 直接独立查表映射为向量$x_i$, 
QKV变换是每一个位置独立乘以权重矩阵，
残差链接是每个位置独立进行向量加法$y_i=x_i+SubLayer(x_i)$, 
Layer Normalization是每一个位置独立对自身的d维向量求均值与方差并归一化$y_i=LN(x_i)$, 
FFN是每一个位置独立通过一个前馈神经网络$y_i=W_2 \cdot ReLU(W_{1}x_i+b_1)+b_2$ , 
Output Logit Computation是每一个位置独立乘以词表映射矩阵预测下一个词$logits_i=W_{vocab}x_i$ 

其中，$logits_i$是一个长度等于词表大小的得分向量。向量里的每一个数字，代表词表里对应词的原始得分(Logit)，分数越高说明模型认为这个词接在后面越合适。拿到logits后会进行归一化，比如softmax，之后会进行采样，如topK，根据概率选出最高的词作为真正预测输出的下一个词。

Logit 是一个特定的数学函数，它是 Logistic（逻辑） 和 Unit（单元） 的组合词。定义是几率（Odds）的自然对数（Log）
$$logit(p)=ln(\frac{p}{1-p} )$$
p是事件发生的概率，$\frac{p}{1-p}$  是odds几率表示发生与不发生的比例，取对数后把在$[0,1]$之间的数映射到$[-\infty,+\infty]$ 的实数范围内。在深度学习中，网络最后一层的线性输出也是一个$[-\infty,+\infty]$的实数。因为这个实数对应softmax/sigmoid激活函数激活前的状态，在数学形式上对应了logit函数的逆运算，所以把这层未归一化的原始得分叫做logits。

**这带来了三个工程与理论优势**
- 并行计算Parallelism， 既然每个位置的输出互不依赖，那GPU可以同时开辟n个线程，并行处理
- 权重共享Parameter Sharing，所有位置共享同一个函数f（即相同的矩阵W_q W_k W_v...），大大减少了模型的参数量，且使模型具备了处理变长序列的能力
- 矩阵批量处理Batch Execution，在实现层面，n个独立的y_i运算可以直接拼成整个矩阵乘以权重的运算如$XW_q^\top$或$FFN(X)$，发挥GPU Tensor Core矩阵乘法的吞吐。

---
# A.2 LLM服务与自回归生成

A request to an LLM service provides a list of input prompt tokens$(x_1,...,x_n)$ and the LLM service generates a list of output tokens $(x_{n+1},...,x_{n+T})$ We refer to the concatenation of the prompt and output lists as sequence.

由于序列联合概率分布公式的分解，LLM 只能逐个采样并生成新词元，每个新词元的生成过程都依赖于序列中所有先前词元的键值向量。在这个顺序生成过程中，现有词元的键值向量通常会被缓存以用于生成未来的词元，这被称为键值缓存。需要注意的是，一个词元的键值缓存依赖于其所有先前的词元。这意味着同一个词元在序列中不同位置的键值缓存会有所不同。

给定请求提示，LLM 服务中的生成计算可以分解为两个阶段：

- **The prompt phase** takes the whole user prompt as input and computes the probability of the first new token $P(x_{n+1}\mid x_1,...,x_n)$ During this process, also generates the key vectors $k_1,...,k_n$ and value vectors $v_1,..., v_n$ .Since prompt tokens $x1,...,x_n$ are  all known, the computation of the prompt phase can be parallelized using matrix-matrix multiplication operations. Therefore this phase can efficiently use the parallelism inherent in GPUs.
- The autoregressive generation phase generates the remaining new tokens sequentially. At iteration t, the model takes one token $x_{n+t}$ as input and compute the probability$P(x_{n+t+1} \mid x_1,...,x_{n+t})$ with key vector $k_1,...,k_{n+t}$ and value vectors $v_1,...,v_{n+t}$ 注意这里键值向量从1到n+t-1都被cached在先前迭代里，只有新的key value向量会在本轮计算。This phase completes either when the sequence reaches a maximum length (specified by users or limited by LLMs) or when an end-of-sequence (`<eos>`) token is emitted. **The computation at different iterations cannot be parallelized due to the data dependency and often uses *matrix-vector multiplication*, which is less efficient. As a result, this phase severely underutilizes GPU computation and becomes memory-bound, being responsible for most portion of the latency of a single request.**

---
# A.3 LLM 的批处理技术

通过批量处理多个请求，可以提高LLM服务的计算利用率。由于请求共享相同的模型权重，权重移动的开销会分摊到批次内的所有请求中，并且当批次大小足够大时，计算开销会超过权重移动的开销。然而，批量处理LLM服务的请求并非易事，原因有二。首先，请求到达的时间可能不同。简单的批量处理策略要么会让较早的请求等待较晚的请求，要么会延迟到达的请求直到较早的请求完成，从而导致显著的排队延迟。其次，请求的输入和输出长度可能差异很大。直接的批量处理技术会填充请求的输入和输出以使其长度相等，从而浪费GPU计算和内存。

为了解决这个问题，可以采用细粒度批处理机制，例如单元批处理Low latency rnn inference with cellular batching.
和迭代级调度Orca: A Distributed Serving System for {Transformer-Based} Generative Models. 
与传统的请求级处理方法不同，这些技术在迭代级运行。每次迭代后，已完成的请求会从批次中移除，新的请求则会被添加进去。因此，只需等待一次迭代即可处理新的请求，而无需等待整个批次完成。此外，借助特殊的GPU内核，这些技术无需对输入和输出进行填充。通过减少排队延迟和填充带来的效率损失，这种细粒度的批处理机制显著提高了LLM服务的吞吐量。

---
# Remark3：Weight movement Overhead

权重移动的开销（Weight Movement Overhead），在硬件层面通常被称为内存带宽瓶颈或从显存（VRAM）到算力核心（SRAM/ALU）的传输延迟 。LLM（如拥有数十亿或数百亿参数的模型）的模型权重体量巨大，平时是储存在 GPU 的显存（HBM 或 VRAM）中的。

当一条请求进来时，GPU 的计算核心（如 Tensor Cores）需要对这些数据进行数学运算。计算核心不能直接在显存里计算，必须把模型权重从显存“搬运”到计算核心内部的高速缓存（SRAM）中。 “搬运”的速度（内存带宽）远远慢于“计算”的速度（算力）。如果每次只处理一个请求，GPU 就要为了这一个请求把庞大的模型权重在显存和核心之间搬一遍。这时候，GPU 绝大多数时间都在“等数据运过来”，计算核心处于闲置状态，这就是巨大的权重移动开销（或访存开销）。

当你把多个请求打包成一个“批次（Batch）”时，GPU 只需要搬运一次模型权重，就可以让这个权重同时服务于批次内的几十个请求。这样，权重移动的开销就被几十个请求共同分摊（Amortized）了，硬件利用率（吞吐量）会大幅提升。

## 为什么要“填充（Padding）”请求的输入和输出？

填充（Padding） 是为了迎合 GPU 的张量（Matrix/Tensor）并行计算特性而做出的妥协。GPU 之所以擅长处理 AI 任务，是因为它有成千上万个计算核心，可以同时进行大规模的矩阵乘法。在标准的深度学习框架中，批处理（Batching）本质上是将多个请求合并成一个大矩阵（张量）送入 GPU。

矩阵的形状必须是规则的矩形（即每一行的长度必须完全一致）。然而，现实中用户的请求五花八门，直接批处理的做法： 为了把请求 A 和请求 B 塞进同一个矩阵里，系统必须以最长的那个请求（10个 Token）为基准。对于较短的请求 A，系统会在其后面或者前面塞入一堆无意义的占位符（通常是 `[PAD]` 标签），硬生生把 3 个 Token 变成 10 个 Token。尽管` [PAD]` 没有任何实际意义，但 GPU 在进行矩阵运算时，依然会老老实实地对这些填充出来的虚无 Token 进行乘法和加法计算，并为其在显存中开辟空间（KV Cache）。

---
# A.4 Memory Challenges in LLM Serving

尽管细粒度批处理减少了计算资源的浪费，并允许以更灵活的方式批处理请求，但可以同时批处理的请求数量仍然受到 GPU 内存容量的限制，特别是分配给存储键值缓存的空间。换句话说，服务系统的吞吐量受限于内存。克服这种内存限制需要解决内存管理方面的以下挑战：

**大型 KV 缓存**。KV缓存的大小会随着请求数量的增加而迅速增长。例如，对于 13B 参数 OPT 模型单个token的 KV 缓存需要 800 KB 的空间，2 (key and value vectors) $\times$  5120 (hidden state size)$\times$
40(number of layers) $\times$ 2 (bytes per FP16) . 由于 OPT 最多可以生成 2048 个 token 的序列，因此存储一个请求的 KV 缓存所需的内存可能高达 1.6 GB。并发 GPU 的内存容量通常在几十 GB 以内。即使将所有可用内存都分配给 KV 缓存，也只能容纳几十个请求。此外，低效的内存管理会进一步减小批处理大小，另外，鉴于目前的趋势，GPU 的计算速度增长速度超过了内存容量的增长速度
Ai and memory wall. RiseLab Medium Post 1 (2021), 6. 
例如，从NVIDIA A100到H100，浮点运算性能提升超过2倍，但GPU显存容量最大仍为80GB。因此，我们认为显存将日益成为性能瓶颈。

**复杂的解码算法。**  LLM 服务提供一系列解码算法供用户选择，每种算法对内存管理复杂性的影响各不相同。例如，当用户从单个输入提示中请求多个随机样本时，a typical use case in program suggestion https://github.com/features/copilot  the KV cache of the prompt part, which accounts for 12% of the total KV cache memory in our experiment (§6.3), can be shared to minimize memory usage. On the other hand, the KV cache during the autoregressive generation phase should remain unshared due to the different sample results and their dependence on context and position. The extent of KV cache sharing depends on the specific decoding algorithm employed. In more sophisticated algorithms like beam search (Sequence to sequence learning with neural networks.) different request beams can share larger portions (up to 55% memory saving, see §6.3) of their KV cache, and the sharing pattern evolves as the decoding process advances.

**针对未知输入输出长度的调度** LLM服务的请求在输入和输出长度上存在差异。这就要求内存管理系统能够适应各种不同的提示符长度。此外，随着解码过程中请求输出长度的增长，其键值缓存所需的内存也会增加，并可能耗尽可用于处理新请求或现有提示符的可用内存。系统需要做出调度决策，例如从 GPU 内存中删除或交换某些请求的键值缓存。

## 现有系统中的内存管理

现有系统中的块预分配方案存在三个主要的内存浪费来源：为未来token预留的内存槽、由于过度配置以应对潜在的最大序列长度而导致的内部碎片，以及来自内存分配器（例如伙伴分配器）的外部碎片。外部碎片永远不会用于生成的token，这一点在处理请求之前就已经确定。内部碎片也未被使用，但只有在请求完成采样后才会意识到这一点。它们都是纯粹的内存浪费。虽然预留的内存最终会被使用，但在整个请求期间都保留这部分空间，尤其是在预留空间很大的情况下，会占用原本可以用于处理其他请求的空间。我们在图 2中可视化了实验中内存浪费的平均百分比，结果表明，先前系统中实际有效内存可能低至 20.4%。


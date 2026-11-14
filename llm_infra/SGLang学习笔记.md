# Abstract

大型语言模型 (LLM) 越来越多地用于处理需要多次链式生成调用、高级提示技术、控制流以及与外部环境交互的复杂任务。然而，目前缺乏高效的 LLM 编程和执行系统。为了弥补这一不足，我们引入了 SGLang，一种用于LLM 的结构化生成语言。SGLang 旨在高效地进行 LLM 编程，并集成了常见 LLM 编程模式的原语。我们已将 SGLang 实现为嵌入 Python 的领域特定语言，并为其开发了解释器、编译器和高性能运行时环境。这些组件协同工作，实现了并行、批处理、缓存、共享和其他编译技术等优化。此外，我们还提出了 RadixAttention，这是一种新颖的技术，它为基数树中的所有请求维护一个键值 (KV) 缓存的最近最少使用 (LRU) 缓存，从而在运行时实现跨多次生成调用的 KV 缓存自动重用。SGLang 简化了 LLM 程序的编写，并提高了执行效率。我们的实验表明，SGLang 可以将常见的 LLM 任务速度提高至多5倍同时降低代码复杂度并增强控制。

---
# 1. intro

LLM能力的提升拓宽了其用途，使其能够处理更广泛的通用任务并作为自主代理运行 。在这些应用中，LLM 会进行多轮规划、推理以及与外部环境的交互。这是通过使用工具来实现的 。这些新的用例都需要多次（通常是相互依赖的）LLM 生成调用，而且至关重要的是，这些程序的控制流可能依赖于LLM的输出 。这些复杂模式的出现标志着我们与语言学习模型（ LLM ）的交互方式发生了转变，从简单的轮流聊天转向了更复杂的LLM程序化使用方式

为了顺应新兴趋势，社区见证了诸如 LangChain, LMQL, Guidance, DSPy
等新interfaces的发展和日益普及 。这些工具可以帮助用户轻松管理提示模板和程序流程。然而，它们主要侧重于前端设计，这往往会导致运行时性能下降。

相反，像 NVIDIA TensorRT-LLM 这样的后端推理引擎,Hugging Face TGI, Orca and vLLM 尽管延迟和吞吐量持续降低，但它们的设计在很大程度上仍然基于传统的单代调用接口。它们通常支持类似于 OpenAI Completion API 
. 其中，用户prompt将被处理，并且响应将以无状态的方式返回。stateless manner.

尽管前端和后端技术都在快速发展，但它们的发展轨迹却截然不同。前端技术在很大程度上仍然对LLM推理过程一无所知，而后端技术往往缺乏对应用程序结构的了解，无法跨多个调用进行优化。这种脱节的发展导致错失了许多性能优化的机会，例如第2.3节讨论的键值缓存共享和并行处理。

本文提出了一种高效的LLM编程系统整体方法，该方法通过协同设计前端语言和后端运行时来实现。这一过程面临三大关键挑战：(i) 创建一种灵活的、领域特定的语言，其基本元素适用于LLM编程和生态系统集成；(ii) 通过自动并行、批处理、缓存以及跨多个调用和多个程序的共享，确保程序高效执行；(iii) 通过编译器优化（例如代码移动和预取注解）来提升程序性能。

在前端，我们引入了 SGLang，一种嵌入 Python 的领域特定语言。它提供了一系列专门用于简化 LLM 编程的原语。这些原语支持对提示和生成进行操作（例如，extend、gen、select），以及控制并行性（例如，fork、join）。此外，SGLang 与 Python 的控制流和库兼容。利用这些原语以及原生 Python 语法，用户可以直观地开发复杂的提示策略和代理，而无需手动优化。此外，统一的语言增强了可移植性，使用户能够轻松地在不同的开源和专有模型（例如 OpenAI 和 Anthropic 的模型）之间进行转换。

我们为 SGLang 开发了解释器和编译器。解释器负责管理提示符状态并执行 SGLang 程序。具体来说，它将提示符视为一个流，并将基本操作提交到该流中进行异步执行，it treats a prompt as a stream and submits primitive operations to this stream for asynchronous execution, 从而确保对同步和程序内部并行性的正确控制。ensuring proper control over synchronization and intra-program parallelism. 此外，我们还可以跟踪 SGLang 程序，并利用编译​​器将其编译成计算图。这种方法为更积极的优化提供了可能。我们在 LLM 编程的背景下，探索了几种经典的编译器优化方法，例如代码移动。code movement

在后端，我们提出了一种名为 RadixAttention 的新技术，用于在多次生成调用中自动重用键值缓存 (KV 缓存)。在现有系统中，请求的 KV 缓存会在处理完成后被丢弃，这导致 KV 缓存无法在多次生成调用中重用。而我们的系统则为基数树内的所有请求维护一个 LRU 缓存。这种方法使得运行时能够高效地实现各种重用模式的自动共享。此外，为了提高缓存命中率，我们还采用了一种缓存感知调度策略。

总而言之，我们做出了以下贡献：

- 我们提出了 SGLang，一种具有 LLM 编程原语的领域特定语言，支持并行、控制流、嵌套调用和外部调用。
- 我们开发了一种能够与 Python 解释器协同执行 SGLang 程序并将提示状态作为异步流进行管理的解释器。
- 我们构建了一个能够将程序编译成计算图的编译器，并在 LLM 编程的背景下探索经典的编译器技术。
- 我们提出了 RadixAttention，这是一种在运行时跨多个生成调用自动重用 KV 缓存的新技术。
- 我们使用 SGLang 实现了各种 LLM 应用，包括智能体控制、逻辑推理、内容生成、基准测试和长文档处理。我们的实验表明，SGLang 可以将常见的 LLM 任务速度提升高达 5倍同时降低代码复杂度并增强控制。

---
# 2. 背景

## 2.1 LLM的推理过程

目前大多数使用的语言学习模型，例如 GPT-3，PaLM，以及 LLaMA，基于自回归Transformer架构这些模型基于序列中前面的词元预测下一个词元的概率。在推理过程中，模型首先通过前向传播处理输入词元序列。然后，它按顺序解码输出词元，每个词元都依赖于前面的词元。我们将接收输入词元序列并生成输出词元序列的过程称为一次生成调用。**a single generation call** 在此过程中，每个词元都会生成一些中间张量，这些张量用于解码后续词元。这些中间张量被称为“KV缓存”，其名称来源于自注意力机制中的键值对。在后续讨论优化时，一个重要的观察是，KV缓存的计算仅依赖于所有前面的词元，因此具有相同前缀的不同序列可以重用前缀词元的KV缓存，从而避免冗余计算。

## 2.2 The Paradigm of Programming LLMs

LLMs越来越多地被用作复杂系统中的多功能任务求解器，在这些应用中，LLM 必须作为可靠且智能的组件运行。这使得 LLM 的使用方式从单一、直接的生成调用演变为多个相互依赖的生成调用与控制流交错进行，从而催生了“LLM 编程”范式。**Programming LLMs**指的是开发用于控制 LLM 生成过程的计算机程序。这一概念可以通过多种方式加以阐释。

首先，高级提示技术是 LLM 编程的示例。元推理就是一个典型的例子 。*arXiv:2304.13007, 2023. Answering questions by meta-reasoning over multiple chains of thought.* 为了解决问题， LLM首先通过链式思维提示生成多个解决方案。然后，它对这些解决方案进行元推理，从而得出最终解决方案。另一个例子是skeleton-of-thought。为了回答一个问题， LLM首先生成答案的框架。然后，它发出多个并行调用，将框架扩展成详细的解释。更复杂的情况是思维树方法 。它将 LLM 调用集成到树搜索算法（例如，广度优先搜索）中。

其次，在基于LLM的代理中，LLM的程序化使用至关重要。以ReAct代理为例 *React: Synergizing reasoning and acting in language models.* 例如，智能体的行为由循环和条件语句控制。智能体每次迭代执行一个动作。在每次迭代中，智能体选择动作类型（搜索或完成），生成动作参数，并处理来自外部环境的观测数据。

第三，更简单的提示形式也属于 LLM 编程的范畴。诸如少样本提示之类的技术few-shot prompting，chain-of-thought prompting，and retrieval-augmented generation are basic elements in programming LLMs.

## 2.3 Challenges and Opportunities

一个优秀的系统应该能够帮助程序员提高效率，为常见的编程任务提供直观且富有表现力的接口，包括LLM状态管理、解码过程控制以及与外部工具的无缝集成。此外，通过优化低延迟和高吞吐量的执行，确保系统性能至关重要。特别是，LLM程序的丰富结构和LLM推理的独特属性为运行时优化提供了大量机会。这些机会包括：

- 缓存：在 LLM 程序中，通常会将多个文本段和生成调用附加到单个提示符上。将先前标记的计算 KV 缓存缓存到多个链式调用中，可以减少冗余计算。然而，这种优化并非免费且简单，因为它需要额外的存储空间和更复杂的内存管理。
-  批处理： LLM 中的解码过程主要受内存限制，这意味着增加批处理大小可以显著提高吞吐量。批量运行大量 LLM 程序，并利用连续批处理技术可能是有益的。
- 共享：在 LLM 程序中，通常会从单个提示符生成多个输出，或者从当前状态派生一个新的提示符。vLLM 中已经研究了基本前缀共享。更高级的共享模式，例如不规则树状结构共享，也可以采用。现有系统均无法自动处理所有这些模式。
- 并行性： LLM 程序通常包含多个生成调用。通过为这些调用创建依赖关系图，我们可以识别独立的调用并并行执行它们。这便创建了程序内并行性，从而在原有的程序间并行性基础上增加了额外的并行性。
- 编译：程序完整编写完成后，可以将其编译成优化的中间表示形式，以提高执行效率。还可以实现更高级的优化，例如根据用户提供的测试用例自动调整原始提示。这些优化可能包括展开、压缩、重新排列提示的某些部分，以及为不同的调用选择不同的模型。
- 
为了系统地探索这些机遇，协同设计编程接口和运行时至关重要。然而，像 Orca 和 vLLM 这样的推理引擎存在不足，因为它们仅针对单次生成调用进行优化。这种局限性会导致在执行涉及多次调用的 LLM 程序时出现冗余计算和错失良机。
## 现有LLM编程系统

鉴于 LLM 编程的重要性，已经开发出专门的编程系统。顶层包含带有预构建模块的高级应用程序开发库，例如 LangChain。The top layer comprises high-level application development libraries with pre-built modules. The middle layer features more flexible pipeline programming frameworks like DSPy . At the bottom layer, there are domain-specific languages designed for controlling a single prompt, including LMQL and Guidance.
 中间层具有更灵活的流水线编程框架，例如 DSPy 底层是一些特定领域的语言，用于控制单个提示符，包括 LMQL and Guidance.

SGLang 的独特之处在于它介于中间层和底层之间，与 LMQL 和 Guidance 有相似之处。然而，与主要关注语言设计而往往忽略运行时效率的 LMQL 和 Guidance 不同，SGLang 巧妙地将前端语言特性与后端运行时优化相结合。这种方法显著提高了运行时效率，有时甚至能提升一个数量级或更多。LMQL 的前端采用类似 SQL 的自定义语法，这与更直观的 Python 语法形成鲜明对比。Guidance 最初使用 Handlebars 模板语言，后来过渡到嵌入在 Python 中的 Pythonic 语言。受 Guidance 启发而开发的 SGLang 前端语言，正与新版 Guidance 同步开发。

SGLang 引入了新的原语，以增强表达能力和运行时效率。它还提供了解释器和编译器两种模式的灵活性。表 1列出了 LMQL、Guidance 和 SGLang 之间的比较。由于 LangChain 和 DSPy 的层次更高，因此我们省略了与它们的比较。可以将 LangChain 和 DSPy 的高级流水线编译成 SGLang。

| Language | Syntax | Primitives                      | Runtime Backends                                         | Execution Modes          |
| :------- | :----- | :------------------------------ | :------------------------------------------------------- | :----------------------- |
| LMQL     | Custom | extend, gen, select             | Hugging Face Transformers, llama.cpp, OpenAI             | Compiler                 |
| Guidance | Python | extend, gen, select             | Hugging Face Transformers, llama.cpp, OpenAI, VertexAI   | Interpreter              |
| SGLang   | Python | extend, gen, select, fork, join | SGVM, OpenAI, Anthropic, Hugging Face TGI\*, llama.cpp\* | Interpreter and compiler |

Table 1: Comparison among LMQL, Guidance, and SGLang. The backends marked with * are experimental.

---
# 3. Language, Interpreter, and Compiler

## 3.1语法和基本元素

SGLang 是以 Python 中嵌入式领域特定语言的形式实现的。

```python
dimensions = ["Clarity", "Originality", "Evidence"]

@function
def essay_judge(s, essay):
    s += "Please evaluate the following essay. " + essay
    # Evaluate an essay from multiple dimensions in parallel
    forks = s.fork(len(dimensions)) ①
    for f, dim in zip(forks, dimensions):
        f += (
            "Evaluate based on the following metric: " +
            dim + ". End your judgement with the word 'END'"
        )
        f += "Judgment: " + f.gen("judgment", stop="END") ②
    # Merge judgments
    for f, dim in zip(forks, dimensions):
        s += dim + ": " + f["judgment"] ③
    # Generate a summary and give a score
    s += "In summary, " + s.gen("summary")
    s += "I give the essay a letter grade of " + \
    s += s.gen("grade", choices=["A", "B", "C", "D"]) ④

ret = essay_judge.run(essay="A long essay ...") ⑤
print(ret["grade"])
```
该程序使用分支求解合并提示方法从多个维度评估文章质量，函数essay_judge接受两个参数：s和essay 。变量s管理提示的状态，允许使用+=添加新的字符串和 SGLang 原语。函数首先将 essay 添加到提示中，然后将提示分成三个副本。这些副本并行运行，从不同维度生成判断结果。最后，函数合并这些判断结果，生成摘要，并为 essay 分配一个字母等级。

SGLang 中多维作文评判器的实现采用了分支求解合并提示技术
*Branch-solve-merge improves large language model evaluation and generation. arXiv:2310.15123, 2023.*
此函数使用 LLM 从多个维度评估文章质量，合并评估结果，生成总结并给出最终成绩。高亮区域展示了 SGLang API 的用法。(1) fork创建多个并行的提示副本。(2) gen调用 LLM 生成并将结果存储在变量中。(3) `[variable_name]`获取生成结果。(4) choices对生成施加约束。(5) run执行带有参数的 SGLang 函数。



```python
@function
def expand(s, tip):
    s += (
        "Please expand the following tip into a "
        "detailed paragraph: " + tip + "\n"
    )
    s += s.gen("paragraph")

@function
def tip_suggestion(s, topic):
    s += "Here are 2 concise tips for " + topic + ".\n"

    # Generate a skeleton of two short tips
    s += "1." + s.gen("tip_1", stop=["\n", ":", "."]) + "\n"
    s += "2." + s.gen("tip_2", stop=["\n", ":", "."]) + "\n"

    # Expand the tips into detailed paragraphs in parallel
    detailed_tip1 = expand(tip=s["tip_1"])
    detailed_tip2 = expand(tip=s["tip_2"])

    # Merge the results and generate a summary
    s += "Tip 1: " + detailed_tip1["paragraph"] + "\n"
    s += "Tip 2: " + detailed_tip2["paragraph"] + "\n"
    s += "In summary" + s.gen("summary")

state = tip_suggestion.run(topic="staying healthy")
print(state.text())
```
该程序使用思维框架提示方法，针对给定主题并行生成两条提示 。它使用了两个函数：`expand`，用于将简短提示扩展成详细段落；以及`tip_suggestion`，用于协调整个过程。它首先生成两条简短提示，然后调用`expand`函数并行扩展它们。最后，它生成一个摘要。

这些示例展示了 SGLang 的用法。SGLang 提供了以下基本函数： `gen`用于调用 LLM 生成， `select`用于让 LLM 从列表中选择概率最高的选项， `+=`或`extend`用于扩展当前提示符， `fork`用于创建当前提示符状态的分支，以及 `join`用于重新连接已分支的提示符状态。用户可以将这些基本函数与任意 Python 控制流和库结合使用。

## 3.2 Interpreter

执行 SGLang 程序最基本的方式是通过解释器。在 SGLang 中，提示符被视为一个异步流。在提示符上执行诸如 extend、gen 和 select 之类的原语操作，会导致解释器将这些原语提交到该流中进行异步执行。这些提交调用是非阻塞的，因此 Python 解释器可以继续执行 Python 代码，而无需等待 LLM 生成完成。这个过程类似于异步地向 CUDA 流启动 CUDA 内核。每个提示符都由后台线程中的流执行器管理，从而使用户能够实现并行处理。

当从提示符获取生成结果时，获取操作将被阻塞，直到所需的生成结果准备就绪。这是通过**事件机制实现的**，其中生成调用会在产生输出时设置一个事件。因此，解释器有效地管理了程序内部的并行性和同步性。

## 3.3 Compiler

运行 SGLang 程序的另一种方法是将其编译为计算图，并使用图执行器运行。这为更多编译优化提供了机会，因为我们可以重写计算图。

## 3.3.1 Tracing and Graph Executor

我们为 SGLang 设计了一种中间表示 (IR)，将 SGLang 程序结构和操作表示为一个计算图。该图包含代表基本运算符的节点和代表依赖关系的边。在该程序中，每次调用带**装饰器**的函数decorated function或 fork 操作都会创建一个新的提示状态，即一个流。prompt state, or a stream.

节点有多种类型。在 SGLang 程序中， +=和+的每个操作数都由一个 IR 节点表示。这些节点包括ConstantText、Argument、Gen、Select、Variable、Fork、GetForkItem和Join。依赖关系有两种类型。第一种是流内依赖，即使用+=提交到流中的操作必须在该流中所有先前的操作之后执行。第二种是流间依赖，当一个流需要从另一个流中获取变量值时，就会出现流间依赖，这就需要进行同步。像 fork 这样的操作会操作多个流，因此会引入流间依赖。

为了生成图，我们使用追踪技术运行带有抽象参数的程序，并动态构建图。这种方法仅限于不涉及数据依赖控制流的程序，我们计划在未来的工作中解决这一限制。构建完成后，可以通过图执行器执行图，无需重新解释原始 Python 程序。这带来了诸多好处，例如图重写优化、降低运行时开销以及程序序列化。执行时，会为每个数据流启动流执行器，并按拓扑顺序将中间表示 (IR) 节点分派到各个数据流。

## 3.3.2 编译优化

**1. 代码移动以改进前缀共享**。**Code Movement for Improving Prefix Sharing** 此优化旨在通过重新排列图中的节点来增加常量前缀的长度，从而提高前缀共享效率。
它并不严格保留原始计算，因此被归类为一种激进的优化。
例如，将提示从“这里有一个问题 + {question}。请扮演数学专家并解答给定的问题”**更改为**“*请扮演数学专家并解答给定的问题。这里有一个问题 + {question}。*”
 For instance, changing the prompt from “Here is a question + {question}. Please act as a math expert and solve the given question” to “Please act as a math expert and solve the given question. Here is a question + {question}.” 

即可获得更长的可共享前缀。此优化很有意思，因为传统的程序分析方法由于 SGLang 中存在自然语言指令而无法实现。因此，我们引导 GPT-4 重新排列图节点。我们编写了一个包含多个示例的提示，以教授 GPT-4 SGLang 信息表示的概念，并且我们发现 GPT-4 可以成功地将此优化应用于一些简单的 SGLang 程序。



**2. Prefetching Annotations** **预取注解**。 这涉及自动向图中添加预取节点，并在缓存存在且提示信息较长时，向运行时发出信号，将缓存的键值缓存从 CPU 移动到 GPU。这可以降低包含较长提示信息的流水线的延迟。

---
# 4. Runtime with RadixAttention

优化 SGLang 程序执行的主要挑战在于跨多个调用和程序重用键值缓存，本节介绍 RadixAttention，一种用于运行时自动重用键值缓存的新技术。与生成请求完成后丢弃键值缓存的传统方法不同，我们的方法将提示信息和生成结果的键值缓存保留在一个基数树中。这种结构能够高效地进行前缀搜索、重用、插入和驱逐操作。我们实现了最近最少使用 (LRU) 驱逐策略，并辅以缓存感知调度策略，以提高缓存命中率。此外，RadixAttention 还兼容现有的连续批处理和分页注意力机制等技术。

## 4.1 RadixAttention

A radix tree基数树是一种数据结构，它是字典树（前缀树）trie (prefix tree)的一种高效节省空间的替代方案。与典型的树不同，基数树的边不仅可以标记单个元素，还可以标记长度不一的元素序列。这一特性显著提高了基数树的效率。在我们的系统中，我们利用基数树来管理映射。该映射存在于作为键的标记序列和它们对应的KV缓存张量（作为值）之间。这些KV缓存张量存储在非连续的分页布局中，每个页面的大小等于一个标记。

考虑到GPU内存容量有限，我们无法无限地重新训练键值缓存张量，因此需要一种驱逐策略。为了解决这个问题，我们实现了一种基于最近最少使用（LRU）的驱逐策略，该策略递归地驱逐叶子节点。在连续批处理场景下，我们不能驱逐当前正在运行的批处理作业所使用的节点。因此，每个节点都维护一个引用计数器，用于指示有多少个正在运行的请求正在使用它。如果节点的引用计数器为零，则该节点可以被驱逐。

前端始终向运行时发送完整的请求，运行时会自动进行前缀匹配、重用和缓存。树结构存储在 CPU 上，维护开销很小。

## 4.2 Cache-Aware Scheduling


$$\begin{array}{l} \hline \textbf{Algorithm 1 } \text{Cache-Aware Scheduling for RadixAttention} \\ \text{with Continuous Batching.} \\ \hline \textbf{Input:} \text{ The radix tree } T, \text{ the memory pool } P, \text{ the current} \\ \qquad\quad\, \text{running batch } B, \text{ the waiting queue } Q. \\ \textbf{Output:} \text{ Finished requests and updated system state.} \\ \text{// Get all requests from the waiting queue} \\ \mathit{requests} \leftarrow Q.\text{get\_all\_requests}() \\ \text{// Search for the prefix matching for all waiting requests} \\ \textbf{for } \mathit{req} \in \mathit{requests} \textbf{ do} \\ \qquad \mathit{req.prefix\_node}, \mathit{req.prefix\_len} \leftarrow T.\text{match\_prefix}(\mathit{req.input\_tokens}) \\ \text{// Sort the requests according to the matched prefix lenghts} \\ \mathit{requests}.\text{sort}() \\ \text{// Select requests for the next batch} \\ \mathit{available\_size} \leftarrow T.\text{evictable\_size}() + P.\text{available\_size}() \\ \mathit{current\_size} \leftarrow 0 \\ \mathit{new\_batch} \leftarrow [] \\ \textbf{for } \mathit{req} \in \mathit{requests} \textbf{ do} \\ \qquad \textbf{if } \mathit{req}.\text{size}() + \mathit{current\_size} < \mathit{available\_size} \textbf{ then} \\ \qquad\qquad \mathit{new\_batch}.\text{append}(\mathit{req}) \\ \qquad\qquad \mathit{delta} \leftarrow T.\text{increase\_ref\_counter}(\mathit{req.prefix\_node}) \\ \qquad\qquad \mathit{available\_size} \leftarrow \mathit{available\_size} + \mathit{delta} \\ Q.\text{remove\_requests}(\mathit{new\_batch}) \\ \text{// Insert requests into the current running batch} \\ B.\text{merge}(\mathit{new\_batch}) \\ \text{// Allocate new memory and do eviction if necessary} \\ \mathit{needed\_size} \leftarrow B.\text{needed\_size}() \\ \mathit{success}, \mathit{buffer} \leftarrow P.\text{alloc}(\mathit{needed\_size}) \\ \textbf{if not } \mathit{success} \textbf{ then} \\ \qquad T.\text{evict}(\mathit{needed\_size}) \\ \qquad \mathit{success}, \mathit{buffer} \leftarrow P.\text{alloc}(\mathit{needed\_size}) \\ B.\text{run}(\mathit{buffer}) \\ \text{// Process finished requests} \\ \mathit{finished\_requests} \leftarrow B.\text{drop\_finished\_requests}() \\ \textbf{for } \mathit{req} \in \mathit{finished\_requests} \textbf{ do} \\ \qquad T.\text{decrease\_ref\_counter}(\mathit{req.prefix\_node}) \\ \qquad T.\text{insert}(\mathit{req}) \\ \textbf{return } \mathit{finished\_requests} \\ \hline \end{array}$$

其中RadixTree T， 保存之前所有请求的kv cache，节点包含前缀token，边表示前缀匹配关系，且维护了引用计数ref_counter
P 物理显存池，负责分配物理Page/Buffer
Q 等待被处理的新请求队列。
B 当前正在 GPU 上运行（Decoding 或 Prefill）的 Request 批次。

```
for req in requests:
    req.prefix_node, req.prefix_len ← T.match_prefix(req.input_tokens)
requests.sort()
```
- 遍历等待队列中的每个请求，拿它的 Prompt 去 T里比对，看能命中多长的已缓存前缀（prefix_len），并记下命中的树节点（prefix_node）
```
available_size ← T.evictable_size() + P.available_size()
for req in requests:
    if req.size() + current_size < available_size:
        new_batch.append(req)
        delta ← T.increase_ref_counter(req.prefix_node)
        available_size ← available_size + delta
```
- 可用显存计算：可用空间 = 显存池剩余空间P+树上可被淘汰（ref_counter == 0）的节点空间T
- 锁定节点（Ref Counter）当决定将 req 选入 new_batch 时，立刻调用 increase_ref_counter 增加其命中前缀节点的引用计数。
- delta：因为增加了引用计数后，该节点被锁定（保护不被清除），但如果该节点之前属于可淘汰状态，锁住它相当于减少了可淘汰空间；相反，如果节点状态变化带来显存收益，这里会更新 available_size 确保显存预估准确。常见情况下delta小于0，起到扣显存作用
```
needed_size ← B.needed_size()
success, buffer ← P.alloc(needed_size)
if not success:
    T.evict(needed_size)
    success, buffer ← P.alloc(needed_size)
B.run(buffer)
```
- 动态合并：将 new_batch 合并进主批次 B 中。
- 尝试分配：向显存池P申请当前 Batch 所需的显存空间（用于存新生成 Token 的 KV Cache）
- 强制淘汰（Eviction）：如果显存不足（not success），树 会按照 LRU（最近最少使用）策略，只淘汰那些 ref_counter == 0 的未锁节点，腾出足够的空间后再重新分配显存。
- 执行：拿到显存 buffer 后，启动 GPU 计算 B.run(buffer)。
```
finished_requests ← B.drop_finished_requests()
for req in finished_requests do
    T.decrease_ref_counter(req.prefix_node)
    T.insert(req)
```
- 解锁：对于运行结束的请求，减少其前缀节点的引用计数（decrease_ref_counter）。这样当后续显存紧张时，这些节点就可以被安全淘汰。
- 插入新 Cache：将本次请求生成的新 Token 和新的 KV Cache 节点插入到 RadixTree T 中，供未来的请求复用。
## 4.3 其他优化

开发新的 CUDA 内核对于提高键值缓存重用的效率至关重要。现有系统大多实现用于两个主要功能的 CUDA 内核：预填充和解码。预填充内核假设查询字符串和键的长度相同。相反，解码内核假设查询字符串的长度为 1。然而，在键值缓存重用的情况下，这些内核无法直接应用。我们为此引入了一种新的“扩展”内核，它能够从连续缓冲区和非连续内存池中读取键值缓存。这种融合的扩展内核无需额外的复制开销即可运行。

---
# 5. Evaluation

SGLang 和 SGVM 的实现大约用了 9000 行 Python 代码。SGVM 重用了 LightLLM 中的一些组件 。一个基于 PyTorch 的 LLM 服务系统和Triton中的部分核心组件。

LightLLM是由商汤科技等机构开源的大语言模型（LLM）推理服务框架。它的特点是采用了纯 Python/PyTorch 的轻量化设计，并首创了 Token 级的智能内存管理（Router），在处理超长文本和高并发请求时效率极高。

Triton是由 OpenAI 开源的一种编程语言和编译器。它允许开发者用类似于 Python 的简单语法，写出运行在 NVIDIA GPU 上、性能堪比原生 CUDA（C/C++）的高性能算子（硬件加速代码）。传统的 PyTorch 在某些特定计算（如 Attention 机制、定制的激活函数）上可能不够快。LightLLM 和 SGVM 通过使用 Triton，编写了高度优化的 GPU 计算核心。这使得系统既保留了 PyTorch 的开发灵活性，又获得了极度接近硬件极限的运行推理速度。
*Triton: an intermediate language and compiler for tiled neural network computations.*
*In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19, 2019.*

性能指标。 我们报告两个性能指标：吞吐量和延迟。对于吞吐量，我们运行足够大的程序实例批次来计算最大吞吐量，并比较每秒执行的程序实例数（每秒程序数，p/s）。对于延迟，我们每次执行单个程序，不进行批处理，并报告多个实例的平均延迟。除非另有说明，我们的大多数优化不会改变计算结果。

## 5.1 基于LLM的Agent
**ReAct agent** *React: Synergizing reasoning and acting in language models.*
*In The Eleventh International Conference on Learning Representations, 2022.*
它通过将多个 LLM 调用与诸如网络搜索之类的操作链接起来，以交错的方式处理任务
**generative agents**
*Generative agents: Interactive simulacra of human behavior.*
*In In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST ’23), UIST ’23, New York, NY, USA, 2023. Association for Computing Machinery.*
应用于游戏环境中，其中智能体通过调用 LLM 相互交互并做出决策。

## 5.2 推理
这些任务涉及在零样本环境下解决 GSM-8K 问题
Multi-chain reasoning该方法涉及针对同一问题生成多个解决方案，并利用这些解决方案进行元推理以获得最终答案。此过程允许共享问题并并行生成多个解决方案。
Tree-of-thought 该方法将逻辑逻辑模型（LLM ）集成到广度优先树搜索算法中。LLM 必须提出分支探索的下一步策略，执行这些策略，并评估解决方案。这种方法能够处理现有系统无法处理的复杂共享模式。在此基准测试中，SGLang 再次凭借更高的吞吐量和更低的延迟超越了其他语言。

## 5.3 消融
尽管 RadixAttention 可以自动检测共享前缀，但我们的运行时实现较为简化：我们仅将待处理的请求与现有的缓存树进行比较。我们不会为等待队列中的所有请求构建树索引。当存在大量共享前缀但尚未处理并插入树中的待处理请求时，这种方法可能会导致性能欠佳。为了缓解这个问题，我们使用解释器interpreter来辅助调度。当解释器识别出一个极有可能共享的前缀（例如，在 fork 过程中），它会首先发出一个针对该前缀的转发请求并将其存储在树中，然后再发出可以重用该前缀的请求。这种方法利用解释器的提示信息来减轻后端调度的负载，凸显了我们协同设计策略的优势。

“禁用扩展内核”禁用了我们优化的内核，导致性能下降，尤其是在长上下文任务中。“禁用解释器并行”关闭了程序内并行，导致并行性和缓存局部性降低。“禁用缓存”禁用了缓存，显著降低了所有基准测试的吞吐量。“禁用基数树”使用列表而不是树来管理缓存，导致内存使用量和复制开销增加，从而降低了性能。

我们分析了运行时维护基数树的开销。树结构在 CPU 上维护。在一个典型场景中，共享输入长度为 256，不可共享输入长度也为 256，共有 128 个请求（分为 16 个子树，每个子树有 8 个分支），我们设置中的前向传播耗时 17.6 秒。同时，树操作仅耗时 0.07 秒。我们还构建了一个没有缓存命中的请求跟踪，观察到树维护仅产生 0.5% 的开销。因此，该开销可以忽略不计。此外，通过使用高性能的 C++ 实现以及 CPU 和 GPU 操作的重叠，可以进一步降低树维护的开销。

# Remark1 : RadixTree 

基数树（Radix Tree，也称压缩前缀树）是一种空间优化型的标准前缀树（Trie）数据结构。它通过将只有一个子节点的父节点与子节点进行合并，减少了树的层数和节点数量，从而显著降低了内存开销并提升了检索效率。

- 合并单子节点：与普通前缀树（每个节点只存一个字符）不同，基数树的边（Edges）可以包含字符或位序列。
- 时间复杂度：查找（Lookup）、插入（Insertion）和删除（Deletion）操作的时间复杂度为 O(k)，其中 k 是键的最大长度（与树中包含的键的总数无关）。
- 基数（Radix）：内部节点的最大子节点数量由基数$r=2^x(x \ge 1)$ 决定。当 r=2 时，它被称为二进制基数树，也被称为 PATRICIA Trie

边（Edge）是存储实际键（Key）片段的载体
- 标准 Trie 的边：仅仅是一个指针，指向下一个字符节点（边本身不携带复杂的文本信息，默认代表单个字符）。
- 基数树的边：是一个标签（Label）加一个指针（Pointer）的组合。标签可以是任意长度的字符串（如 `"apple"、"inf"`）或二进制位序列（如 `10110`）。

 **边在插入时的“分裂”机制（Split）**,基数树的边不是静态的，它会随着新数据的插入动态分裂。

场景：已有键 "tester"，现在插入新键 "test" 和 "team"

`[Root] -- "tester" --> [Node 1 (IsEnd=true)]`

插入 "test"（"test" 是 "tester" 的前缀）：边 "tester" 被拆分为公共前缀 "test" 和剩余部分 "er" 结果：产生一个中间节点。

`[Root] -- "test" --> [Node 2 (IsEnd=true)] -- "er" --> [Node 1 (IsEnd=true)]`

插入 "team"（"team" 与 "test" 有公共前缀 "te"）：边 "test" 被进一步拆分为公共前缀 "te" 和剩余部分 "st"。

```
                  -- "st" --> [Node 2 (IsEnd=true)] -- "er" --> [Node 1 (IsEnd=true)]
                 |
[Root] -- "te" --+
              |
               -- "am" --> [Node 3 (IsEnd=true)]

```

边在删除时的“合并”机制（Merge）当删除某个键导致某个内部节点只剩下一个子节点时，这条边就会与子节点的边发生合并（压缩）。
场景：承接上图，如果删除了 "team"
`[Node 3]` 被销毁 此时 `[Root]` 出来的 "te" 分支后面，只剩下一个子分支（即通往 Node 2 的 "st"）。
算法会将边 "te" 和边 "st" 自动合并为一条新边 "test"。
树的结构恢复到插入 "team" 之前的状态

边在底层的物理实现方式
在实际编程中（如 C、Go 或 Rust），为了节省内存，边上的字符串通常不会真正拷贝一份。
- 指针/切片引用：如果所有的键存储在一个巨大的缓冲区或字符串数组中，边（Edge）往往只存储一个切片或索引范围（例如 start_index 和 end_index），通过指针直接引用原始数据。
- 联合体/指针低位利用：在 C 语言实现中（如 Linux 内核），如果基数树是二进制的（Radix=2），边往往直接映射为指针的位操作，通过掩码（Mask）来判断走左边（0）还是右边（1）。

**标准前缀树（Trie，也称字典树、单词查找树）**

是一种用于快速检索字符串键的多叉树数据结构。它的核心思想是利用字符串的公共前缀来减少查询时间、节省存储空间（虽然在稀疏数据下并不省空间），从而实现高效的字符串匹配。

核心特征
- 节点不存完整单词：除了根节点（通常为空）外，树中的每个节点通常只存储一个字符（或一个固定的字符块）。
- 路径决定键的值：从根节点到某一个节点，路径上经过的字符连接起来，就是该节点对应的字符串。
- 标记单词结尾：节点中通常包含一个布尔值（如 isEnd），用来标识从根到当前节点的路径是否构成了一个完整的单词，而不仅仅是其他单词的前缀。
**时间与空间复杂度**
- 时间复杂度：O(k) 无论是插入、查找还是删除，时间复杂度只取决于被检索字符串的长度 k，而与树中已知单词的总数量（N）无关。这使得它在处理海量数据时比哈希表更稳定（哈希表可能存在哈希冲突，最坏情况下会退化）。

- 空间复杂度：O(Alphabet_Size * N * k)这是标准前缀树的最大短板。因为每个节点都需要为所有可能的后续字符（例如英文字母表的 26 个字母）预留指针空间。如果树很稀疏（很多节点只有一个子节点），会产生大量的空指针（Null Pointers），造成严重的内存浪费。

**典型应用场景**
- 搜索框自动补全（Autocomplete）：当你在 Google 搜索框输入 "dat" 时，系统能瞬间列出 "data"、"database"，就是利用了前缀树对相同前缀的快速遍历。
- 拼写检查（Spell Checker）：快速判断一个单词是否存在于词典中。
- IP 路由表的最长前缀匹配（LPM）：在网络中用于决定数据包的转发路径。




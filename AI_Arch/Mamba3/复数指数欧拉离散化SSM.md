
**Proposition2**  **Complex-to-Real SSM Equivalence** 注：Re表示取复数的实部 Im表示虚部

Consider a complex-valued SSM$$\begin{align}
    \dot{h}(t) &= \mathrm{Diag}\big(A(t) + i \boldsymbol{\theta}(t)\big)\,h(t) 
    + \big(B(t) + i\hat{B}(t)\big)\,x(t), \\ \nonumber
    y(t) &= \mathrm{Re}\!\left(
        \big(C(t) + i\hat{C}(t)\big)^{\top}h(t)
    \right),
\end{align}$$
where $h(t) \in \mathbb{C}^{N/2}$, $\boldsymbol{\theta}(t),B(t),\hat{B}(t),C(t),\hat{C}(t)\in\mathbb{R}^{N/2}$, and $x(t),A(t)\in\mathbb{R}$. Under exponential-Euler discretization, this system is equivalent to a real-valued SSM$$\begin{align}
    h_t &= e^{\Delta_t A_t}\,R_t\,h_{t-1} + \Delta_t B_t x_t, \\  \nonumber
    y_t   &= C_t^{\top}h_t,
\end{align}$$
---
## 注
$z=u+iv$ u v可以用三角函数表示
$z=r\cos \phi +i(r\sin \phi)$
$z=r(\cos \phi +i\sin \phi )$
使用欧拉公式（Euler's Formula）改写
$e^{i\phi }=\cos \phi +i\sin \phi$
$z=re^{i\phi }$
---

with state $h_t \in \mathbb{R}^N$, projections$$
B_t := \begin{bmatrix} B_t \\ \hat{B}_t \end{bmatrix} \in \mathbb{R}^N, 
\qquad 
C_t := \begin{bmatrix} C_t \\ -\hat{C}_t \end{bmatrix} \in \mathbb{R}^N, 
$$
and a transition matrix$$R_t \;:= \text{Block}\Big(\{R(\Delta_t \boldsymbol{\theta_t}[i])\}_{i=1}^{N/2}\Big)
\in \mathbb{R}^{N \times N}
,
\qquad 
R(\theta) := 
\begin{bmatrix}
    \cos(\theta) & -\sin(\theta) \\
    \sin(\theta) & \cos(\theta)
\end{bmatrix}.$$
**Proof.**
We first present the derivation for $N=2$; the block-diagonal structure for general even $N$ follows by grouping pairs of coordinates.

Let $h_t + i\hat{h}_t$ denote the complexified 复数化 hidden state, with parameters $A(t)+i\theta(t)$ and $B(t)+i\hat{B}(t)$ for the transition and input, respectively. By the variation of constants formula, applying zero-order hold and Euler's rule over a step $[t_{k-1},t_k]$ gives$$h_k + i\hat{h}_k 
= e^{\Delta_t (A_t + i\theta_t)}(h_{k-1}+ i\hat{h}_{k-1})
+ \Delta_t (B_t + i\hat{B}_t)x_t.$$
---
## 注 中文版再推导

在 $N=2$ 时，一对实数状态 $(h_t, \hat{h}_t) \in \mathbb{R}^2$ 可以合并成一个一维**复数隐状态** $z(t) = h_t + i\hat{h}_t \in \mathbb{C}$。 同样地，把状态转移参数（实部控制衰减 $A(t)$，虚部控制旋转角速度 $\theta(t)$）和输入投影参数 $B(t) + i\hat{B}(t)$ 复数化，连续时间状态空间方程写为：
$$\frac{dz(t)}{dt} = \big(A(t) + i\theta(t)\big) z(t) + \big(B(t) + i\hat{B}(t)\big) x(t)$$

---
## 注注
**在二维实数空间 $\mathbb{R}^2$ 中，$A$ 代表的是一个 $2\times 2$ 矩阵（线性算子），而 $B$ 代表的是一个 $2\times 1$ 列向量（状态坐标）**。两者的代数角色和自由度完全不同。


当标量输入 $x(t) \in \mathbb{R}$ 进入 $N=2$ 的实数系统时，它需要分别向 $h_t$ 和 $\hat{h}_t$ 两个维度投射，因此需要一个二维列向量：$$\mathbf{B}(t) = \begin{bmatrix} B(t) \\ \hat{B}(t) \end{bmatrix} \in \mathbb{R}^{2\times 1}$$这里的 $B(t)$ 和 $\hat{B}(t)$ 分别是输入在实轴和虚轴方向上的分量，二者地位完全对等、相互独立，所以直接组合成复数 $B(t) + i\hat{B}(t)$。

如果状态转移也像 $B$ 那样直接放两个独立的实参数（即对角矩阵 $\begin{bmatrix} A & 0 \\ 0 & \hat{A} \end{bmatrix}$），那么两个维度之间没有任何耦合，只能各自独立指数衰减，**无法产生旋转**（**这也是纯实数对角 SSM 无法追踪奇偶性/周期状态的原因**）。
要让二维实向量 $\begin{bmatrix} h_t \\ \hat{h}_t \end{bmatrix}$ 既能缩放又能旋转，其对应的 $2\times 2$ 实矩阵必须是**反对称耦合结构**（共形矩阵）：“反对称耦合结构”指的是矩阵非对角线元素呈大小相等、符号相反的交叉连接方式。
$$\mathbf{A}(t) = \begin{bmatrix} A(t) & -\theta(t) \\ \theta(t) & A(t) \end{bmatrix} \in \mathbb{R}^{2\times 2}$$
验证它作用在 $\mathbf{h}(t)$ 上的结果：
$$\begin{bmatrix} A(t) & -\theta(t) \\ \theta(t) & A(t) \end{bmatrix} \begin{bmatrix} h_t \\ \hat{h}_t \end{bmatrix} = \begin{bmatrix} A(t)h_t - \theta(t)\hat{h}_t \\ \theta(t)h_t + A(t)\hat{h}_t \end{bmatrix}$$
而这恰好严格等于复数乘法 $(A(t) + i\theta(t))(h_t + i\hat{h}_t)$ 展开后的实部与虚部：
$$(A + i\theta)(h + i\hat{h}) = (Ah - \theta\hat{h}) + i(\theta h + A\hat{h})$$
---

## 第二步：使用常数变易法（Variation of Constants Formula）求解析解

对于一阶线性非齐次微分方程 $z'(t) = \alpha(t) z(t) + \beta(t) x(t)$，在区间 $[t_{k-1}, t_k]$ 上（设步长为 $\Delta_t = t_k - t_{k-1}$），根据常微分方程的**常数变易法**：(其中非齐次代表有外部输入，常数变易法是先假设没有外部输入（把方程变成齐次的）求出基础解，然后把解里面的“任意常数”变成“随时间变化的函数”，再代回原方程去反推这个函数。)
$$z(t_k) = e^{\int_{t_{k-1}}^{t_k} (A(\tau) + i\theta(\tau)) d\tau} z(t_{k-1}) + \int_{t_{k-1}}^{t_k} e^{\int_{s}^{t_k} (A(\tau) + i\theta(\tau)) d\tau} \big(B(s) + i\hat{B}(s)\big) x(s)\, ds$$


----
## 注注 常数变易法推导
为了清晰展示推导过程，先将原方程记为标准的一阶线性非齐次微分方程形式：
$$z'(t) = \alpha(t)z(t) + u(t)$$

其中，状态转移系数 $\alpha(t) = A(t) + i\theta(t)$，输入驱动项 $u(t) = \big(B(t) + i\hat{B}(t)\big)x(t)$。
$$z'(t) = \alpha(t)z(t)$$
$$\frac{dz(t)}{z(t)} = \alpha(t)\,dt$$

从区间起点 $t_{k-1}$ 到任意时刻 $t$ 两边定积分（用 $\tau$ 作为积分哑变量以区分上限 $t$）：
$$\int_{t_{k-1}}^{t} \frac{dz(\tau)}{z(\tau)} = \int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau \implies \ln\frac{z(t)}{z(t_{k-1})} = \int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau$$
两边取指数，得到齐次方程的通解（其中 $C$ 为任意常数）：
$$z(t) = C \cdot e^{\int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau}$$
现在恢复外部输入 $u(t)$。设非齐次方程的特解形式与齐次通解一致，但把常数 $C$ 变成随时间变化的待定函数 $C(t)$：（**齐次通解包含任意常数，反映系统自身的“固有规律”；非齐次特解不含任意常数，反映“外部输入”对系统产生的特定影响。**）
$$z(t) = C(t) e^{\int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau} \tag{1}$$
对该式两边关于 $t$ 求导（使用乘法法则和变上限积分求导法则 $\frac{d}{dt}\int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau = \alpha(t)$）：
$$z'(t) = C'(t) e^{\int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau} + C(t) \alpha(t) e^{\int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau}$$
注意到后半部分中的 $C(t) e^{\int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau}$ 正是 $z(t)$ 本身，因此：
$$z'(t) = C'(t) e^{\int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau} + \alpha(t)z(t)$$

将上式代入原非齐次方程 $z'(t) = \alpha(t)z(t) + u(t)$：
$$C'(t) e^{\int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau} + \alpha(t)z(t) = \alpha(t)z(t) + u(t)$$

两边消去 $\alpha(t)z(t)$，解出 $C'(t)$：
$$C'(t) = e^{-\int_{t_{k-1}}^{t} \alpha(\tau)\,d\tau} u(t)$$

在区间 $[t_{k-1}, t_k]$ 上对 $t$（改记积分变量为 $s$）定积分：
$$C(t_k) - C(t_{k-1}) = \int_{t_{k-1}}^{t_k} e^{-\int_{t_{k-1}}^{s} \alpha(\tau)\,d\tau} u(s)\,ds$$

确定初值 $C(t_{k-1})$：将 $t = t_{k-1}$ 代入 $(1)$ 式，由于 $\int_{t_{k-1}}^{t_{k-1}} \alpha(\tau)\,d\tau = 0$，得 $z(t_{k-1}) = C(t_{k-1}) \cdot e^0 = C(t_{k-1})$。因此：
$$C(t_k) = z(t_{k-1}) + \int_{t_{k-1}}^{t_k} e^{-\int_{t_{k-1}}^{s} \alpha(\tau)\,d\tau} u(s)\,ds$$
将 $t = t_k$ 和 $C(t_k)$ 代回 $(1)$ 式：
$$z(t_k) = e^{\int_{t_{k-1}}^{t_k} \alpha(\tau)\,d\tau} z(t_{k-1}) + e^{\int_{t_{k-1}}^{t_k} \alpha(\tau)\,d\tau} \int_{t_{k-1}}^{t_k} e^{-\int_{t_{k-1}}^{s} \alpha(\tau)\,d\tau} u(s)\,ds$$

把积分号外面的指数项乘进积分号内（与积分变量 $s$ 无关的因子可直接移入），利用积分区间的可加性：
$$\int_{t_{k-1}}^{t_k} \alpha(\tau)\,d\tau - \int_{t_{k-1}}^{s} \alpha(\tau)\,d\tau = \int_{s}^{t_k} \alpha(\tau)\,d\tau$$

最终得到：
$$z(t_k) = e^{\int_{t_{k-1}}^{t_k} \alpha(\tau)\,d\tau} z(t_{k-1}) + \int_{t_{k-1}}^{t_k} e^{\int_{s}^{t_k} \alpha(\tau)\,d\tau} u(s)\,ds$$

最后把 $\alpha(\tau) = A(\tau) + i\theta(\tau)$ 和 $u(s) = \big(B(s) + i\hat{B}(s)\big)x(s)$ 替换回去，就得到了笔记中的公式：

$$z(t_k) = e^{\int_{t_{k-1}}^{t_k} (A(\tau)+i\theta(\tau))\,d\tau} z(t_{k-1}) + \int_{t_{k-1}}^{t_k} e^{\int_{s}^{t_k} (A(\tau)+i\theta(\tau))\,d\tau} \big(B(s) + i\hat{B}(s)\big)x(s)\,ds$$

公式右边的第一项是上一时刻状态 $z(t_{k-1})$ 在整个步长 $[t_{k-1}, t_k]$ 内经历衰减与旋转后的剩余量；第二项则是在任意中间时刻 $s$ 注入的瞬时输入，从 $s$ 时刻开始直到 $t_k$ 时刻为止经历剩余时间段 $[s, t_k]$ 衰减与旋转后的累积贡献。

-----


**零阶保持（ZOH）** 假设在单个时间步 $[t_{k-1}, t_k]$ 内，所有连续参数和输入信号都保持为常数（取该步的离散值）：
$$A(\tau) \equiv A_t, \quad \theta(\tau) \equiv \theta_t, \quad B(s) \equiv B_t, \quad \hat{B}(s) \equiv \hat{B}_t, \quad x(s) \equiv x_t \quad (\forall \tau, s \in [t_{k-1}, t_k])$$

_(注：这里下标 $t$ 和 $k$ 混用是原论文符号习惯，$k$ 表示当前离散步索引，$\Delta_t, A_t, \theta_t, B_t, \hat{B}_t, x_t$ 表示第 $k$ 步对应的步长与参数。)_

  
把常数提出积分号：

1. **齐次项（状态转移项）**：
$$\int_{t_{k-1}}^{t_k} (A_t + i\theta_t)\, d\tau = \Delta_t (A_t + i\theta_t)$$
    $$e^{\Delta_t (A_t + i\theta_t)} z(t_{k-1})$$
2. **非齐次项（输入积分项）**：$$(B_t + i\hat{B}_t) x_t \int_{t_{k-1}}^{t_k} e^{(t_k - s)(A_t + i\theta_t)}\, ds$$
## 用欧拉法（Euler's Rule）近似输入积分

如果严格计算上面的积分，会得到精确的 ZOH 离散化形式 $\bar{B}_t = \frac{e^{\Delta_t(A_t + i\theta_t)} - 1}{A_t + i\theta_t}(B_t + i\hat{B}_t)$，这样分母上会出现复数除法，计算开销较大。 

因此，作者对第二项的积分在区间 $[t_{k-1}, t_k]$ 上直接应用**前向/左端点矩形积分法则（Euler's rule）**——即直接用被积函数在区间端点（或一阶泰勒展开 $e^{(t_k - s)(\dots)} \approx 1$）的值乘以区间长度 $\Delta_t$：
$$\int_{t_{k-1}}^{t_k} e^{(t_k - s)(A_t + i\theta_t)}\, ds \approx \Delta_t \cdot 1 = \Delta_t$$
于是第二项就简化为：
$$\Delta_t (B_t + i\hat{B}_t) x_t$$
_(这也正是 Mamba-1 / Mamba-2 中著名的简化技巧：$A$ 矩阵采用精确指数离散化 $\bar{A}_t = \exp(\Delta_t A_t)$ **保证稳定性和旋转相位精确**，而 $B$ 矩阵采用欧拉近似 $\bar{B}_t = \Delta_t B_t$ 避免矩阵求逆或复数除法。)_

将两项合并，并把 $z(t_k) = h_k + i\hat{h}_k$ 和 $z(t_{k-1}) = h_{k-1} + i\hat{h}_{k-1}$ 代回，即得截图中的递推公式：
$$h_k + i\hat{h}_k = e^{\Delta_t (A_t + i\theta_t)}(h_{k-1} + i\hat{h}_{k-1}) + \Delta_t (B_t + i\hat{B}_t) x_t $$
## 补充：为什么这一步对解决前文说的“状态追踪问题”很关键？

利用欧拉公式展开转移因子：

$$e^{\Delta_t (A_t + i\theta_t)} = e^{\Delta_t A_t} \big(\cos(\Delta_t \theta_t) + i\sin(\Delta_t \theta_t)\big)$$

把实部和虚部分开写成 $2\times 2$ 实矩阵作用在 $\begin{bmatrix} h_{k-1} \\ \hat{h}_{k-1} \end{bmatrix}$ 上，它就是**缩放（$e^{\Delta_t A_t}$）$\times$ 二维旋转矩阵（$R(\Delta_t \theta_t)$）**。这样既保留了全实数运算的硬件友好性，又打破了“非负实数特征值只能衰减不能旋转”的限制，从而能够表达 $\mathbb{Z}_p$ 循环群等状态追踪任务。

---

Expanding the exponential,$$e^{\Delta_t(A_t+i\theta_t)} 
= e^{\Delta_t A_t}\Big(\cos(\Delta_t\theta_t)+i\sin(\Delta_t\theta_t)\Big),$$
so in real coordinates $h_t=\begin{bmatrix}h_t \\ \hat{h}_t\end{bmatrix}\in\mathbb{R}^2$ the recurrence becomes $$h_t
= e^{\Delta_t A_t}
\underbrace{\begin{bmatrix}
    \cos(\Delta_t\theta_t) & -\sin(\Delta_t\theta_t) \\
    \sin(\Delta_t\theta_t) & \cos(\Delta_t\theta_t)
\end{bmatrix}}_{R(\Delta_t\theta_t)}
h_{t-1}
+ \Delta_t
\begin{bmatrix}
B_t \\ \hat{B}_t
\end{bmatrix}x_t.$$
Stacking across $N/2$ such pairs yields the block-diagonal transition $$h_t
= e^{\Delta_t A_t}\,\mathrm{Block}\big(\{R(\Delta_t\theta_t[i])\}_{i=1}^{N/2}\big)h_{t-1}
+ \Delta_t\begin{bmatrix}B_t \\ \hat{B}_t\end{bmatrix}x_t.$$
For the output, $$y_t = \mathrm{Re}\Big((C_t+i\hat{C}_t)^\top(h_t+i\hat{h}_t)\Big)
= \begin{bmatrix}C_t \\ -\hat{C}_t\end{bmatrix}^\top h_t,$$
which defines the real projection $C_t\in\mathbb{R}^N$ in the proposition.
This proves the equivalence between complex SSM and the real block-diagonal system with rotations.

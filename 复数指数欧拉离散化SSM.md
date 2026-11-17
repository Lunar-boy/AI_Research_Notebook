
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

$$h_k + i\hat{h}_k = e^{\Delta_t (A_t + i\theta_t)}(h_{k-1} + i\hat{h}_{k-1}) + \Delta_t (B_t + i\hat{B}_t) x_t$$

是将连续时间线性时变微分方程（ODE）离散化后的结果（正是 Mamba-3 / 复数状态空间模型将 $2\times 2$ 旋转-缩放块等效为复数标量更新的核心推导）。

## 第一步：建立连续时间复数 ODE（复数化）

在 $N=2$ 时，一对实数状态 $(h_t, \hat{h}_t) \in \mathbb{R}^2$ 可以合并成一个一维**复数隐状态** $z(t) = h_t + i\hat{h}_t \in \mathbb{C}$。 同样地，把状态转移参数（实部控制衰减 $A(t)$，虚部控制旋转角速度 $\theta(t)$）和输入投影参数 $B(t) + i\hat{B}(t)$ 复数化，连续时间状态空间方程写为：

  

$$\frac{dz(t)}{dt} = \big(A(t) + i\theta(t)\big) z(t) + \big(B(t) + i\hat{B}(t)\big) x(t)$$

### 第二步：使用常数变易法（Variation of Constants Formula）求解析解

对于一阶线性非齐次微分方程 $z'(t) = \alpha(t) z(t) + \beta(t) x(t)$，在区间 $[t_{k-1}, t_k]$ 上（设步长为 $\Delta_t = t_k - t_{k-1}$），根据常微分方程的**常数变易法**，精确解为：

  

$$z(t_k) = e^{\int_{t_{k-1}}^{t_k} (A(\tau) + i\theta(\tau)) d\tau} z(t_{k-1}) + \int_{t_{k-1}}^{t_k} e^{\int_{s}^{t_k} (A(\tau) + i\theta(\tau)) d\tau} \big(B(s) + i\hat{B}(s)\big) x(s)\, ds$$

### 第三步：零阶保持（Zero-Order Hold, ZOH）近似

**零阶保持（ZOH）** 假设在单个时间步 $[t_{k-1}, t_k]$ 内，所有连续参数和输入信号都保持为常数（取该步的离散值）：

  

$$A(\tau) \equiv A_t, \quad \theta(\tau) \equiv \theta_t, \quad B(s) \equiv B_t, \quad \hat{B}(s) \equiv \hat{B}_t, \quad x(s) \equiv x_t \quad (\forall \tau, s \in [t_{k-1}, t_k])$$

_(注：这里下标 $t$ 和 $k$ 混用是原论文符号习惯，$k$ 表示当前离散步索引，$\Delta_t, A_t, \theta_t, B_t, \hat{B}_t, x_t$ 表示第 $k$ 步对应的步长与参数。)_

  

把常数提出积分号：

  

1. **齐次项（状态转移项）**：
    
      
    
    $$\int_{t_{k-1}}^{t_k} (A_t + i\theta_t)\, d\tau = \Delta_t (A_t + i\theta_t)$$
    
    所以第一项直接变成：
    
      
    
    $$e^{\Delta_t (A_t + i\theta_t)} z(t_{k-1})$$
    
2. **非齐次项（输入积分项）**：
    
      
    
    $$(B_t + i\hat{B}_t) x_t \int_{t_{k-1}}^{t_k} e^{(t_k - s)(A_t + i\theta_t)}\, ds$$
    

### 第四步：用欧拉法（Euler's Rule）近似输入积分

如果严格计算上面的积分，会得到精确的 ZOH 离散化形式 $\bar{B}_t = \frac{e^{\Delta_t(A_t + i\theta_t)} - 1}{A_t + i\theta_t}(B_t + i\hat{B}_t)$，这样分母上会出现复数除法，计算开销较大。

  

因此，作者对第二项的积分在区间 $[t_{k-1}, t_k]$ 上直接应用**前向/左端点矩形积分法则（Euler's rule）**——即直接用被积函数在区间端点（或一阶泰勒展开 $e^{(t_k - s)(\dots)} \approx 1$）的值乘以区间长度 $\Delta_t$：

  

$$\int_{t_{k-1}}^{t_k} e^{(t_k - s)(A_t + i\theta_t)}\, ds \approx \Delta_t \cdot 1 = \Delta_t$$

于是第二项就简化为：

  

$$\Delta_t (B_t + i\hat{B}_t) x_t$$

_(这也正是 Mamba-1 / Mamba-2 中著名的简化技巧：$A$ 矩阵采用精确指数离散化 $\bar{A}_t = \exp(\Delta_t A_t)$ 保证稳定性和旋转相位精确，而 $B$ 矩阵采用欧拉近似 $\bar{B}_t = \Delta_t B_t$ 避免矩阵求逆或复数除法。)_

  

### 第五步：合并得到截图公式

将两项合并，并把 $z(t_k) = h_k + i\hat{h}_k$ 和 $z(t_{k-1}) = h_{k-1} + i\hat{h}_{k-1}$ 代回，即得截图中的递推公式：

  

$$h_k + i\hat{h}_k = e^{\Delta_t (A_t + i\theta_t)}(h_{k-1} + i\hat{h}_{k-1}) + \Delta_t (B_t + i\hat{B}_t) x_t \quad \text{[cite: 1]}$$

### 补充：为什么这一步对解决前文说的“状态追踪问题”很关键？

利用欧拉公式展开转移因子：

  

$$e^{\Delta_t (A_t + i\theta_t)} = e^{\Delta_t A_t} \big(\cos(\Delta_t \theta_t) + i\sin(\Delta_t \theta_t)\big)$$

把实部和虚部分开写成 $2\times 2$ 实矩阵作用在 $\begin{bmatrix} h_{k-1} \\ \hat{h}_{k-1} \end{bmatrix}$ 上，它就是**缩放（$e^{\Delta_t A_t}$）$\times$ 二维旋转矩阵（$R(\Delta_t \theta_t)$）**。这样既保留了全实数运算的硬件友好性，又打破了“非负实数特征值只能衰减不能旋转”的限制，从而能够表达 $\mathbb{Z}_p$ 循环群等状态追踪任务。

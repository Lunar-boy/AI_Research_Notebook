**Proposition 3** consider the real SSM defined in $$\begin{align}

    h_t &= e^{\Delta_t A_t}\,R_t\,h_{t-1} + \Delta_t B_t x_t, \\  \nonumber
    y_t   &= C_t^{\top}h_t
\end{align}$$
unrolled for $T$ time-steps. The output of the above SSM is equivalent to that of a vanilla scalar transition matrix-based SSM$$
\begin{align}
    h_t \;&\approx\; e^{\Delta_t A_t}\,h_{t-1} 
        \;+\; (\tau_t - \tau_{t-1}) e^{(\tau_t-\tau_t) A_t}\,B_t\,x_t \nonumber \\
    &=\; e^{\Delta_t A_t}\,h_{t-1} 
        \;+\; \Delta_t\,B_t\,x_t.
\end{align}
$$with a data-dependent rotary旋转的 embedding applied on the $B,C$ components of the SSM, as defined by:$$\begin{equation}
    h_t = e^{\Delta_tA_t}h_{t-1} + \left( \prod_{i=0}^t R^\top_i \right) \Delta_tB_tx_t, \qquad \qquad
    y_t = \left[\left( \prod_{i=0}^t R^\top_i \right)C_t\right]^\top h_t \qquad
\end{equation}$$
where the matrix product represents right matrix multiplication, e.g., $\prod_{i=0}^1R_i = R_0 R_1$. 
We refer to the usage of a transformed real-valued SSM to compute the complex SSM as the RoPE trick.

# Proof

Consider the SSM$$\begin{equation}
    h_t \;=\; e^{\Delta_t A_t}\,R_t\,h_{t-1} \;+\; \Delta_tB_t x_t, 
    \qquad 
    y_t \;=\; C_t^\top h_t,
\end{equation}$$
where (as in Proposition 3) $A_t\in\mathbb{R}$ is a scalar (so that $e^{\Delta_t A_t}$ is a scalar and commutes with rotations), and $R_t$ is block-diagonal orthogonal/unitary, hence $R_t^{-1}=R_t^\top$ and the matrices $R_i, R_j$ commute, i.e. $R_iR_j=R_jR_i$.

Unrolling the recurrence with the convention惯例 that an empty product is the identity,空乘积（最后一项t+1为单位矩阵）$$\begin{equation}
    h_t \;=\; \sum_{i=0}^{t} 
    \bigg(\prod_{s=i+1}^{t} e^{\Delta_s A_s}R_s\bigg)\,\Delta_iB_i x_i.
\end{equation}$$
Thus$$\begin{align}
    y_t 
    \;=\; C_t^\top h_t
    \;=\; \sum_{i=0}^{t} C_t^\top 
    \bigg(\prod_{s=i+1}^{t} e^{\Delta_s A_s}R_s\bigg)\Delta_iB_i x_i.
\end{align}
$$
Using its unitary property,$$\begin{align*}
    \prod_{s=i+1}^{t}R_s 
    = \left(\prod_{s=0}^{t}R_s\right)\left(\prod_{s=0}^{i}R_s\right)^{-1}
    = \left(\prod_{s=0}^{t}R_s\right)\left(\prod_{s=0}^{i}R_s^\top\right).
\end{align*}
$$
Since $e^{\Delta_s A_s}$ are scalars, they commute with rotations; hence$$
\begin{align}
    y_t 
    &= \sum_{i=0}^{t} 
       C_t^\top \left(\prod_{s=0}^{t}R_s\right)
       \left(\prod_{s=i+1}^{t} e^{\Delta_s A_s}\right)
       \left(\prod_{s=0}^{i}R_s^\top\right) \Delta_i B_i x_i \\
    &= \left[\left(\prod_{s=0}^{t}R_s^\top\right)C_t\right]^\top 
       \sum_{i=0}^{t} 
       \left(\prod_{s=i+1}^{t} e^{\Delta_s A_s}\right)
       \left(\prod_{s=0}^{i}R_s^\top\right)\Delta_i B_i x_i.
\end{align}$$
Define the rotated parameters $\overline{C}_t := \left(\prod_{s=0}^{t}R_s^\top\right)C_t$ and $\overline{B}_i := \left(\prod_{s=0}^{i}R_s^\top\right)B_i.$
Then,$$\begin{equation}
    y_t \;=\; \overline{C}_t^\top 
    \sum_{i=0}^{t} \left(\prod_{s=i+1}^{t} e^{\Delta_s A_s}\right)\Delta_i \overline{B}_i x_i.
\end{equation}$$
Equivalently, introducing the rotated state $\tilde{h}_t := \left(\prod_{s=0}^{t}R_s^\top\right)h_t$, 
$$\begin{equation}
    \tilde{h}_t \;=\; e^{\Delta_t A_t}\,\tilde{h}_{t-1} \;+\; \Delta_t \overline{B}_t x_t,
    \qquad
    y_t \;=\; \overline{C}_t^\top \tilde{h}_t,
\end{equation}$$






**Proposition 4**  Discretizing a complex SSM with the exponential-trapezoidal rule yields the recurrence$$\begin{align}
    h_t 
    &= \alpha_t h_{t-1} 
    + \beta_t \left(\prod_{i=0}^{t-1} R_i^\top \right)B_{t-1} x_{t-1} 
    + \gamma_t \left(\prod_{i=0}^t R_i^\top \right)B_t x_t, \nonumber \\
    y_t
    &= \left[\left(\prod_{i=0}^t R_i^\top\right)C_t\right]^\top h_t .
\end{align}$$
Here, $R_t$ is the block-diagonal rotation matrix

**Proof**
We begin from the complex SSM 
$$\begin{align*}
    \dot{h}(t)
    &= \mathrm{Diag}\,\left(A(t)+i\boldsymbol{\theta}(t)\right)\,h(t)
      \;+\;\left(B(t)+i\hat{B}(t)\right)\,x(t),\\
    y(t)
    &= \mathrm{Re}\,\left((C(t)+i\hat{C}(t))^\top h(t)\right),
\end{align*}
$$
where $A(t)\in\mathbb{R}$ is a scalar and $\boldsymbol{\theta}(t),B(t),\hat{B}(t),C(t),\hat{C}(t)\in\mathbb{R}^{N/2}$.
$$\begin{align*}
h_t
&\approx e^{\Delta_t(A_t+i\boldsymbol{\theta}_t)}\,h_{t-1}
\;+\; \int_{\tau_{t-1}}^{\tau_t}
      e^{(\tau_t-\tau)(A_t+i\boldsymbol{\theta}_t)}
      \left(B(\tau)+i\hat{B}(\tau)\right)\,x(\tau)\,d\tau.
\end{align*}$$
$$\begin{align}

h_t
&= e^{\Delta_t(A_t+i\boldsymbol{\theta}_t)}\,h_{t-1}
  \;+\; \beta_t\,e^{i\Delta_t\boldsymbol{\theta}_t}\left(B_{t-1}+i\hat{B}_{t-1}\right)x_{t-1}
  \;+\; \gamma_t\left(B_t+i\hat{B}_t\right)x_t,
\end{align}$$


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





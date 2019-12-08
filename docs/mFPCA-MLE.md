# Multivariate FPCA

This is the code documentation for MFPCA. 

- Manifold Optimization
- EM Algorithm

## Manifold Optimization

### The Objective Function

Let $S_n = y_ny_n^T$ and $\Sigma_n = B_n DB_n^T + 
\sigma_e^2 I$. 

The Euclidean objective function is
$$
\log \det \Sigma_n + \mathrm{tr}(S_n\Sigma_n^{-1})
$$

The manifold version is
$$
\log \det \Sigma_n + \mathrm{tr}(S_n\Sigma_n^{-1})
+ \mu_2 \mathrm{tr}(U^T\Gamma U),
$$
where $\Sigma_n = B_n UWU^TB_n^T + \sigma_e^2 I$.

### Update Sigma

> Useful results:
> $$\frac{\partial \log\det A}{\partial t} = \mathrm{tr}(A^{-1}\frac{\partial A}{\partial t}).$$
> $$\frac{\partial A^{-1}}{\partial t} =- A^{-1} \frac{\partial A}{\partial t}  A^{-1}$$

- The first order gradient with respect to $\sigma^2_e$ is
$$
\begin{align}
\frac{d F}{d\sigma_e^2} = 
& \sum_{n}\mathrm{tr}(\Sigma_n^{-1} \frac{\partial \Sigma_n}{\partial \sigma^2_e}) - \mathrm{tr}(S_n\Sigma_n^{-1} \frac{\partial \Sigma_n}{\partial \sigma^2_e}\Sigma_n^{-1} ) \\
=& \sum_{n}\mathrm{tr}(\Sigma_n^{-1}) - \mathrm{tr}(S_n\Sigma_n^{-2}) \\
=&\sum_{n}\mathrm{tr}(\Sigma_n^{-1}-S_n\Sigma_n^{-2}) \\
=&\sum_{n} \left[\mathrm{tr}(\Sigma_n^{-1})-
\langle\Sigma_n^{-1}y_n,\Sigma_n^{-1}y_n \rangle\right]
\end{align} 
$$ The last line is implemented in the code. 

- The second order gradient with respect to $\sigma^2_e$ is
$$
\begin{align}
&\frac{d^2 F}{(d\sigma_e^2)^2}\\
 = &\sum_{n} -\mathrm{tr}(\Sigma_n^{-1} \frac{\partial \Sigma_n}{\partial \sigma^2_e}\Sigma_n^{-1}) +  \mathrm{tr}(S_n\Sigma_n^{-1} \frac{\partial \Sigma_n}{\partial \sigma^2_e}\Sigma_n^{-1} \cdot \Sigma_n^{-1}  ) + \\
 &\qquad\qquad \mathrm{tr}(S_n\Sigma_n^{-1} \cdot \Sigma_n^{-1} \frac{\partial \Sigma_n}{\partial \sigma^2_e} \Sigma_n^{-1}  )
 \\
=&\sum_{n}-\mathrm{tr}(\Sigma_n^{-2}) +2 \mathrm{tr}(S_n\Sigma_n^{-3}) \\
=& \sum_{n}\mathrm{tr}(2S_n\Sigma_n^{-3} - \Sigma_n^{-2})\\
=&\sum_{n}2 \langle \Sigma_n^{-1}y_n, \Sigma_n^{-1}\Sigma_n^{-1}y_n\rangle  - \langle \Sigma_n^{-1}, \Sigma_n^{-1}\rangle
\end{align}
$$
The last line is implemented in the code. 


In the code, to ensure the positiveness of $\sigma_e^2$, it is implemented as $\sigma_e^2 = \exp(\gamma)$, and 
$\gamma$ is `sigmaSqLog` in the code.
$$
\frac{d F}{d\gamma} =
\frac{d F}{d\sigma_e^2} \frac{d \sigma_e^2}{d \gamma}.
$$
and
$$
\begin{align}
\frac{d^2 F}{(d\gamma)^2}
 =&\frac{d^2 F}{(d\sigma_e^2)^2} 
\left(\frac{d \sigma_e^2}{d \gamma}\right)^2 +
\frac{d F}{d\sigma_e^2} \frac{d^2 \sigma_e^2}{(d \gamma)^2} \\
=&\frac{d^2 F}{(d\sigma_e^2)^2} \times \exp(2\gamma) +
\frac{d F}{d\sigma_e^2} \times \exp(\gamma)
\end{align}
$$


### The Gradient Function

Define 
$$
K = \sum_{n=1}^N B_n^T (\Sigma_n^{-1}-
\Sigma_n^{-1} S_n\Sigma_n^{-1}) B_n
$$

Suppose we have Cholesky decompostion 
$\Sigma_n = LL^T$, the way to compute $K$ becomes
$$
\begin{align}
K &= \sum_{n=1}^N (L^{-1}B_n)^T ( I - L^{-1}S_n(L^T)^{-1}) L^{-1}B_n \\
 &= \sum_{n=1}^N (L^{-1}B_n)^T ( I - (L^{-1}y_n)(L^{-1}y_n)^T) L^{-1}B_n
 \end{align}
$$

Suppose the matrix $W$ depends on $t$
$$
\begin{align}
\frac{d F}{dt} =
&  \sum_{n=1}^N\mathrm{tr}(\Sigma_n^{-1} \frac{\partial \Sigma_n}{\partial t}) - \mathrm{tr}(S_n\Sigma_n^{-1} \frac{\partial \Sigma_n}{\partial t}\Sigma_n^{-1} ) \\
=&   \sum_{n=1}^N\mathrm{tr}(\Sigma_n^{-1} B_n U\frac{\partial W}{\partial t}U^TB_n^T )- \mathrm{tr}(S_n\Sigma_n^{-1} B_n U\frac{\partial W}{\partial t}U^TB_n^T \Sigma_n^{-1} ) \\
=&  \sum_{n=1}^N\left\langle
U^TB_n^T\Sigma_n^{-1} B_n U
- U^TB_n^T \Sigma_n^{-1}S_n\Sigma_n^{-1} B_n U
, \frac{\partial W}{\partial t}
\right\rangle \\
=& \left\langle
U^TK U, \frac{\partial W}{\partial t}
\right\rangle
\end{align}
$$


> The Euclidean gradient function is just $K$, without separation of $D=UWU^T$.


Suppose the matrix $U$ depends on $t$
$$
\begin{align}
\frac{d F}{dt} = 
&  \sum_{n=1}^N\mathrm{tr}(\Sigma_n^{-1} \frac{\partial \Sigma_n}{\partial t}) - \mathrm{tr}(S_n\Sigma_n^{-1} \frac{\partial \Sigma_n}{\partial t}\Sigma_n^{-1} ) \\
=&  \sum_{n=1}^N \mathrm{tr}(\Sigma_n^{-1} B_n [\frac{\partial U}{\partial t}WU^T+ UW\frac{\partial U^T}{\partial t}]B_n^T )\\
&\quad - \mathrm{tr}(S_n\Sigma_n^{-1} B_n [\frac{\partial U}{\partial t}WU^T+ UW\frac{\partial U^T}{\partial t}]B_n^T \Sigma_n^{-1} ) \\
=& \sum_{n=1}^N \left\langle
2B_n^T\Sigma_n^{-1} B_n UW
- B_n^T \Sigma_n^{-1}S_n\Sigma_n^{-1} B_n UW
, \frac{\partial W}{\partial t}
\right\rangle \\
=& \left\langle
2K UW, \frac{\partial W}{\partial t}
\right\rangle
\end{align}
$$








> The gradient funciton for the manifold optimization is
$$
\frac{\partial F}{\partial W} = U^TKU, \quad
\frac{\partial F}{\partial U} = 2KUW + 2\mu_2 \Gamma U.
$$


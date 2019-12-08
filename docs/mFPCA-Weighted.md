# Weighted LOCAL FPCA



## Maximum Likelihood 

### The Objective Function

Let $S_n = y_ny_n^T$ and $\Sigma_n = B_n DB_n^T + 
\sigma_e^2 I$. 

The Euclidean objective function is
$$
\sum_{n=1}^N w_n \left[\log \det \Sigma_n + \mathrm{tr}(S_n\Sigma_n^{-1})\right]
$$

The manifold version is
$$
\sum_{n=1}^N w_n \left[\log \det \Sigma_n + \mathrm{tr}(S_n\Sigma_n^{-1})\right]
+ \mu_2 \mathrm{tr}(U^T\Gamma U),
$$
where $\Sigma_n = B_n UWU^TB_n^T + \sigma_e^2 I$.


### The Gradient Function

The weights act on the matrix $K$. 
$$
K = \sum_{n=1}^N w_n \left[B_n^T (\Sigma_n^{-1}-
\Sigma_n^{-1} S_n\Sigma_n^{-1}) B_n\right]
$$




## Least Square Optimization

**Let square root of the weight act on $m_n$**

Let $Z_n = y_ny_n^T - \sigma^2_eI$
$$
F(S) = \frac{1}{2N}\sum_{n=1}^N \frac{w_n}{m_n^2}\Vert Z_n - 
B_nSB_n^T\Vert_F^2  
$$

This is coded in the function `setNewZ(double)`. No modification is required for `gradF_Euc` and `objF_Euc`.
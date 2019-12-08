# Multivariate FPCA

This is the code documentation for MFPCA. 

- Manifold Optimization
- EM Algorithm

## Manifold Optimization
### Objective Function
Let $Z_n = y_ny_n^T - \sigma^2_eI$
$$
\begin{align}
F(U, W) &= \frac{1}{2N}\sum_{n=1}^N \frac{1}{m_n^2}\Vert Z_n - B_n UWU^TB_n^T\Vert_F^2  \\
&= \frac{1}{2N}\sum_{n=1}^N \frac{1}{m_n^2}\langle Z_n - B_n UWU^TB_n^T, Z_n - B_n UWU^TB_n^T\rangle \\
&= \frac{1}{2N}\sum_{n=1}^N \frac{-2}{m_n^2} \langle B_n UWU^TB_n^T, Z_n \rangle
+ \frac{1}{2N}\sum_{n=1}^N \frac{1}{m_n^2}\langle  B_n UWU^TB_n^T, B_n UWU^TB_n^T\rangle
 +C\\
 &= \frac{1}{2N}\sum_{n=1}^N \frac{-2}{m_n^2} \langle  UWU^T, B_n^TZ_n B_n\rangle
+ \frac{1}{2N}\sum_{n=1}^N \frac{1}{m_n^2}\langle   UWU^T, B_n^TB_n UWU^TB_n^TB_n\rangle
 +C\\
 &= \frac{1}{2N} \times -2 \langle  UWU^T, \sum_{i=1}^N B_n^TZ_n B_n/m_n^2\rangle
+ \frac{1}{2N}\sum_{n=1}^N 
 \left\langle   UWU^T, \frac{B_n^TB_n}{m_n} UWU^T\frac{B_n^TB_n}{m_n}\right\rangle  +C\\
\end{align}
$$

***Define*** 

- $S_B = \mathrm{BtZBSum} =\sum_{i=1}^N B_n^TZ_n B_n/m_n^2$ 
- $C_n = 
\mathrm{BtB[n]} = B_n^TB_n/m_n$. 

> The above matrices are stored in the memory to avoid replicated computation. 

$$
\begin{align}
F(U, W) 
 &= \frac{1}{2N} \times -2 \langle  UWU^T, S_b\rangle
+ \frac{1}{2N}\sum_{n=1}^N  \left\langle UWU^T, C_n UWU^TC_n\right\rangle  +C\\
 &= \frac{1}{2N} \times -2 \langle  W, U^TS_bU\rangle
+ \frac{1}{2N}\sum_{n=1}^N  \left\langle W, \underline{U^TC_n U}W
\underline{U^TC_nU}\right\rangle  +C\\
\end{align}
$$

Inner product with $W$ has smaller computation burden. 

For cross-validation, we need the extra constant term $C$ to evaluate the effect of $\sigma_e^2$.
$$
\begin{align}
C &= 
\frac{1}{2N}\sum_{n=1}^N\frac{1}{m_n^2}\langle Z_n, Z_n\rangle  \\
& = \frac{1}{2N}\sum_{n=1}^N\frac{1}{m_n^2}\left(\langle y_ny_n^T, y_ny_n^T\rangle - 2\langle y_ny_n^T, \sigma_e^2I\rangle 
+\langle \sigma_e^2I, \sigma_e^2I\rangle \right) \\
&=C' - \frac{1}{N}\sum_{n=1}^N\frac{y_n^Ty_n}{m_n^2} \times \sigma_e^2 + \frac{1}{2N}\sum_{n=1}^N\frac{1}{m_n} \times 
\sigma_e^4 \\
& = C' + \mathrm{cv1} \times \sigma_e^2 + \mathrm{cv2} \times \sigma_e^4 \,.
\end{align}
$$
The variables `cv1` and `cv2` in the code are used to adjust cross-validation value. 

### Gradient Function
The followings are the `grad1` and `grad2` in the code
$$
\begin{align}
\frac{\partial F}{\partial U} & = -\frac{1}{N} \sum_{n=1}^N
\frac{1}{m_n^2} B_n^T (y_ny_n^T - B_nUWU^TB_n^T - \sigma_e^2I)B_nU W \\
\frac{\partial F}{\partial W} &= -\frac{1}{N} \sum_{n=1}^N
\frac{1}{m_n^2} U^TB_n^T (y_ny_n^T - B_nUWU^TB_n^T - \sigma_e^2I)B_nU 
\end{align}
$$
We can compute the following (`coreGrad`) first for the above two
$$
\begin{align}
&\sum_{n=1}^N
\frac{1}{m_n^2} B_n^T (y_ny_n^T - B_nUWU^TB_n^T - \sigma_e^2I)B_n \\
=\, & S_b   - \sum_{n=1}^N \underline{C_nU}W\underline{U^TC_n}
\end{align}
$$


### Update sigma

The measurement error is estimated by
$$
\begin{align}
\sigma_e^2 =& \frac{1}{\sum m_n}
\sum \mathrm{tr}\left(y_ny_n^T - B_nUWU^TB_n^T\right) \\
=& \frac{1}{\sum m_n}
\sum_{n=1}^N
y_n^T y_n- \mathrm{tr}\left(B_n^TB_nUWU^T\right) \\
=& \frac{1}{\sum m_n}
\sum_{n=1}^N\left(
y_n^T y_n- m_n\mathrm{tr}\left(C_n UWU^T\right) \right) \\
=& \frac{1}{\sum m_n}
\left( \sum_{n=1}^N
y_n^T y_n- 
\left\langle \sum_{n=1}^N m_nC_n, UWU^T\right\rangle \right)
\end{align}
$$

***Define*** (implemented?)

- `nTotal`$=\sum_{n=1}^N m_n$ (Y)
- `ySqSum`$=\sum_{n=1}^N y_n^Ty_n$ (N)
- `BtBSum`$=\sum_{n=1}^N m_nC_n = \sum_{n=1}^N B_n^TB_n$ (N)
- `BtB_mn2_Sum`$ = \sum_{n=1}^N B_n^TB_n/m_n^2$ (Y)

The $\sigma_e^2$ can be computed with minimal amount of computation. 
$$
\sigma_e^2 = \frac{1}{\text{nTotal}}
\left( \text{ySqSum} - \langle \text{BtBSum},
UWU^T\rangle
\right)
$$
Subtraction of two large numbers, not enough accuracy? 
After getting the new $\sigma_e^2$ and the old $\tilde{\sigma}_e^2$, 
$S_B =\mathrm{BtZBSum}$ will be updated by
$$
S_B = S_B + (\sigma_e^2 - \tilde{\sigma}_e^2)\times
\text{BtB_mn2_Sum}
$$


## EM Version

For each component $l=1,\cdots, p$, 
$$ 
y_{il}(t) = \mu_l(t) +\xi_l(t)^T\alpha_i + \epsilon_{il}(t) 
$$
The $p$-vector of function $$y_i(t) = \mu(t) + \Xi^T\alpha_i + \epsilon_i(t).$$ $\Xi = (\xi_1(t), \cdots, \xi_p(t))$ is a $K$-by-$p$ matrix. We use orthonormalized basis to expand the mean and PC functions ($\psi(t)$ is $M$-vector).

$\Psi(t)=\mathrm{diag}\{\psi^T(t),\cdots, \psi^T(t)\}$  is a $p\times pM$ matrix.
$$ 
y_{i}(t) = \Psi(t)\theta_u + \Psi(t)\Theta_v\alpha_i + \epsilon_{i}(t)
$$

* $\theta_u$ is $pM$-vector, each block of size $M$ is the coefficients of one component mean function. 
* $\Theta_v$ is $pM\times K$ matrix. Each column is the coefficients of one PC function. For each column, it is blocked in the same way as $\theta_u$.
* The row is repeated accordingly to accout for discretized observation for each component. $$ 
y_{i} = \Psi\theta_u + \Psi\Theta_v\alpha_i + \epsilon_{i}.$$ The total number of observation for the $i$-th object is $n_i$. Each component $l$ does not have to be observed at the same time points.

The full penalized log-likelihood is
$$
\begin{align}
& \sum_{i} \{
\frac{n_i+k}{2}\log(2\pi) + \frac{n_i}{2}\log\sigma^2
+\frac{1}{2}\log|D| \\
& \quad +\frac{1}{2\sigma^2} 
(y_{i} - \Psi_i\theta_u - \Psi_i\Theta_v\alpha_i )^T
(y_{i} - \Psi_i\theta_u - \Psi_i\Theta_v\alpha_i ) \\
&\quad +\frac{1}{2} \alpha_i^TD^{-1}\alpha_i \}
+\frac{\lambda N}{2}\{\theta_u^T\Omega\theta_u
+\sum_{k=1}^K \Theta_k^T\Omega\Theta_k
\}
\end{align}
$$

Storage

- $\Psi_i^Ty_i$'s are stored in `BiTYi`, and their summation in `BiTYiSum`.
- $\Psi_i^T\Psi_i$'s are stored in `BiTBi`, and their summation in `BiTBiSum`.
- $\sum_{i=1}^N n_i$ in `nTotal`

### Expectation Step
For the expectation step, $\alpha_i|y_i$ is normal distributed with
$$
\mathcal{N}\left(
(\sigma^2D^{-1}+\Theta^T\Psi_i^T\Psi_i\Theta)^{-1}
\Theta^T\Psi_i^T(y_i-\Psi_i\theta_u),\, 
(D^{-1}+\Theta^T\Psi_i^T\Psi_i\Theta/\sigma^2)^{-1}
\right)
$$
The mean and covarinace are stored in `alphaIHat` and `alphaICov` respectively.
We have
$$
\hat{\alpha}_i = (\sigma^2D^{-1}+\Theta^T\Psi_i^T\Psi_i\Theta)^{-1} \Theta^T\Psi_i^T(y_i-\Psi_i\theta_u)
$$
and the following second moment stored in `alphaI2M`.
$$
\widehat{\alpha_i\alpha_i^T} = \hat{\alpha}_i\hat{\alpha}_i^T+
(D^{-1}+\Theta^T\Psi_i^T\Psi_i\Theta/\sigma^2)^{-1}
$$


### Maximization Step I
Update $\sigma^2$. Let $\epsilon_i = y_{i} - \Psi\theta_u - \Psi\Theta_v\alpha_i$. 
$$
\begin{align}
 \sigma^2&\leftarrow \frac{1}{\sum n_i}\sum_i E(\epsilon_i^T\epsilon_i | X) \\
& = \frac{1}{\sum n_i}\sum_i \{(y_{i} - \Psi_i\theta_u - 
\Psi_i\Theta_v\hat{\alpha}_i )^T
(y_{i} - \Psi_i\theta_u - \Psi_i\Theta_v\hat{\alpha}_i ) \\
&\qquad +\mathrm{tr}\left[\Theta_v^T\Psi_i^T\Psi_i\Theta_v
(D^{-1}+\Theta^T\Psi_i^T\Psi_i\Theta/\sigma^2)^{-1}\right]\}
\end{align}
$$


Update $D_{ll}$
$$
\begin{align}
D_{ll} \leftarrow \frac{1}{N}\sum_i E(\alpha_{il}^2 | X)
\end{align} = \frac{1}{N}\sum_i \widehat{\alpha_{il}^2}
$$

### Maximization Step II
Update $\theta_u$ and $\Theta$ until convergence. 
$z_i = y_{i} -  \Psi_i\Theta_v\hat{\alpha}_i$
$$
\begin{align}
\theta_u \leftarrow 
\left(\sum_i \Psi_i^T\Psi_i + \lambda N \Omega\right)^{-1}
\left(\sum_i \Psi^T_i z_i \right)
\end{align}
$$
where $\sum_i \Psi^T_i z_i = 
\sum_i\Psi^T_iy_i - \sum_i\Psi_i^T\Psi_i\Theta_\nu \hat{\alpha_i}$.
Besides, 
$$
\begin{align}
\Theta_l \leftarrow 
 \left(\sum_i  \widehat{\alpha^2_{il}}\Psi_i^T\Psi_i + \lambda N \Omega\right)^{-1}  
 \sum_i \Psi^T_i \left(
\hat{\alpha}_{il}(y_i-\Psi_i\theta_u) 
-\sum_{m\neq l} \widehat{\alpha_{il}\alpha_{im}} \Psi_i\Theta_m
\right)
\end{align}
$$


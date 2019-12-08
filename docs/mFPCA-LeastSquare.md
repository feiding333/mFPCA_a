# Multivariate FPCA

The first version with least square loss. This does not appear to have good performance.


## Least Square Optimization
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


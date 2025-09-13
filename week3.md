Mean of stroke intensity:

$$
\mu_I = \frac{\sum_{m,n} I[m,n] \cdot B[m,n]}{\sum_{m,n} B[m,n]}
$$
this is first vector i want you to add

Standard deviation of stroke intensity:

$$
\sigma_I = \sqrt{ \frac{\sum_{m,n} (I[m,n] - \mu_I)^2 \cdot B[m,n]}{\sum_{m,n} B[m,n]} }
$$

this is the second vector

Grayscale conversion formula:

$$
I = 0.299R + 0.587G + 0.114B
$$

Binary mask definition:

$$
B[m,n] = 
\begin{cases}
1 & \text{stroke} \\
0 & \text{non-stroke}
\end{cases}
$$

Morphology (Erosion):

$$
B_0[I[m,n]] = B[m,n]
$$

$$
B_k[m,n] = B_{k-1}[m,n] \land B_{k-1}[m+1,n] \land B_{k-1}[m-1,n] \land B_{k-1}[m,n+1] \land B_{k-1}[m,n-1]
$$

Erosion ratio:

$$
e_k = \frac{\sum_{m,n} B_k[m,n]}{\sum_{m,n} B[m,n]} \quad \text{for } k = 1, 2, 3
$$

i want you to add k=1,2,3 as three different vector
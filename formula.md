**Moments**

定義二值圖像 \( B(m, n) \) 為：

\[
B(m, n) =
\begin{cases}
1 & \text{stroke} \\
0 & \text{background}
\end{cases}
\]

其中 \( m, n \) 對應到圖像的座標（像素點位置）。我們以圖像大小為 \(189 \times 189\) 的情況為例，設定圖像中心點 \((95, 95)\) 為原點 \((0, 0)\)，則實際的 \( m, n \) 可視為以此為基準的 \( x, y \) 座標。

---

一階中心點（質心）可由下式求得：

\[
m_0 = \frac{\sum_m \sum_n m \cdot B(m, n)}{\sum_m \sum_n B(m, n)}
\]

\[
n_0 = \frac{\sum_m \sum_n n \cdot B(m, n)}{\sum_m \sum_n B(m, n)}
\]

---

中心化的高階混合動差（central mixed moments）定義為：

\[
V_{a,b} = \frac{\sum_m \sum_n (m - m_0)^a (n - n_0)^b \cdot B(m, n)}{\sum_m \sum_n B(m, n)} = \mathbb{E} \left[ (m - m_0)^a (n - n_0)^b \right]
\]

其中 \( a + b = k \) 為某一階次（order \(k\)）的總和。舉例來說，若為二階（second-order）moment，則會包含以下三項：

- \( V_{2,0} \)
- \( V_{1,1} \)
- \( V_{0,2} \)

這些可以用來描述圖像的橢圓性、偏斜程度、分佈方向等特徵。

---

**應用於 SVM 特徵擴展：**

將上述的三個二階動差 \( V_{2,0}, V_{1,1}, V_{0,2} \) 作為額外的三個特徵維度，加入原來的特徵向量之中，以提升 Support Vector Machine (SVM) 的分類能力。這樣原始的特徵向量會增加 3 維度（dimensions）。

\[
\text{Original Feature Vector: } [x_1, x_2, ..., x_n]
\]
\[
\text{Enhanced Feature Vector: } [x_1, x_2, ..., x_n, V_{2,0}, V_{1,1}, V_{0,2}]
\]

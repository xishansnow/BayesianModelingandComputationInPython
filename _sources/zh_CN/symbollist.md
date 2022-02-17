---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(symbols)= 
# 符号表

$\log(x)$ ：$x$ 的自然对数 

$\mathbb{R}$ ：实数 

$\mathbb{R}^n$ ：n维实数向量空间 

$\mathcal{A, S}$ ：集合 

$x \in A$ ：集合成员。$x$ 是集合 $A$ 的一个元素

$\unicode{x1D7D9}_A$ ：指示函数。当 $x \in A$ 时返回 $1$ ，否则返回 $0$

$a \propto b$ ：$a$ 正比于 $b$  

$a \underset{\sim}{\propto}  b$ ：$a$ 近似正比于 $b$  

$a \approx b$ ：$a$ 近似等于 $b$

$\forall x$ ：对于所有 $x$  

$a, c, \alpha, \gamma$：标量采用小写 

$\mathbf{x, y}$ ：向量采用粗体小写字母表示，默认为列向量形式。因此有 $\mathbf{x}=[x_1,\dots,x_n]^T$ 

$\mathbf{X, Y}$ ：矩阵采用粗体大写字母表示 

$X, Y$ ：随机变量采用罗马字体的大写字母表示

$x, y$ ：随机变量的结果采用罗马字体的小写字母表示

$\boldsymbol{X, Y}$ ：随机向量采用粗斜体的大写字母表示, $\boldsymbol{X} = [X_1,\dots,X_n]^T$ 

$\boldsymbol{\theta}$ ：模型参数用小写的希腊字母表示。需要注意的是，在贝叶斯统计中，参数通常被视为随机变量 

$\hat \theta$ ： $\boldsymbol{\theta}$ 的点估计

$\mathbb{E}_{X}[X]$ ：随机变量 $X$ 关于 $X$ 的期望，更多时候被简写为 $\mathbb{E}[X]$ 

$\mathbb{V}_{X}[X]$ ：随机变量 $X$ 关于 $X$ 的方差，更多时候被简写为 $\mathbb{V}[X]$ 

$X \sim p$ ：随机变量 $X$ 服从分布 $p$  

$p(\cdot)$ ：概率密度函数或概率质量函数 

$p(y \mid \boldsymbol{x})$ ： 在给定 $\boldsymbol{x}$ 时，随机变量 $y$ 的概率（密度）。 这是 $p(Y=y \mid \boldsymbol{X}=\boldsymbol{x})$ 的简写。

$f(x)$ ：关于 $x$ 的任意函数 

$f(\boldsymbol{X}; \theta, \gamma)$ ：$f$ 是 $\boldsymbol{X}$ 的函数，其参数为 $\theta$ 和 $\gamma$ 。我们使用这个符号来强调 $\boldsymbol{X}$ 是传递给函数（或模型）的数据，而 $\theta$ 和 $\gamma$ 是函数的参数。 

$\mathcal{N}(\mu, \sigma)$ ：均值为 $\mu$ 标准差为 $\sigma$ 的高斯（或高斯）分布。

$\mathcal{HN}(\sigma)$ ：标准差为 $\sigma$ 的半高斯（或半高斯）分布

$\text{Beta}(\alpha, \beta)$ ：形状参数为 $\alpha$ 和 $\beta$ 的贝塔分布

$\text{Expo}(\lambda)$ ：速率参数为 $\lambda$ 的指数分布

$\mathcal{U}(a, b)$ ：下界为 $a$ 上界为 $b$ 的均匀分布

$\mathcal{T}(\nu, \mu, \sigma)$ ：高斯等级为 $\nu$ （ 也称自由度 ）、位置参数为 $\mu$ （ 当 $\nu > 1$ 时指均值 ）、尺度参数为 $\sigma$ （ 当 $\lim_{\nu\to\infty}$ 时指标准差 ）的学生 $t$ 分布

$\mathcal{HT}( \nu \sigma)$ ：高斯等级为 $\nu$ （ 也称自由度 ）、尺度参数为 $\sigma$ 的半学生 $t$ 分布 $\nu$ 

$\text{Cauchy}(\alpha, \beta)$ ：位置参数为 $\alpha$ 、尺度参数为 $\beta$ 的柯西分布

$\mathcal{HC}(\beta)$ ：尺度参数为 $\beta$ 的半柯西分布

$\text{Laplace}(\mu, \tau)$ ：均值为 $\mu$ 、尺度为 $\tau$ 的拉普拉斯分布

$\text{Bin}(n, p)$ ：总实验次数为 $n$ ，成功次数为 $p$ 的二项分布 

$\text{Pois}(\mu)$ ：均值为 $\mu$ 的泊松分布

$\mathcal{NB}(\mu, \alpha)$ ：泊松参数为 $\mu$ 、伽马分布参数为 $\alpha$ 的负二项分布

$\mathcal{GRW}(\mu, \sigma)$ ：新偏移为 $\mu$、 新标准差为 $\sigma$ 的高斯随机游走分布

$\mathbb{KL}(p \parallel q)$ ： $p$ 到 $q$ 的 $KL$ 散度

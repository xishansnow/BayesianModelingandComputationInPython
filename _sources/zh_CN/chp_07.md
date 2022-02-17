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

(chap6)= 

# 第七章：贝叶斯加性回归树 

<style>p{text-indent:2em;2}</style>

在第 [5](chap3_5) 章中，我们看到了如何通过对一系列（简单）基函数求和来逼近一个函数。我们展示了 `B-样条` 在用作基函数时如何具有一些不错的属性。在本章中，我们将讨论一种类似方法，但将使用**决策树**而不是 `B-样条`。决策树是表示分段常数函数或阶跃函数（ 曾在第 5 章中出现 ）的另外一种灵活方式。我们将特别关注**贝叶斯加性回归树（ BART ）**，一种贝叶斯非参数模型，它通过对多个决策树求和来获得更灵活的模型 [^1]。加性回归树通常以更接近机器学习的术语而不是统计术语来讨论 {cite:p}`breiman2001`。从某种意义上说，相较其他章节中的模型而言，`BART` 更像一个“一劳永逸的模型”。

在 `BART` 的文献中，人们通常不写基函数，而是讨论学习器，但总体思路非常相似。我们使用简单函数（也称为学习器）的组合来逼近复杂函数，并具备足够的正则化，这样就可以在模型不太复杂的情况下获得足够的灵活性，即不会过拟合。使用多个学习器来求解同一问题的方法被称为**集成方法（ Ensemble Methods ）**。在这种情况下，学习器可以是你能想到的任何统计模型或数据算法。集成方法基于这样一个基本观察：**组合多个弱学习器通常比使用单个强学习器效果更好**。为了在准确度和泛化方面获得良好效果，一般认为基础学习器应该尽可能准确，并且尽可能多样化 {cite:p}`ZhouEnsembleMethods2012` 。 `BART` 的主要贝叶斯思想是：决策树很容易过拟合，因此我们添加了一个正则化先验（ 或收缩先验 ），以使每棵树都表现为一个弱学习器。

为了将上述描述转化为可以理解和应用的东西，我们下面会首先讨论决策树。如果你已经熟悉这个板块的内容，可以跳过下一节。

(decision-trees)= 

## 7.1 决策树 

假设有两个变量 $X_1$ 和 $X_2$，我们希望依据这些变量将对象分为两类： ⬤ 或 ▲ 。为了实现此目标，可以使用 {numref}`fig:decision_tree` 左侧子图中所示的树结构。树是节点的集合，其中任何两个节点之间最多通过一条线或一条边相连。 {numref}`fig:decision_tree` 中的树被称为二叉树，因为每个节点最多可以有两个子节点。没有子节点的节点被称为叶子节点或终端节点。在此示例中，我们有 $2$ 个内部节点（ 表示为矩形 ）和 $3$ 个终端节点（ 表示为圆角矩形 ）。每个内部节点都有一个与之关联的决策规则。如果遵循这些决策规则，我们最终将到达其中一个叶节点，该节点将提供决策问题的答案。

例如，如果变量 $X_1$ 的实例大于 $c_1$，决策树会将类 ⬤ 分配给该实例。相反，如果 $x_{1i}$ 的值小于 $c_1$ 并且 $x_{2i}$ 的值小于 $c_2$，那么必须分配类 ▲ 给该实例。

从算法上讲，可以将树概念化为一组 `if-else` 语句，我们遵循这些语句来执行分类等特定任务。还可以从几何角度理解二叉树，将其视为将样本空间划分为不同块的一种方式，如 {numref}`fig:decision_tree` 的右侧子图所示。每个块由轴相互垂直的分割线定义，因此样本空间的每次分割都将与预测变量的其中一个轴对齐。

在数学上，我们可以说决策树 $g$ 完全由两个集合定义：

- $\mathcal{T}$ ：边和节点的集合（ {numref}`fig:decision_tree` 中的正方形、圆角正方形以及连接它们的线）， 以及与内部节点相关联的决策规则。

- $\mathcal{M} = \{\mu_1, \mu_2, \dots, \mu_b\}$ ：与 $\mathcal{T}$ 的每个终端节点相关联的一组参数值。

那么决策树 $g(X; \mathcal{T}, \mathcal{M})$ 就是将 $\mu_i \in M$ 分配给 $X$ 的那个函数。例如，在 {numref}`fig:decision_tree` 中，$\mu_{i}$ 的值为（ ⬤ 、⬤ 和 ▲ ）。 $g$ 函数将 ⬤ 分配给 $X_1$ 大于 $c_1$ ； 将 ⬤ 分配给 $X_1$ 小于 $c_1$ 并且 $X_2$ 大于 $c_2$ ； 将 ▲ 分配给 $X_1$ 小于 $c_1$ 并且 $X_2$ 小于 $c_2$ 。

当我们讨论树的先验时，将其抽象定义为两个集合构成的元组 $g(\mathcal{T}, \mathcal{M})$ 将非常有用。

```{figure} figures/decision_tree.png
:name: fig:decision_tree
:width: 8.00in

二叉树（左）和相应的空间分区（右）。树的内部节点是那些有子节点的节点，它们有一个到下面节点的链接，内部节点具有与之关联的拆分规则。终端节点是没有子节点的节点，它们包含要返回的值，在本例中为 ⬤ 或 ▲ 。决策树将样本空间划分为由垂直轴分割线分隔的子空间块。这意味着样本空间的每次分割都将与预测变量轴之一对齐。
``` 

虽然 {numref}`fig:decision_tree` 展示了如何将决策树用于分类，其中 $\mathcal{M}_j$ 包含类或标签值，但它们也可以用于回归。此时不是将终端节点与类标签相关联，而是与实数相关联，例如块内数据点的均值。 {numref}`fig:decision_tree_reg` 显示了只有一个预测变量的回归问题。左边有一棵类似于 {numref}`fig:decision_tree` 的二叉树，主要区别在于 {numref}`fig:decision_tree_reg` 中的二叉树在每个终端节点返回一个实数值。将树与右侧的类正弦数据比较，特别注意将数据分成三个块而不是连续函数的逼近，并且估计了每个块的均值。

```{figure} figures/decision_tree_reg.png
:name: fig:decision_tree_reg
:width: 8.00in

二叉树（左）和相应的分区空间（右）。树的内部节点是那些有子节点的节点，内部节点有与之相关联的拆分规则。终端节点是那些没有子节点的节点，包含要返回的值（ 在本例中为 $1.1$、$1.9$ 和 $0.1$ ）。可以看到树是表示分段函数的一种方式。
``` 

回归树不限于返回块内数据点的均值，还有其他选择。例如，可以将叶节点与数据点的中值关联，或者可以是每个块内数据点的线性回归拟合，甚至更复杂的函数。不过均值可能是回归树中最常见的选择。

需要特别注意：回归树的输出不是平滑函数，而是分段阶梯函数。但这并不意味着回归树一定是一种拟合平滑函数的错误选择。因为理论上，我们可以用阶梯函数逼近任何连续函数，而且在实践中这种逼近已经足够好了。

决策树的一个吸引人的特性是其可解释性，你可以从字面上阅读决策树并按照解决某个问题所需的步骤进行操作。因此，你可以透明地了解该方法在做什么，为什么它会以这种方式执行，为什么某些类可能无法正确分类，或者为什么某些数据的近似度很差。

此外，用简单术语向非技术型的观众解释结果也更容易。

不幸的是，决策树的灵活性意味着它们很容易过拟合，因为你总能找到一个足够复杂的树，使得每个数据点都对应一个分区。关于过于复杂的分类解决方案，请参见 {numref}`fig:decision_tree_overfitting`。拿来一张纸，绘制几个数据点，然后为每个数据点创建一个单独隔离的分区，你可以很容易看到这种过拟合。在进行此练习时，你可能还会注意到，实际上有不止一棵树可以拟合数据。

```{figure} figures/decision_tree_overfitting.png
:name: fig:decision_tree_overfitting
:width: 4.5in

过于复杂的样本空间分区。每个数据点都分配给了一个单独的块。之所以称之为过于复杂的分区，因为我们完全可以使用更简单的分区方案（ 如 {numref}`fig:decision_tree` 中的分区 ）以相同准确度来解释和预测数据。简单的分区方案比复杂方案更有可能泛化，亦即它最有可能预测和解释新数据。

``` 

当从主效应和交互作用的角度来考虑树时，树的一个有趣特性就会出现。注意 $\mathbb{E}(​​Y \mid \boldsymbol{X})$ 项等于所有终端节点参数 $\mu_{ij}$ 的和，因此：

- 当一棵树依赖于单个变量（ 如 {numref}`fig:decision_tree_reg` ）时，每个这样的 $\mu_{ij}$ 代表了一个主效应；

- 当一棵树依赖多个变量时（ 如 {numref}`fig:decision_tree` ），每个这样的 $\mu_{ij}$ 代表了一种交互效应。例如，请注意返回三角形需要 $X_1$ 和 $X_2$ 的交互，因为子节点的条件 ( $X_2 > c_2$ ) 基于父节点的条件 ( $X_1 > c_1$ ) 。 

树的大小可变，因此我们可以使用树来模拟不同阶次的交互效果。随着树变得更深，更多变量进入树的机会增加，然后表示更高阶交互的潜力也在增加。此外，因为使用了一组树，所以我们几乎可以构建主效果和交互效果的任意组合。

(ensembles-of-decision-trees)= 

### 7.1.1 决策树的集成 

考虑到过于复杂的树可能不太擅长预测新数据，因此通常会引入一些工具来降低决策树的复杂性，并获得更好地适应手头数据复杂性的拟合。其中一种解决方案依赖于拟合一组决策树，其中每棵树都被正则化为浅层树，因此，每棵树单独只能解释一小部分数据。只有通过组合许多这样的树，我们才能提供正确的答案。

`BART` 等贝叶斯方法和随机森林等非贝叶斯方法都遵循此集成策略。一般来说，集成模型可以降低泛化误差，同时保持拟合给定数据集的足够的灵活能力。

使用集成也有助于减轻阶跃性，因为输出是树的组合，虽然它仍然是一个阶跃函数，但它是一个具有更多阶跃的函数，因此在某种程度上是更平滑的近似。

只要我们确保树足够多样化，这就能称为事实。

使用树集成的一个缺点是失去了单个决策树的可解释性。现在要获得一个答案，我们不只是遵循一棵树，而是遵循许多树，这会混淆任何简单的解释。或者换句话说，我们用可解释性换取了灵活性和泛化能力。

(the-bart-model)= 

## 7.2 BART 模型 

如果我们假设方程 [eq:bfr](eq:bfr) 中的 $B_i$ 函数是决策树，我们可以这样写：

```{math} 
:label: eq:bart

\mathbb{E}[Y] = \phi \left(\sum_{j=0}^m g_j(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j), \theta \right)

```

其中每个 $g_j$ 是 $g(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j)$ 形式的树，其中 $\mathcal{T}_j$ 表示二叉树，即内部节点的集合及其相关的决策规则和终端节点的集合。

而 $\mathcal{M}_j = \{\mu_{1,j}, \mu_{2,j}, \cdots, \mu_{b, j} \}$ 表示 $b_j$ 终端节点的值，$\phi$ 表示任意概率分布，将用作我们模型中的似然性，$\phi$ 中的 $\theta$ 其他参数未建模为树的总和。

例如，我们可以将 $\phi$ 设置为高斯，然后我们将有：

```{math} 
Y = \mathcal{N}\left(\mu = \sum_{j=0}^m g_j(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j), \sigma \right)
```

或者我们可以像在 [ 第 3 章 ](chap2) 中对广义线性模型所做的那样，尝试其他分布。例如，如果 $\phi$ 是泊松分布，我们得到

```{math} 
Y = \text{Pois}\left(\lambda = \sum_{j}^m g_j(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j)\right)
```

或者 $\phi$ 是学生的 t 分布，那么：

```{math} 
Y = \text{T}\left(\mu = \sum_{j}^m g_j(\boldsymbol{X}; \mathcal{T}_j, \mathcal{M}_j), \sigma, \nu \right)
```

像往常一样，要完全指定 BART 模型，我们需要选择先验。我们已经熟悉高斯似然的 $\sigma$ 或学生 t 分布的 $\sigma$ 和 $\nu$ 的先前规范，因此现在我们将关注那些特定于 BART 模型的先验。

(priors-for-bart)= 

## 7.3 BART 模型的先验

原始 BART 论文 {cite:p}`ChipmanBARTBayesianadditive2010`，以及大多数后续修改和实现都依赖于共轭先验。

PyMC3 中的 BART 实现不使用共轭先验，并且在其他方​​面也存在偏差。我们将专注于 `PyMC3` 实现，而不是讨论差异，这是我们将用于示例的实现。

(prior-independence)= 

### 7.3.1 先验独立性 

为了简化先验的说明，我们假设树的结构 $\mathcal{T}_j$ 和叶值 $\mathcal{M}_j$ 是独立的。此外，这些先验独立于其他参数，即方程 {eq}`eq:bart` 中的 $\theta$。通过假设独立性，我们可以将先前的规范分成几部分。否则，我们应该设计一种方法来指定树空间上的单个先验 [^2]。

(prior-for-the-tree-structure-mathcalt_j)= 

### 7.3.2 树结构 $\mathcal{T}_j$ 的先验

树结构 $\mathcal{T}_j$ 的先验由三个方面指定：

- 深度 $d=(0, 1, 2, \dots)$ 的节点是非终结点的概率，由 $\alpha^{d}$ 给出。 $\alpha$ 建议为 $\in [0, 0.5)$ {cite:p}`Rockova2018` [^3]

- 分裂变量的分布。这就是树中包含的协变量（$X_i$ in {numref}`fig:decision_tree`）。最常见的是，这在可用协变量上是一致的。

- 分裂规则的分布。也就是说，一旦我们选择了一个分裂变量，我们使用哪个值来做出决定（$c_i$ in {numref}`fig:decision_tree`）。这通常与可用值一致。

(bart_mu_m_priors)= 

### 7.3.3 叶子结点 $\mu_{ij}$ 和树数目 $m$ 的先验

默认情况下，PyMC3 不会为叶值设置先验值，而是在采样算法的每次迭代中返回残差的均值。

关于集合 $m$ 中的树数。这通常也是由用户预定义的。在实践中已经观察到，通常通过设置 $m=200$ 甚至低至 $m=10$ 的值来获得良好的结果。此外，据观察，推断对于 $m$ 的确切值可能非常稳健。因此，一般的经验法则是尝试一些 $m$ 的值并执行交叉验证以选择最适合特定问题的值 [^4]。

(fitting-bayesian-additive-regression-trees)= 

## 7.4 拟合贝叶斯加性回归树 

到目前为止，我们已经讨论了如何使用决策树来编码分段函数，我们可以使用这些函数来建模回归或分类问题。我们还讨论了如何为决策树指定先验。我们现在将讨论如何有效地对树进行采样，以便找到给定数据集的树的后验分布。有很多策略可以做到这一点，而且细节对于本书来说太具体了。出于这个原因，我们将只描述主要元素。

为了拟合 BART 模型，我们不能使用像 Hamiltonian MonteCarlo 这样的基于梯度的采样器，因为树的空间是离散的，因此对梯度不友好。出于这个原因，研究人员开发了针对树木量身定制的 MCMC 和顺序蒙特卡洛 (SMC) 变体。 `PyMC3` 中实现的 BART 采样器以顺序和迭代的方式工作。简而言之，我们从一棵树开始并将其拟合到 $Y$ 结果变量，然后残差 $R$ 计算为 $R = Y - g_0(\boldsymbol{X}; \mathcal{T}_0, \数学{M}_0)$。第二棵树适合$R$，而不是$Y$。然后我们通过考虑到目前为止我们已经拟合的树的总和来更新残差 $R$，因此 $R - g_1(\boldsymbol{X}; \mathcal{T}_0, \mathcal{M}_0) + g_0( \boldsymbol{X}; \mathcal{T}_1, \mathcal{M}_1)$ 我们一直这样做，直到我们适合 $m$ 树。

此过程将导致后验分布的单个样本，一个具有 $m$ 树的样本。请注意，第一次迭代很容易导致次优树，主要原因是：第一次拟合的树会比必要的更复杂，树可能会陷入局部最小值，最后后面的树的拟合会受到影响以前的树。当我们继续采样时，所有这些影响将趋于消失，因为采样方法将多次重新访问以前拟合的树，并让它们有机会重新适应更新的残差。事实上，在拟合 BART 模型时，一个常见的观察结果是，在第一轮中，树往往更深，然后它们 *塌陷* 成较浅的树。

在文献中，特定的 BART 模型通常是为特定的采样器量身定制的，因为它们依赖于共轭性，因此具有高斯似然的 BART 模型与具有泊松似然的 BART 模型不同。 `PyMC3` 使用基于粒子 Gibbs 采样器 {cite:p}`Lakshminarayanan` 的采样器，该采样器专门用于处理树木。 `PyMC3` 会自动将此采样器分配给“pm.BART”分布，如果模型中存在其他随机变量，PyMC3 会将其他采样器（如 NUTS）分配给这些变量。

(bart_bike)= 

## 7.5 BART 自行车

让我们看看 BART 如何适合我们之前在 [5](chap3_5) 中研究的自行车数据集。该模型将是：

```{math} 
\begin{aligned}
\begin{split}
  \mu \sim& \; \text{BART}(m=50) \\
  \sigma \sim& \; \mathcal{HN}(1) \\
  Y \sim& \; \mathcal{N}(\mu, \sigma)
\end{split}\end{aligned}
```

在 `PyMC3` 中构建 BART 模型与构建其他类型的模型非常相似，不同之处在于指定随机变量 `pm.BART` 需要知道自变量和因变量。主要原因是用于拟合 BART 模型的采样方法在残差方面提出了一个新树，如上一节所述。

在做了所有这些澄清之后，PyMC3 中的模型如下所示：

```{code-block} ipython3
:name: bart_model_gauss
:caption: bart_model_gauss

with pm.Model() as bart_g:
    σ = pm.HalfNormal("σ", Y.std())
    μ = pm.BART("μ", X, Y, m=50)
    y = pm.Normal("y", μ, σ, observed=Y)
    idata_bart_g = pm.sample(2000, return_inferencedata=True)
```

在展示拟合模型的最终结果之前，我们将稍微探索一下中间步骤。这将使我们更直观地了解 BART 的工作原理。

{numref}`fig:bart_bikes_samples` 显示了从代码 [bart_model_gauss](bart_model_gauss) 中的模型计算的后验采样的树。在顶部，我们有 `m=50` 棵树中的三棵单独的树。树返回的实际值是实心点，线条是连接它们的视觉辅助。数据的范围（每小时租用的自行车数量）大约在每小时租用的 0-800 辆自行车的范围内。因此，即使这些数字省略了数据，我们也可以看到拟合相当粗糙，并且这些分段函数在数据规模上大多是平坦的。这是我们对树是*弱学习者*的讨论所预期的。鉴于我们使用了高斯似然，模型允许负计数值。

在底部子图上，我们有来自后验的样本，每个样本都是 $m$ 树的总和。

```{figure} figures/BART_bikes_samples.png
:name: fig:bart_bikes_samples
:width: 8.00in

后树实现。顶部子图，从后验取样的三棵树。底部子图，三个后验样本，每个样本都是 $m$ 树的总和。实际 BART 采样值由圆圈表示，而虚线是视觉辅助。小圆点（仅在底部子图中）表示观察到的租用自行车数量。

``` 

{numref}`fig:bart_bikes` 显示了将 BART 拟合到自行车数据集的结果（租用自行车的数量与一天中的小时数）。与使用样条线创建的 {numref}`fig:bikes_data2` 相比，该图提供了类似的拟合。与使用样条曲线获得的拟合相比，BART 拟合的锯齿状特征越明显，差异越明显。

这并不是说没有其他差异，例如 HDI 的宽度。

```{figure} figures/BART_bikes.png
:name: fig:bart_bikes
:width: 8.00in

使用 BART（特别是 `bart_model`）拟合的自行车数据（黑点）。
阴影曲线代表 $94\%$ HDI 区间（均值），蓝色曲线代表平均趋势。与 {numref}`fig:bikes_data2` 进行比较。

``` 

围绕 BART 的文献倾向于强调其通常无需调整即可提供有竞争力的答案的能力 [^5]。例如，与拟合样条曲线相比，我们无需担心手动设置节点或选择先验来调整节点。当然，有人可能会争辩说，对于某些问题，能够调整结对手头的问题可能是有益的，这很好。

(generalized-bart-models)= 

## 7.6 广义 BART 模型

BART 的 `PyMC3` 实现试图使使用不同的可能性 [^6] 变得容易，类似于我们在 [ 第 3 章 ](chap2) 中看到的广义线性模型所做的那样。让我们看看如何在 BART 中使用伯努利似然。对于这个例子，我们将使用太空流感疾病的数据集，该疾病主要影响年轻人和老年人，但不影响中年人。幸运的是，太空流感并不是一个严重的问题，因为它已经完全弥补了。在这个数据集中，我们记录了接受太空流感检测的人，以及他们是生病 (1) 还是健康 (0) 以及他们的年龄。使用来自代码 [bart_model_gauss](bart_model_gauss) 的具有高斯似然的 BART 模型作为参考，我们看到差异很小：

```{code-block} ipython3
:name: bart_model_bern
:caption: bart_model_bern

with pm.Model() as model:
    μ = pm.BART("μ", X, Y, m=50,
                inv_link="logistic")
    y = pm.Bernoulli("y", p=μ, observed=Y)
    trace = pm.sample(2000, return_inferencedata=True)
```

首先，我们不再需要定义 $\sigma$ 参数，因为伯努利分布只有一个参数“p”。对于 BART 本身的定义，我们有一个新参数，`inv_link`，这是反向链接函数，我们需要将 $\mu$ 的值限制在区间 $[0, 1]$ 内。为此，我们指示 `PyMC3` 使用逻辑函数，就像我们在 [ 第 3 章 ](chap2) 中对逻辑回归所做的那样）。

{numref}`fig:BART_space_flu_comp` 显示了代码 [bart_model_bern](bart_model_bern)  中的模型与 $m$ 的 4 个值的比较，即 (2, 10, 20, 50) 使用 LOO。 {numref}`fig:BART_space_flu_fit` 显示数据加上拟合函数和 HDI $94\%$ 波段。我们可以看到，根据 LOO，$m=10$ 和 $m=20$ 提供了很好的拟合。这与目视检查在定性上一致，因为 $m=2$ 明显欠拟合（ELPD 的值很低，但样本内和样本外 ELPD 之间的差异并不大）和 $ m=50$ 似乎过拟合（ELPD 的值很低，样本内和样本外 ELPD 之间的差异很大）。

```{figure} figures/BART_space_flu_comp.png
:name: fig:BART_space_flu_comp
:width: 8.00in

代码 [bart_model_bern](bart_model_bern) 中的模型与 $m$ 值（2、10、20、50）的 LOO 比较。根据 LOO，$m=10$ 提供了最佳拟合。

``` 

```{figure} figures/BART_space_flu_fit.png
:name: fig:BART_space_flu_fit
:width: 8.00in

BART 适合具有 4 个 $m$ 值的 Space Influenza 数据集（2、10、20、50）。与 LOO 一致，具有 $m$ 的模型是欠拟合的，而具有 $m=50$ 的模型是过拟合的。

``` 

到目前为止，我们已经讨论了具有单个协变量的回归，我们这样做是为了简单起见。但是，可以使用更多协变量来拟合数据集。从 `PyMC3` 的实现角度来看，这是微不足道的，我们只需要传递一个包含超过 1 个协变量的 $X$ 二维数组。但它提出了一些有趣的统计问题，例如如何轻松解释具有许多协变量的 BART 模型，或者如何找出每个协变量对结果的贡献程度。在接下来的部分中，我们将展示这是如何完成的。

(interpretability-of-barts)= 

## 7.7 BART 的可解释性

单个决策树通常很容易解释，但是当我们将一堆树加在一起时，情况就不再正确了。有人可能认为原因是通过添加树我们得到了一些奇怪的无法识别或难以表征的对象，但实际上树的总和只是另一棵树。解释这个*组装*树的困难在于，对于一​​个复杂的问题，决策规则将很难掌握。这就像在钢琴上弹奏一首歌曲，弹奏单个音符相当容易，但以一种悦耳的方式弹奏组合的音符，既能带来丰富的声音，又能带来个性化解释的复杂性。

我们仍然可以通过直接检查树的总和来获得一些有用的信息（参见第 {ref}`sec:variable_selection` 部分，但不如使用更简单的单个树那样透明或有用。因此，为了帮助我们解释 BART 模型的结果，我们通常依赖模型诊断工具 {cite:p}`Molnarbook, Molnar2020`，例如也用于多元线性回归和其他非参数方法的工具。我们将在下面讨论两个相关工具：**部分依赖图**（PDP）{ cite:p}`Friedman2001` 和 **Individual Conditional Expectation** (ICE) 图 {cite:p}`Goldstein2014`。


(partial-dependence-plots)= 

### 7.7.1 部分依赖图 

BART 文献中出现的一种非常常见的方法是所谓的部分依赖图 (PDP) {cite:p}`Friedman2001`（参见 {numref}`fig:pdp_fake_example`）。 PDP 显示了当我们更改协变量同时对其余协变量的边际分布进行平均时，预测变量的值如何变化。也就是说，我们计算然后绘制：

```{math} 
:label: eq:partial_dependence

\tilde{Y}_{\boldsymbol{X}_i}= \mathbb{E}_{\boldsymbol{X}_{-i}}[\tilde{Y}(\boldsymbol{X}_i, \boldsymbol{X}_{-i})] \approx \frac{1}{n}\sum_{j=1}^{n} \tilde{Y}(\boldsymbol{X}_i, \boldsymbol{X}_{-ij})

```

其中 $\tilde{Y}_{\boldsymbol{X}_i}$ 是预测变量的值，作为 $\boldsymbol{X}_i$ 的函数，而除了 $i$ ($\boldsymbol{X }_{-i}$) 已被边缘化。通常 $X_i$ 将是 1 或 2 个变量的子集，原因是在更高维度上绘图通常很困难。

如方程 {eq}`eq:partial_dependence` 所示，期望可以通过对以观察到的 $\boldsymbol{X}_{-i}$ 为条件的预测值进行平均来在数值上近似。但是请注意，这意味着 $\boldsymbol{X}_i、\boldsymbol{X}_{-ij}$ 中的某些组合可能与实际观察到的组合不对应。此外，甚至可能无法观察到某些组合。这类似于我们已经讨论过的关于 [ 第 3 章 ](chap2) 中介绍的反事实图。事实上，部分依赖图是一种反事实装置。

```{figure} figures/partial_dependence_plot.png
:name: fig:pdp_fake_example
:width: 8.00in

部分依赖图。每个变量 $X_i$ 对 $Y$ 的部分贡献，同时边缘化其余变量 ($X_{-i}$) 的贡献。灰色带代表 HDI $94\%$。均值和 HDI 波段都已被平滑（参见 `plot_ppd` 函数）。地毯图（每个子图底部的黑条）显示了每个协变量的观察值。

``` 

{numref}`fig:pdp_fake_example` 显示了将 BART 模型拟合到合成数据后的 PDP：$Y \sim \mathcal{N}(0, 1)$ $X_{0} \sim \mathcal{N}(Y, 0.1)$ 和 $X_{1} \sim \mathcal{N}(Y, 0.2)$ $X_{2} \sim \mathcal{N}(0, 1)$。我们可以看到 $X_{0}$ 和 $X_{1}$ 都与 $Y$ 呈现线性关系，正如合成数据的生成过程所预期的那样。我们还可以看到，与 $X_{1}$ 相比，$X_{0}$ 对 $Y$ 的影响更强，因为 $X_{0}$ 的斜率更陡峭。

因为协变量尾部的数据稀疏（它们是高斯​​分布的），这些区域显示出更高的不确定性，这是所期望的。最后，$X_{2}$ 的贡献在变量 $X_{2}$ 的整个范围内几乎可以忽略不计。

现在让我们回到自行车数据集。这次我们将使用四个协变量对租用自行车的数量（预测变量）进行建模；一天中的小时、温度、湿度和风速。

{numref}`fig:partial_dependence_plot_bikes` 显示拟合模型后的部分依赖图。我们可以看到一天中小时的部分依赖图看起来与 {numref}`fig:bart_bikes` 非常相似，我们通过在没有其他变量的情况下拟合这个变量获得的。随着温度的升高，租用自行车的数量也增加了，但在某些时候这种趋势趋于平稳。

使用我们的外部领域知识，我们可以推测这种模式是合理的，因为当温度太低时人们不会太有动力骑自行车，但在*太高*的温度下骑自行车也不太吸引人。湿度呈平缓趋势，随后出现负增长，我们再次可以想象为什么较高的湿度会降低人们骑自行车的动力。风速显示出更平坦的贡献，但我们仍然看到了效果，因为似乎更少的人倾向于在大风条件下租用自行车。

```{figure} figures/partial_dependence_plot_bikes.png
:name: fig:partial_dependence_plot_bikes
:width: 8.00in

部分依赖图。变量、小时、温度、湿度和风速对租用自行车数量的部分贡献，同时边缘化其余变量的贡献（$X_{-i}$）。灰色带代表 HDI $94\%$。均值和 HDI 波段均已平滑（请参阅 `plot_ppd` 函数）。地毯图（每个子图底部的黑条）显示了每个协变量的观察值。

``` 

计算部分依赖图时的一个假设是变量 $X_i$ 和 $X_{-i}$ 不相关，因此我们在边际上执行均值。在大多数实际问题中，情况并非如此，然后部分依赖图可以隐藏数据中的关系。

然而，如果所选变量的子集之间的依赖性不是太强，那么部分依赖性图可能是有用的总结 {cite:p}`Friedman2001`。

::: {admonition} 部分依赖的计算成本

计算部分依赖图的计算要求很高。

因为在我们想要评估变量 $X_i$ 的每一点，我们都需要计算 $n$ 预测（其中 $n$ 是样本大小）。

为了 BART 获得预测 $\tilde Y$，我们需要首先对 $m$ 树求和以获得 $Y$ 的点估计，然后我们还要对树总和的整个后验分布进行平均以获得可信区间。这最终需要相当多的计算！如果需要，减少计算的一种方法是在 $p$ 点处以 $p << n$ 评估 $X_i$。我们可以选择 $p$ 等间距的点，也可以选择一些分位数。或者，如果我们将它们固定在它们的均值上，而不是边缘化 $\boldsymbol{X}_{-ij}$，我们可以实现显着的加速。当然，这意味着我们将丢失信息，并且均值可能实际上并不能很好地代表基础分布。另一个对大型数据集特别有用的选项是对 $\boldsymbol{X}_{-ij}$ 进行二次采样。

:::

(individual-conditional-expectation)= 

### 7.7.2 个体条件期望

个体条件期望 (ICE) 图与 PDP 密切相关。不同之处在于，我们没有绘制目标协变量对预测响应的平均偏效应，而是绘制了 $n$ 估计的条件期望曲线。也就是说，ICE 图中的每条曲线都将部分预测响应反映为固定值 $\boldsymbol{X}_{-ij}$ 的协变量 $\boldsymbol{X}_{i}$ 的函数。有关示例，请参见 {numref}`fig:individual_conditional_expectation_plot_bikes`。如果我们在每个 $\boldsymbol{X}_{ij}$ 值处平均所有灰色曲线，我们会得到蓝色曲线，如果我们计算了 {numref} 中的平均部分相关性，我们应该得到相同的曲线。图：partial_dependence_plot_bikes`。

```{figure} figures/individual_conditional_expectation_plot_bikes.png
:name: fig:individual_conditional_expectation_plot_bikes
:width: 8.00in

个体条件期望图。变量对租用自行车数量的部分贡献；小时、温度、湿度和风速，同时将其余部分 ($X_{-i}$) 固定在一个观察值。
蓝色曲线对应于灰色曲线的均值。所有曲线都经过平滑处理（参见 `plot_ice` 函数）。地毯图（每个子图底部的黑条）显示了每个协变量的观察值。

``` 

单个条件期望图最适合变量具有强交互作用的问题，当情况并非如此时，部分依赖图和单个条件期望图传达相同的信息。 {numref}`fig:pdp_vs_ice_toy` 显示了一个示例，其中部分依赖图隐藏了数据中的关系，但单个条件期望图能够更好地显示它。该图是通过将 BART 模型拟合到合成数据中生成的： $Y = 0.2X_0 - 5X_1 + 10X_1 \unicode{x1D7D9}_{X_2 \geq 0} + \epsilon$ 其中 $X \sim \mathcal{U}( -1, 1)$\epsilon \sim \mathcal{N}(0, 0.5)$。

请注意 $X_1$ 的值如何取决于 $X_2$ 的值。

```{figure} figures/pdp_vs_ice_toy.png
:name: fig:pdp_vs_ice_toy
:width: 8.00in

部分依赖图与个体条件期望图。第一个子图，$X_1$ 和 $Y$ 之间的散点图，中间子图部分依赖图，最后一个子图个体条件期望图。

``` 

在 {numref}`fig:pdp_vs_ice_toy` 的第一个子图中，我们绘制了 $X_1$ 与 $Y$。考虑到 $Y$ 的值可以根据 $X_2$ 变量的值随 $X_1$ 线性增加或减少的交互效应，该图显示了 *X 形* 模式。中间子图显示了一个部分依赖图，我们可以看到，根据这个图，关系是平坦的，*平均*是正确的，但隐藏了交互作用。相反，最后一个子图，一个单独的条件期望图有助于揭示这种关系。原因是每条灰色曲线代表一个$X_{0,2}$ [^7] 的值。

蓝色曲线是灰色曲线的均值，虽然与部分相关平均曲线不完全相同，但它显示了相同的信息 [^8]。

(sec:variable_selection)= 

## 7.7 变量选择 

当用多个预测变量拟合回归时，了解哪些预测变量最重要通常很有意义。在某些情况下，我们可能真的有兴趣更好地了解不同变量如何促成特定输出。例如，哪些饮食和环境因素会导致结肠癌。在其他情况下，收集具有许多协变量的数据集可能在经济上无法负担、花费太长时间或逻辑上过于复杂。例如，在医学研究中，测量来自人类的大量变量可能是昂贵的、耗时的或烦人的（甚至对患者来说是有风险的）。因此，我们有能力在试点研究中测量很多变量，但要将此类分析扩展到更大的人群，我们可能需要减少变量的数量。在这种情况下，我们希望保留最小（最便宜，更方便获得）的变量集，这些变量仍然提供合理的高预测能力。 BART 模型提供了一种非常简单且几乎无需计算的启发式方法来估计变量的重要性。它跟踪协变量被用作分裂变量的次数。例如，在 {numref}`fig:decision_tree` 中有两个分裂节点，一个包括变量 $X_1$ 和另一个 $X_2$，因此基于这棵树，两个变量同等重要。如果相反，我们将计算 $X_1$ 两次和 $X_2$ 一次。我们会说 $X_1$ 的重要性是 $X_2$ 的两倍。对于 BART 模型，变量重要性是通过对 $m$ 树和所有后验样本进行平均来计算的。请注意，使用这个简单的启发式方法，我们只能以相对方式报告重要性，因为没有简单的方法可以说这个变量很重要，而另一个不重要。

为了进一步简化解释，我们可以报告归一化的值，因此每个值都在区间 $[0, 1]$ 内，总重要性为 1。很容易将这些数字解释为后验概率，但我们应该记住，这只是一个简单的启发式，没有非常强大的理论支持，或者用更细微的术语来说，它还没有被很好地理解 {cite:p}`Liu2020`。

{numref}`fig:bart_vi_toy` 显示了来自已知生成过程的 3 个不同数据集的相对变量重要性。

- $Y \sim \mathcal{N}(0, 1)$ $X_{0} \sim \mathcal{N}(Y, 0.1)$ 和 $X_{1} \sim \mathcal{N}(Y, 0.2)$\boldsymbol{X}_{2:9}\sim \mathcal{N}(0, 1)$. 只有前 2 个自变量与预测变量无关，第一个比第二个更相关。

- $Y = 10 \sin(\pi X_0 X_1 ) + 20(X_2 - 0.5)^2 + 10X_3 + 5X_4 + \epsilon$ 其中 $\epsilon \sim \mathcal{N}(0, 1)$ 和 $\ boldsymbol{X}_{0:9} \sim \mathcal{U}(0, 1)$ 这通常被称为弗里德曼的五维测试函数 {cite:p}`Friedman2001`。

请注意，虽然前 5 个随机变量与 $Y$ 相关（不同程度地），但后 5 个则不相关。

- $\boldsymbol{X}_{0:9} \sim \mathcal{N}(0, 1)$ 和 $Y \sim \mathcal{N}(0, 1)$。所有变量都与结果变量无关。

```{figure} figures/bart_vi_toy.png
:name: fig:bart_vi_toy
:width: 8.00in

相对变量重要性。左子图，前 2 个输入变量对预测变量有贡献，其余是噪声。中间子图，前 5 个变量与输出变量有关。最后在右侧子图上，10 个输入变量与预测变量完全无关。如果所有变量都同等重要，黑色虚线表示变量重要性的值。

``` 

我们可以从 {numref}`fig:bart_vi_toy` 中看到一件事是增加树的数量 $m$ 的效果。一般来说，随着我们增加 $m$，相对重要性的分布往往会变得*平坦*。这是一个众所周知的观察结果，具有直观的解释。随着我们增加 $m$ 的值，我们要求每棵树的预测能力降低，这意味着不太相关的特征更有可能成为给定树的一部分。相反，如果我们减少 $m$ 的值，我们对每棵树的要求会更高，这会导致变量之间更严格的*竞争*成为树的一部分，因此只有*真正重要的*变量才会成为包含在最终的树中。

{numref}`fig:bart_vi_toy` 之类的图可用于帮助将更重要的变量与不太重要的变量分开 {cite:p}`ChipmanBARTBayesianadditive2010, Carlson2020`。这可以通过查看当我们从 $m$ 的低值转移到更高的值时会发生什么来完成。如果相对重要性降低，则变量*更重要*，如果变量重要性增加，则变量*不那么重要*。例如，在第一个子图中，很明显对于不同的 $m$ 值，前两个变量比其他变量重要得多。对于前 5 个变量，可以从第二个子图得出类似的结论。在最后一个子图上，所有变量都同样（不）重要。

这种评估变量重要性的方法可能很有用，但也很棘手。

在某些情况下，为变量重要性设置置信区间会有所帮助，而不仅仅是点估计。我们可以通过使用相同的参数和数据多次运行 BART 来做到这一点。

然而，缺乏将重要变量与不重要变量分开的明确阈值可能被视为有问题。已经提出了一些替代方法 {cite:p}`Carlson2020, Bleich2014`。其中一种方法可以总结如下：

1. 使用 $m$ 的小值（例如 25 [^9]）多次拟合模型（大约 50 次）。记录均方根误差。

2. 消除所有 50 次运行中信息最少的变量。

3. 重复 1 和 2，每次在模型中少一个变量。一旦达到模型中给定数量的协变量（不一定是 1），就停止。

4. 最后，选择平均均方根误差最小的模型。

根据 Carlson {cite:p}`Carlson2020` 的说法，这个过程似乎总是返回与创建像 {numref}`fig:bart_vi_toy` 这样的图形相同的结果。然而，人们可以争辩说这是更自动的（具有自动决策的所有优点和缺点）。也没有什么能阻止我们进行自动程序，然后将绘图用作视觉检查。

让我们转向具有四个协变量的租赁自行车示例：小时、温度、湿度和风速。从 {numref}`fig:bart_vi_bikes` 我们可以看到小时和温度与预测租用自行车的数量比湿度或风速更相关。我们还可以看到，变量重要性的顺序与部分依赖图（ 图 {numref}`fig:partial_dependence_plot_bikes` ）和个体条件期望图（ 图 {numref}`fig:individual_conditional_expectation_plot_bikes` ）的结果定性一致。


```{figure}  figures/bart_vi_bikes.png 
:name:   fig:bart_vi_bikes 
:width:  8.00in 
 
具有不同树数的拟合 BART 的相对变量重要性。小时是最重要的协变量，其次是温度。湿度和风速似乎是不太相关的协变量。
``` 

(priors-for-bart-in-pymc3)= 

## 7.8 `PyMC3` 中 BART 先验的选择

与本书中的其他模型相比，BART 是最“黑盒”的。

我们无法设置我们想要生成 BART 模型的任何先验。

相反，我们通过一些参数来控制预定义的先验。 `PyMC3` 允许使用 3 个参数控制 BARTS 的先验：

- 树的数量 $m$

- 树的深度$\alpha$

- 分裂变量的分布。

我们看到了改变树的数量的效果，这已被证明可以为 50-200 区间内的值提供可靠的预测。还有很多例子表明使用交叉验证来确定这个数字可能是有益的。我们还看到，通过扫描 $m$ 寻找相对较低的值，例如 25-100 范围内的值，我们可以评估变量的重要性。我们没有费心去改变 $\alpha=0.25$ 的默认值，因为这个改变似乎影响更小，尽管仍然需要研究来更好地理解这个先前的 {cite:p}`Rockova2018`。

与 $m$ 一样，交叉验证也可用于调整它以提高效率。最后，PyMC3 提供了传递权重向量的选项，因此不同的变量具有不同的先验概率被选中，当用户有证据表明某些变量可能比其他变量更重要时，这可能很有用，否则最好保持统一。已经提出了更复杂的基于 Dirichlet 的先验 [^10] 来实现这一目标，并在需要诱导稀疏性时允许更好的推理。这在我们有很多协变量的情况下很有用，但只有少数可能有贡献，而且我们事先不知道哪些是最相关的。这是一个常见的情况，例如，在基因研究中，测量数百或更多基因的活性相对容易，但它们之间的关系不仅未知，而且是研究的目标。

大多数 BART 实施都是在单个软件包的上下文中完成的，在某些情况下甚至面向特定的子学科。

它们通常不是概率编程语言的一部分，因此不希望用户过多地调整 BART 模型。因此，即使可以将先验直接放在树的数量上，但在实践中通常不是这样做的。相反，BART 文献赞扬了 BART 在默认参数下的良好性能，同时认识到可以使用交叉验证来获得额外的收益。 `PyMC3` 中的 BART 实现稍微偏离了这一传统，并允许一些额外的灵活性，但与我们使用其他像高斯或泊松分布，甚至像高斯过程这样的非参数分布相比，仍然非常有限。我们预计这可能会在不远的将来发生变化，部分原因是我们有兴趣探索更灵活的 BART 实现，这可以允许用户构建灵活和针对问题的模型，这通常是概率编程语言的情况。

(exercises7)= 

## 7.9 练习 

**7E1.** Explain each of the following 

1.  How is BART different from linear regression and splines.

2. When you may want to use linear regression over BART? 

3.  When you may want to use splines over BART? 

**7E2.** Draw at least two more trees that could be used to explain the data in {numref}`fig:decision_tree`.

**7E3.** Draw a tree with one more internal node than the one in {numref}`fig:decision_tree` that explains the data equally well.

**7E4.** Draw a decision tree of what you decide to wear each morning. Label the leaf nodes and the root nodes.

**7E5.** What are the priors required for BART? Explain what is the role of priors for BART models and how is this similar and how is this different from the role of priors in the models we have discussed in previous chapters.

**7E6.** In your own words explain why it can be the case that multiple small trees can fit patterns better than one single large tree. What is the difference in the two approaches? What are the tradeoffs? 

**7E7.** Below we provide some data. To each data fit a BART model with m=50. Plot the fit, including the data. Describe the fit.

1. `x = np.linspace(-1, 1., 200)` and `y = np.random.normal(2*x, 0.25)` 

2.  `x = np.linspace(-1, 1., 200)` and   `y = np.random.normal(x**2, 0.25)` 

3.  pick a function you like 

4.  compare the results with the exercise **5E4.** from Chapter [5](chap3_5) 

**7E8.** Compute the PDPs For the dataset used to generate {numref}`fig:bart_vi_toy`. Compare the information you get from the variable importance measure and the PDPs.

**7M9**. For the rental bike example we use a Gaussian as likelihood, this can be seen as a reasonable approximation when the number of counts is large, but still brings some problems, like predicting negative number of rented bikes (for example, at night when the observed number of rented bikes is close to zero). To fix this issue and improve our models we can try with other likelihoods: 

1.  use a Poisson likelihood (hint you will need to use an inverse link   function, check `pm.Bart` docstring). How the fit differs from the   example in the book. Is this a better fit? In what sense? 

2.  use a NegativeBinomial likelihood, how the fit differs from the   previous two? Could you explain the result.

3. how this result is different from the one in Chapter   [5](chap3_5)? Could you explain the difference? 

**7M10.** Use BART to redo the first penguin classification examples we performed in Section {ref}`classifying_penguins` (i.e. use "bill_length_mm" as covariate and the species "Adelie" and "Chistrap" as the response). Try different values of `m` like, 4, 10, 20 and 50 and pick a suitable value as we did in the book. Visually compare the results with the fit in {numref}`fig:Logistic_bill_length`. Which model do you think performs the best? 

**7M11.** Use BART to redo the penguin classification we performed in Section {ref}`classifying_penguins`. Set `m=50` and use the covariates "bill_length_mm", "bill_depth_mm", "flipper_length_mm" and "body_mass_g".

 Use Partial Dependence Plots and Individual Conditional Expectation. To find out how the different covariates contribute the probability of identifying "Adelie", and "Chinstrap" species.

 Refit the model but this time using only 3 covariates "bill_depth_m", "flipper_length_mm", and "body_mass_g". How results differ from using the four covariates? Justify.

**7M12.** Use BART to redo the penguin classification we performed in Section {ref}`classifying_penguins`. Build a model with the covariates "bill_length_mm", "bill_depth_mm", "flipper_length_mm", and "body_mass_g" and assess their relative variable importance. Compare the results with the PDPs from the previous exercise.

 ---

[^1]: Maybe you have heard about its non-Bayesian cousin: Random Forest   {cite:p}`BreimanForests2001` 

[^2]: for alternatives see   {cite:p}`BalogMondrianProcessMachine2015, royMondrianProcess` 

[^3]: Node depth is defined as distance from the root. Thus, the root   itself has depth 0, its first child node has depth 1, etc.

[^4]: In principle we can go fully Bayesian and estimate the number of   tree $m$ from the data, but there are reports showing this is not   always the best approach. More research is likely needed in this   area.

[^5]: The same literature generally shows that using cross-validation to   tune the number of trees and/or the prior over the depth of the tree   can be further beneficial.

[^6]: Other implementations are less flexible or require adjustments   under the hood to make this work.

[^7]: This notation means the variables ($X_0$, $X_2$), that is,   excluding $X_1$ 

[^8]: The mean of the ICE curves and the mean partial dependence curve   are slightly different. This is due to internal details on how these   plots were made including the order in which we average over the   posterior samples or over the observations. What really matter is   the general features, for instance in this case that both curves are   essentially flat. Also, to speed up computation we evaluate $X_1$   over 10 equally separated points for partial dependence plots and we   subsample $X_{0,2}$ for computing the individual conditional   expectation plot 

[^9]: The original proposal suggests 10, but our experience with the   BART implementation in PyMC3 is that values of $m$ below 20 or 25   could be problematic.

 [^10]: This is likely to be added in the future versions of PyMC3.

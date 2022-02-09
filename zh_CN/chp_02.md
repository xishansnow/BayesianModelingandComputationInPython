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

(chap1bis)= 

# 第二章: 贝叶斯模型的探索性分析 

<style>p{text-indent:2em;2}</style>

正如我们在第 [1](chap1) 章中所见，**贝叶斯推断**使用可用数据对模型进行条件化并获得后验分布。我们可以使用笔和纸、计算机或其他设备 [^1] 来实现推断；此外，推断过程通常还包括一些其他量的计算，例如先验预测分布和后验预测分布。但是，**贝叶斯建模**相对于贝叶斯推断而言更为广泛。我们通常希望贝叶斯建模能够仅仅靠指定模型和计算后验就能实现，但通常情况下并非如此。现实情况是，成功的贝叶斯数据分析需要完成许多其他同等重要的任务。

在本章中，我们将讨论其中一些任务，包括：**模型假设的检查**、**模型推断结果的诊断** 和 **模型的比较**。

(there-is-life-after-inference-and-before-too)= 

## 2.1 “贝叶斯建模”大于“贝叶斯推断” 

成功的贝叶斯建模方法除了贝叶斯推断之外，还需要执行其他额外的任务 [^2]。

典型如：

- 模型诊断，对使用数值方法获得的推断结果进行诊断，评估其质量。

- 模型评判，包括对模型假设和模型预测的评估。

- 模型比较，包括模型选择或模型平均。

- 模型沟通，为特定受众准备结果。

实现上述任务需要一些数字汇总和可视化手段来帮助从业者对模型进行分析，我们称此类方法为**贝叶斯模型的探索性分析**。此名称源于统计方法中的探索性数据分析 (`EDA`) {cite:p}`tukey77` 。该分析方法旨在汇总数据集的主要特征，并且通常使用可视化方法。用 `Persi Diaconis` 的话来说 {cite:p}`Diaconis2011` ：

> 探索性数据分析 (`EDA`) 旨在揭示数据中的结构或简单描述。人们查看数字或图表并尝试找到其中蕴含的模式。人们追求由背景信息、想象力、感知模式和其他数据的经验所蕴含的线索分析。

`EDA` 通常在推断步骤之前执行，有时甚至代替推断步骤。我们以及之前许多研究者 {cite:p}`gabry_visualization_2017, Gelman2020` 认为，`EDA` 中的许多想法都可以被使用、重新解释，并扩展为强大的贝叶斯建模方法。

在本书中将主要使用 `Python` 的 `ArviZ` 库 [^3] 来帮助我们对贝叶斯模型进行探索性分析。

在现实生活中，贝叶斯推断步骤和模型的探索性分析步骤经常交织在一个迭代的工作流中，其中可能还包括编码错误、计算问题、对模型充分性的怀疑、对我们当前对数据的理解的怀疑、非线性模型构建、模型检查等很多方面。试图在一本书中描述这种复杂的工作流非常具有挑战性，而且也不是本书的重点。因此，我们可能会省略部分甚至全部探索性分析步骤，或者将其留作练习。这并非因为探索性分析没有必要或不重要；相反，它非常地重要，在编写本书过程中，我们实际上在 “幕后” 进行了大量地迭代工作。但我们也确实在某些地方省略了它们，以便将重点放在其他注意力方面，例如模型细节、计算特征或基础数学。

(prior_predictive_checks)= 

## 2.2 理解你的先验假设 

正如在 {ref}`make_prior_count` 节中所讨论的，“什么是最好的先验？” 是一个很诱人的话题。然而，除了 “这取决于 ？” 之外，很难给出一个直接令人满意的答案。我们当然可以找到给定模型或模型族的默认先验，将其推广到广泛的数据集，并产生良好结果。但如果能够为特定问题生成更多信息性先验，那么一定也能够找到在特定问题上优于它们的方法。事实上，良好的缺省先验不仅可以作为快速分析的基础，当我们可以投入时间和精力进入迭代的探索性贝叶斯建模工作流程时，它也可以作为更好先验的一个占位符。

选择先验时的一个问题是：当先验按照模型传导到数据中时，有时候很难理解其导致的效应。我们在参数空间中所作的选择，可能会在观测数据空间中引发一些意想不到的结果。因此，为了更好地理解假设，我们需要能够查看其结果的工具，这个工具就是**先验预测分布**，我们在第 {ref}`Bayesian_inference` 和公式 [eq:prior_pred_dist](eq:prior_pred_dist) 中提到过它。

在实际工作中，我们在假设先验的条件下，通过从模型中采样来获得（计算）先验预测分布，而这种预测无需以观测数据为条件。这些先验预测分布的样本，将我们在参数空间中所做的选择转换成了观测数据空间中的预测结果（很有可能是错误的）。我们将生成这些样本，并用这些样本来评估先验的过程，称为 **先验预测检查（ Prior Predictive Check ）**。

假设我们想要建立一个足球模型。具体来说，我们对从罚球点射门的进球概率感兴趣。经过一段时间思考，我们决定使用几何模型来建模 [^4]。根据 {numref}`fig:football_sketch` 中的草图和一点三角函数知识，可以得出以下进球概率公式：

```{math}
:label: eq:geometric_football
p\left(|\alpha| < \tan^{-1}\left(\frac{L}{x}\right)\right) = 2\Phi\left(\frac{\tan^{-1}\left(\frac{L}{x}\right)}{\sigma}\right) - 1

```

公式 {eq}`eq:geometric_football` 背后的直觉是，假设进球概率由角度 $\alpha$ 的绝对值小于某个阈值 $\tan^{-1}\left(\frac{L}{x}\right)$ 的条件决定；此外，假设球员试图将球踢直（即以零角度），但存在其他因素导致的轨迹偏差 $\sigma$。

```{figure} figures/football_sketch.png
:name: fig:football_sketch
:width: 4in

罚球示意草图。虚线表示进球得分所必须的角度 $\alpha$ 。 $x$ 代表罚球距离（ 11 米），$L$ 代表球门长度的一半（ 3.66 米）。
```

公式 {eq}`eq:geometric_football` 中唯一的未知量是 $\sigma$ ，而 $L$ 和 $x$ 的值都可以从足球规则中得到。作为贝叶斯工作者，当我们不知道一个量时，通常会为其分配一个先验，然后尝试建立贝叶斯模型去估计它。例如，我们可以这样写：

```{math}
:label: eq:geometric_model

\begin{split}
\sigma &= \mathcal{HN}(\sigma_{\sigma}) \\
\text{p_goal} &= 2\Phi\left(\frac{\tan^{-1}\left(\frac{L}{x}\right)}{\sigma}\right) - 1 \\
Y &= \text{Bin}(n=1, p=\text{p_goal})

\end{split}
```

现在我们尚不完全确定模型对足球领域知识的编码程度如何，因此可以从先验预测中采样以获得一些直觉。

{numref}`fig:prior_predictive_check_00` 显示了三个先验样本（编码为 $\sigma_{\sigma}$ 的三个值：$5$、$20$ 和 $60$ 度）对应的结果。灰色的圆形区域代表应该导致进球的一组角度，假设球员踢球完全笔直并且没有其他因素（如风、摩擦等）影响。可以看到，模型假设即使球员以比灰色区域更大的角度射门，也有可能进球。有趣的是，对于较大的 $\sigma_{\sigma}$ 值，模型认为朝相反方向踢并不一定是坏事。

```{figure} figures/prior_predictive_distributions_00.png
:name: fig:prior_predictive_check_00
:width: 8.00in

公式 {eq}`eq:geometric_model` 中模型的先验预测检查。每个子图对应于 $\sigma$ 的不同先验值。每个圆形图中心的黑点代表罚分。边缘处的点代表射门，其位置是角度 $\alpha$ 的值（参见 {numref}`fig:football_sketch`），颜色代表进球的概率。
```

现在我们有几个选择：一是可以重新考虑模型以结合更多的几何性质；二是可以调整先验来减少无意义结果的机会，即使我们并没有完全排除它们；三是直接用当前先验来拟合数据，并查看数据是否具备足够信息来得到（能够排除无意义值的）后验。

{numref}`fig:prior_predictive_check_01` 显示了另一个我们可能认为例外的示例 [^5] 。该示例显示了一个预测变量为二值类型的逻辑斯蒂回归 [^6] 以及其回归系数上的先验 $\mathcal{N}(0, 1)$ 。随着我们增加预测变量的数量，先验预测分布的平均值从更集中在 $0.5$ 左右（第一个子图）变为均匀（中间），以支持极端值 $0$ 或 $1$（最后一个子图）。这个例子向我们展示了：随着预测变量数量的增加，先验预测分布会将更多质量放在极值上。因此，我们可能需要一个 *更强的正则化先验* ，以使模型远离那些不太可能发生的极值。

```{figure} figures/prior_predictive_distributions_01.png
:name: fig:prior_predictive_check_01
:width: 8.00in

具有 $2$ 个、$5$ 个或 $15$ 个二值预测变量和 $100$ 个数据点的逻辑斯蒂回归模型的先验预测分布。 `KDE` 表示模拟数据在 $10000$ 次模拟中的均值分布。即使每个系数的先验 $\mathcal{N}(0, 1)$ 对于所有 $3$ 个子图都相同，增加预测变量数量的增加实际上也等价于使用了一个偏爱极值的先验。

```

前面两个例子都表明，不能孤立地理解先验，我们需要将它们放在特定模型的上下文中。由于根据观测值进行思考通常比根据模型参数进行思考更容易，因此先验预测分布可以帮助我们简化模型评估。这在参数通过多次数学转换或若干先验存在交互的复杂模型中，作用更为明显。

此外，先验预测分布可用于以更直观的方式向广大受众展示结果或讨论模型。领域专家不熟悉统计符号或代码，因此使用这些专业符号和代码往往不会导致富有成效的讨论，但如果你展示的是一个或多个模型的含义，那么就可以为他们提供更多讨论材料。这可以为你和领域合作伙伴提供有价值的见解。计算先验预测还具有其他优势，例如帮助我们调试模型，确保模型被正确地编写，并能够在计算环境中正确运行。

(posterior_pd)= 

## 2.3 理解你的预测结果 

既然可以使用来自先验预测分布的合成数据（即生成的数据）来帮助我们检查模型，那么也可以使用**后验预测分布**进行类似的分析，这个概念在 {ref}`Bayesian_inference` 和公式 [eq: post_pred_dist](eq:post_pred_dist) 中做过介绍。生成后验预测样本并基于样本做模型评估的过程，通常被称为**后验预测检查（ posterior predictive checks ）**。其基本思想是评估生成的数据与实际观测数据的接近程度。

理想情况下，我们评估接近度的方式取决于问题本身，但也存在一些通用规则。我们甚至可能想要使用多个指标来评估模型匹配数据（或错配数据）的不同方式。

{numref}`fig:posterior_predictive_check` 显示了一个非常简单的二项式模型和数据示例。在左侧子图中，我们将数据中观测到的成功次数（蓝线）与后验预测分布中超过 $1000$ 个样本的预测成功次数进行比较。右侧子图为表示结果的另一种方式，显示了观测数据（蓝线）与来自后验分布的 $1000$ 个样本的成功和失败比率。正如我们所见，在当前设置下，模型在捕捉均值方面做得非常好，即使模型认识到存在很多不确定性。我们不应该对模型在捕捉均值方面做得好而感到惊讶，因为我们直接对二项分布的均值进行了建模。在接下来的章节中，我们将看到后验预测检查提供的关于模型与数据的拟合不太明显但因此更有价值的示例。

```{figure} figures/posterior_predictive_check.png
:name: fig:posterior_predictive_check
:width: 8.00in

Beta-Binomial 模型的后验预测检查。在左侧子图中，有预测成功的数量（灰色直方图），黑色虚线表示预测成功的均值。蓝线是根据数据计算的平均值。在右侧子图中，有相同的信息，但以另一种方式表示。我们绘制的是获得 $0$ 或 $1$ 的概率，而不是成功的数量。我们用一条线来表示 $p(y=0) = 1-p(y=1)$ 的概率。

黑虚线是平均预测概率，蓝线是从数据计算的平均值。
```

后验预测检查不仅限于绘图，我们还可以执行数值检验 {cite:p}`GelmanBayesianDataAnalysis2013`。其中一种方法是通过计算贝叶斯 $p$ 值：

```{math}
:label: eq:post_pred_test_quantity

p_{B} = p(T_{sim} \leq T_{obs} \mid \tilde Y)

```

其中 $p_{B}$ 是贝叶斯 $p$ 值，定义为模拟测试统计量 $T_{sim}$ 小于或等于观测统计量 $T_{obs}$ 的概率。统计量 $T$ 可以是我们想用来评估模型是否适合数据的任何指标。

按照上述二项示例，我们可以选择 $T_{obs}$ 作为观测到的成功率，然后将其与后验预测分布 $T_{sim}$ 进行比较。 $p_{B}=0.5$ 的理想值意味着我们模拟生成的统计量 $T_{sim}$ ，其中一半时间低于观测统计量 $T_{obs}$，一半时间高于观测统计量 $T_{obs}$，这是一个正确拟合的预期结果。

因为绘图更加直观一些，所以也可以使用贝叶斯 $p$ 值来创建绘图。

{numref}`fig:posterior_predictive_check_pu_values` 的第一个子图以黑色实线显示贝叶斯 $p$ 值的分布，虚线表示相同大小数据集的预期分布。我们可以使用 `ArviZ` 的 `az.plot_bpv(., kind="p_value")` 获得这样的图。第二个子图在概念上相似，不同之处在于我们评估了有多少模拟低于（或高于）观测数据。对于一个校准良好的模型，所有观测值都应该得到同样好的预测，即高于或低于预期的预测数量应该相同。因此应该得到一个均匀分布。对于任何有限数据集，即使是完美校准的模型也会显示出与均匀分布的偏差，我们绘制了一条带，我们预计会看到 $94\%$ 的均匀曲线。

::: {admonition} 贝叶斯 $p$ 值 

我们将 $p_{B}$ 称为贝叶斯 $p$ 值，是因为公式 {eq}`eq:post_pred_test_quantity` 中的量实质上是 $p$ 值的定义，但我们称其为贝叶斯的，因为我们并没有使用零假设下的统计量 $T$ 作为后验预测分布的抽样分布。请注意，我们没有以任何零假设为条件；我们也没有使用任何预定义的阈值来声明统计显著性或执行假设检验。

::: 

```{figure} figures/posterior_predictive_check_pu_values.png
:name: fig:posterior_predictive_check_pu_values
:width: 8.00in

Beta-Binomial 模型的后验预测分布。在第一幅图中，实线曲线是小于或等于观测数据的预测值比例的 KDE。虚线表示与观测数据大小相同的数据集的预期分布。在第二个子图上，黑线是一个 KDE，表示预测值的比例小于或等于每次观测计算的观测值，而不是像第一个子图中的每个模拟。白线代表理想情况，标准均匀分布，以及我们期望在相同大小的数据集上看到的该均匀分布的灰带偏差。
```

正如之前所说，我们可以从许多 $T$ 统计量中进行选择来总结观测和预测。

{numref}`fig:posterior_predictive_check_tstat` 显示了两个示例，第一个子图中的 $T$ 是平均值，第二个子图中的 $T$ 是标准差。曲线是 KDE，表示来自后验预测分布的 $T$ 统计量的分布，点是观测数据的值。

```{figure} figures/posterior_predictive_check_tstat.png
:name: fig:posterior_predictive_check_tstat
:width: 8.00in

Beta-Binomial 模型的后验预测分布。在第一幅图中，实线曲线是平均值小于或等于观测数据的预测值的模拟比例的 KDE。在第二个子图上，相同，但标准差。黑点代表从观测数据计算的平均值（第一组）或标准差（第二组）。
```

在继续阅读之前，你应该花点时间仔细检查 {numref}`fig:posterior_predictive_many_examples` 并尝试理解为什么这些图看起来像这样。在这个图中，我们有一系列简单的例子来帮助我们获得关于如何解释后验预测检查图的直觉[^7]。在所有这些示例中，观测数据（蓝色）遵循高斯分布。

1. 在第一行，模型预测相对于观测数据系统地转移到更高值的观测值。

2. 在第二行，模型做出的预测比观测数据更广泛。

3. 在第三行，我们有相反的情况，模型没有在尾部生成足够的预测。

4. 最后一行显示了一个模型在混合高斯后进行预测。

我们现在要特别注意 {numref}`fig:posterior_predictive_many_examples` 的第三列。此列中的图表非常有用，但同时它们一开始可能会令人困惑。

从上到下，你可以将它们阅读为：

1. 模型左尾缺少观测值（而右尾观测值更多）。

2. 模型在中间做出更少的预测（而在尾部做出更多的预测）。

3. 模型在尾部的预测较少。

4. 该模型正在或多或少地做出经过良好校准的预测，但我是一个持怀疑态度的人，所以我应该运行另一个后验预测检查来确认。

如果这种阅读图表的方式仍然让你感到困惑，我们可以从完全等效的不同角度尝试，但它可能更直观，只要你记住你可以更改模型，而不是观测结果 [^8]。从上到下，你可以将它们阅读为：

1. 左侧观测较多

2. 中间观测较多

3. 尾部观测较多。

4. 观测结果似乎分布良好（至少在预期范围内），但你不应该相信我。我只是柏拉图世界中的柏拉图模型。

我们希望 {numref}`fig:posterior_predictive_many_examples` 和随附的讨论为你提供足够的直觉，以便更好地在实际场景中执行模型检查。

```{figure} figures/posterior_predictive_many_examples.png
:name: fig:posterior_predictive_many_examples
:width: 8.00in

一组简单假设模型的后验预测检查。在第一列，蓝色实线代表观测数据和来自假设模型的浅灰色预测。在第二列，实线是小于或等于观测数据的预测值比例的 KDE。虚线表示与观测数据大小相同的数据集的预期分布。在第三个子图上，KDE 是指每个观测值小于或等于观测值的预测值的比例。白线代表预期的均匀分布，灰带代表与所用数据集大小相同的数据集的预期偏差。该图是使用 ArviZ 的函数 `az.plot_ppc(.)`、`az.plot_bpv(., kind="p_values")` 和 `az.plot_bpv(., kind="u_values")` 制作的。
```

后验预测检查，无论是使用图表还是数字摘要，甚至两者的组合，都是一个非常灵活的想法。这个概念足够笼统，可以让从业者发挥他们的想象力，通过他们的预测以及一个或多个模型对特定问题的工作情况，提出不同的方法来探索、评估和更好地理解模型。

(diagnosing_inference)= 

## 2.4 数值推断的诊断 

使用数值方法来近似后验分布使我们能够求解贝叶斯模型，这些模型用笔和纸求解可能很乏味，或者在数学上可能难以解决。不幸的是，它们并不总是按预期工作。出于这个原因，我们必须始终评估他们提供的结果是否有用。我们可以使用一系列数字和视觉诊断工具来做到这一点。在本节中，我们将讨论马尔可夫链蒙特卡洛方法最常见和最有用的诊断工具。

为了帮助我们理解这些诊断工具，我们将创建三个*合成后验*。第一个是来自 $\text{Beta}(2, 5)$ 的样本。我们使用 SciPy 生成它，我们称之为“good_chains”。这是一个“好”样本的示例，因为我们正在生成独立且同分布 (iid) 的绘图，理想情况下，这是我们想要近似后验的结果。第二个称为“bad_chains0”，代表来自后验的不良样本。我们通过对“good_chains”进行排序然后添加一个小的高斯误差来生成它。 `bad_chains0` 是一个糟糕的样本，原因有两个：

- 这些值不是独立的。相反，它们是高度自相关的，这意味着给定序列中任何位置的任何数字，我们都可以高精度地计算前后的值。

- 这些值的分布并不相同，因为我们正在将先前展平和排序的数组重塑为二维数组，代表两条链。

第三个*合成后验*称为 `bad_chains1` 是从 `good_chains` 生成的，我们通过随机引入连续样本彼此高度相关的部分，将其转化为来自后验的不良样本的表示。 `bad_chains1` 代表了一个很常见的场景，一个采样器可以很好地解析参数空间的一个区域，但是一个或多个区域很难采样。

```python
good_chains = stats.beta.rvs(2, 5,size=(2, 2000))
bad_chains0 = np.random.normal(np.sort(good_chains, axis=None), 0.05,
                               size=4000).reshape(2, -1)

bad_chains1 = good_chains.copy()
for i in np.random.randint(1900, size=4):
    bad_chains1[i%2:,i:i+100] = np.random.beta(i, 950, size=100)

chains = {"good_chains":good_chains,
          "bad_chains0":bad_chains0,
          "bad_chains1":bad_chains1}
```

请注意，3 个合成后验是来自标量（单参数）后验分布的样本。这对于我们当前的讨论来说已经足够了，因为我们将看到的所有诊断都是根据模型中的参数计算的。

(ess)= 

### 2.4.1 有效样本数量 

使用 MCMC 抽样方法时，有理由怀疑特定样本是否足够大以可靠地计算感兴趣的数量，例如平均值或 HDI。这是我们不能仅通过查看样本数量直接回答的问题，原因是来自 MCMC 方法的样本将具有一定程度的**自相关**，因此该样本中包含的实际*信息量*将为比我们从相同大小的 iid 样本中得到的要少。当我们可以观测到它们之间的相似性作为它们之间的时间滞后的函数时，我们说一系列值是自相关的。例如，如果今天的日落时间是早上 6:03，那么你知道明天的日落时间大约是同一时间。事实上，考虑到今天的价值，你离赤道越近，预测未来日落时间的时间就越长。也就是说，赤道处的自相关比靠近两极的地方大 [^9]。

我们可以将有效样本量 (ESS) 视为一个考虑自相关的估计量，并提供如果我们的样本实际上是 iid 时我们将拥有的抽签次数。这种解释很吸引人，但我们必须小心不要过度解释它，我们将在下面看到。

使用 ArviZ，我们可以使用 `az.ess()` 计算均值的有效样本大小

```python
az.ess(chains)
```

```none
<xarray.Dataset>
Dimensions:      ()
Data variables:
    good_chains  float64 4.389e+03
    bad_chains0  float64 2.436
    bad_chains1  float64 111.1
```

我们可以看到，即使我们的合成后验中的实际样本数为 4000，“bad_chains0”的效率也与大小为 $\约 2$ 的 iid 样本相当。这肯定是一个较低的数字，表明采样器存在问题。鉴于 ArviZ 使用的计算 ESS 的方法以及我们如何创建“bad_chains0”，这个结果是完全可以预期的。`bad_chains0` 是双峰分布，每条链都卡在每种模式中。对于这种情况，ESS 将大约等于 MCMC 链探索的模式数量。对于 `bad_chains1`，我们也得到了一个较低的数字 $\约 111$，只有 `good_chains` 的 ESS 接近实际样本数。

::: {admonition} 关于有效样本的有效性

如果你使用不同的随机种子重新运行这些合成后验的生成，你将看到每次获得的有效样本量都不同。这是预期的，因为样本不会完全相同，它们毕竟是样本。对于`good_chains`，平均而言，有效样本量的值将低于样本数。但请注意，ESS 实际上可能更大！当使用 NUTS 采样器（参见第 {ref}`inference_methods`）时，对于后验分布接近高斯分布且几乎独立于模型中其他参数的参数，可能会出现大于样本总数的 ESS 值。

::: 

马尔可夫链的收敛在参数空间 {cite:p}`vehtari_rank_2019` 上并不均匀，直观地说，从分布的主体中获得良好的近似值比从尾部更容易，因为尾部由罕见事件主导。 `az.ess()` 返回的默认值是 `bulk-ESS`，它主要评估分布的 *center* 的解析程度。如果你还想报告后验间隔或者你对罕见事件感兴趣，你应该检查 `tail-ESS` 的值，它对应于百分位数 5 和 95 处的最小 ESS。如果你对特定分位数感兴趣，你可以使用 `az.ess(., method='quantile')` 向 ArviZ 询问这些特定值。

由于 ESS 值在参数空间中变化，我们可能会发现在单个图中可视化这种变化很有用。我们至少有两种方法可以做到这一点。绘制 ESS 的具体分位数 `az.plot_ess(., kind="quantiles")` 或两个分位数 `az.plot_ess(., kind="local")` 之间定义的小间隔，如 {numref}`fig:plot_ess`。

```python
_, axes = plt.subplots(2, 3, sharey=True, sharex=True)
az.plot_ess(chains, kind="local", ax=axes[0]);
az.plot_ess(chains, kind="quantile", ax=axes[1]);
```

```{figure} figures/plot_ess.png
:name: fig:plot_ess
:width: 8.00in

上图：小区间概率估计的局部 ESS。底部：分位数 ESS 估计。虚线表示我们认为有效样本量足够的最小建议值 400。理想情况下，我们希望局部和分位数 ESS 在参数空间的所有区域中都很高。
```

作为一般的经验法则，我们建议 ESS 的值大于 400，否则，对 ESS 本身的估计和对其他量的估计，比如我们接下来会看到的 $\hat R$，基本上是不可靠的 {cite: p}`vehtari_rank_2019`。最后，我们说 ESS 提供了如果我们的样本实际上是 iid 的抽签次数。

然而，我们必须小心这种解释，因为对于参数空间的不同区域，ESS 的实际值不会相同。考虑到这些细节，直觉似乎仍然有用。

(potential-scale-reduction-factor-hat-r)= 

### 2.4.2 潜在的比例缩减因子 $\hat R$ 

在非常一般的条件下，马尔可夫链蒙特卡洛方法有理论上的保证，无论起点如何，它们都会得到正确的答案。不幸的是，细则表明保证仅对无限样本有效。因此在实践中，我们需要一些方法来估计有限样本的收敛性。一个普遍的想法是运行多个链，从非常不同的点开始，然后检查生成的链，看看它们是否*看起来相似*彼此。

这个直观的概念可以形式化为称为 $\hat R$ 的数字诊断。这个估计器有很多版本，因为它多年来一直在改进 {cite:p}`vehtari_rank_2019`。最初，$\hat R$ 诊断被解释为由于 MCMC 有限抽样而高估了方差。这意味着如果你继续无限采样，你应该通过 $\hat R$ 因子减少你的估计方差。因此得名“潜在尺度缩减因子”，目标值为 1 表示增加样本数不会进一步降低估计的方差。

然而，在实践中，最好将其视为一种诊断工具，而不是试图过度解释它。

参数 $\theta$ 的 $\hat R$ 计算为 $\theta$ 的所有样本的标准差，即包括所有链，除以分离的链内标准差的均方根.实际计算涉及更多一点，但总体思路仍然正确 {cite:p}`vehtari_rank_2019`。理想情况下，我们应该得到一个值 1，因为链之间的方差应该与链内的方差相同。从实用的角度来看，$\hat R\lessapprox 1.01$ 的值被认为是安全的。

 Using ArviZ we can compute the $\hat R$ diagnostics with the `az.rhat()` function 

使用 ArviZ，我们可以使用 `az.rhat()` 函数计算 $\hat R$ 诊断


```python
az.rhat(chains)
```

```none
<xarray.Dataset>
Dimensions:      ()
Data variables:
    good_chains  float64 1.000
    bad_chains0  float64 2.408
    bad_chains1  float64 1.033
```

从这个结果我们可以看到 $\hat R$ 正确地将 `good_chains` 识别为好样本，将 `bad_chains0` 和 `bad_chains1` 正确识别为具有不同程度问题的样本。虽然 `bad_chains0` 完全是一场灾难，但 `bad_chains1` 似乎更接近于达到 *ok-chain 状态*，但仍处于关闭状态。

(Monte_Carlo_standard_error)= 

### 2.4.3 蒙特卡洛标准误差 

当使用 MCMC 方法时，我们引入了额外的不确定性层，因为我们用有限数量的样本来近似后验。

我们可以使用基于马尔可夫链中心极限定理的蒙特卡洛标准误差 (MCSE) 来估计引入的误差量（参见 {ref}`markov_chains` 节）。 MCSE 考虑到样本并非真正相互独立，实际上是从 ESS {cite:p}`vehtari_rank_2019` 计算得出的。虽然 ESS 和 $\hat R$ 的值与参数的规模无关，但解释 MCSE 是否足够小需要领域专业知识。如果我们想要将估计参数的值报告到小数点后第二位，我们需要确保 MCSE 低于小数点后第二位，否则，我们将错误地报告比我们实际拥有的精度更高的精度。只有当我们确定 ESS 足够高并且 $\hat R$ 足够低时，我们才应该检查 MCSE；否则，MCSE 是没有用的。

使用 ArviZ，我们可以使用函数 `az.mcse()` 计算 MCSE

```python
az.mcse(chains)
```

```none
<xarray.Dataset>
Dimensions:      ()
Data variables:
    good_chains  float64 0.002381
    bad_chains0  float64 0.1077
    bad_chains1  float64 0.01781
```

与 ESS 一样，MCSE 在参数空间中有所不同，然后我们可能还想针对不同的区域评估它，例如特定的分位数。此外，我们可能还希望一次可视化多个值，如 {numref}`fig:plot_mcse`。

```python
az.plot_mcse(chains)
```

```{figure} figures/plot_mcse.png
:name: fig:plot_mcse
:width: 8.00in

分位数的本地 MCSE。子图 y 轴共享相同的比例以方便它们之间的比较。理想情况下，我们希望 MCSE 在参数空间的所有区域中都很小。请注意，与两个坏链的 MCSE 相比，“good_chains”的 MCSE 值在所有值中都相对较低。
```
最后，ESS、$\hat R$ 和 MCSE 都可以通过一次调用 `az.summary(.)` 函数来计算。

```python
az.summary(chains, kind="diagnostics")
```

```{list-table}
* -
  - **mcse_mean***
  - **mcse_sd**
  - **ess_bulk**
  - **ess_tail**
  - **r_hat**
* - good_chains
  - 0.002
  - 0.002
  - 4389.0
  - 3966.0
  - 1.00
* - bad_chains0
  - 0.108
  - 0.088
  - 2.0
  - 11.0
  - 2.41
* - bad_chains1
  - 0.018
  - 0.013
  - 111.0
  - 105.0
  - 1.03
```

第一列是均值或（期望）的蒙特卡洛标准误差，第二列是标准差 [^10] 的蒙特卡洛标准误差。然后我们有批量和尾部有效样本大小，最后是 $\hat R$ 诊断。

(trace-plots)= 

### 2.4.4 轨迹图 

轨迹图可能是贝叶斯文学中最流行的图。

它们通常是我们在推断后制作的第一个图，以直观地检查*我们得到了什么*。通过在每个迭代步骤中绘制采样值来绘制轨迹图。从这些图中，我们应该能够看到不同的链是否收敛到相同的分布，我们可以得到自相关程度的*感觉*等。在 ArviZ 中，通过调用函数 `az.plot_trace(.)` 我们得到右侧的迹线图加上样本值分布的表示，使用 KDE 表示连续变量，使用直方图表示左侧的离散变量。

```python
az.plot_trace(chains)
```

```{figure} figures/trace_plots.png
:name: fig:trace_plots
:width: 8.00in

在左栏中，我们看到每个链一个 KDE。在右侧的列中，我们看到每条链每一步的采样值的值。请注意每个示例链之间的 KDE 和跟踪图的差异，特别是 `good_chains` 中的 *fuzzy Caterpillar* 外观与其他两个中的不规则性。
```

{numref}`fig:trace_plots` 显示了 `chains` 的轨迹图。从中，我们可以看到 `good_chains` 中的抽奖属于相同的分布，因为两条链之间只有很小的（随机）差异。当我们看到按迭代排序的绘制（即跟踪本身）时，我们可以看到链看起来相当*嘈杂*，没有明显的趋势或模式，也很难区分一条链与另一条链。这与我们得到的 `bad_chains0` 形成鲜明对比。对于这个样本，我们清楚地看到两个不同的分布，只有一些重叠。这很容易从 KDE 和跟踪中看到。这些链正在探索参数空间的两个不同区域。 `bad_chains1` 的情况有点微妙。 KDE 显示的分布似乎与 `good_chains` 中的分布相似，两条链之间的差异更加明显。我们有 2 个或 3 个峰值吗？分布似乎不一致，也许我们只有一种模式，额外的峰值是伪影！峰值通常看起来很可疑，除非我们有理由相信多模态分布，例如，来自我们数据中的子群体。该迹线似乎也与“good_chains”中的迹线有些相似，但更仔细的检查发现存在长单调性区域（平行于 x 轴的线）。这清楚地表明采样器卡在参数空间的某些区域中，可能是因为我们有一个多峰后验，在非常低概率的模式之间存在障碍，或者可能是因为我们有一些参数空间区域的曲率是和其他人太不一样了。

(autocorr_plot)= 

### 2.4.5 自相关图 

正如我们在讨论有效样本量时看到的那样，自相关减少了样本中包含的实际信息量，因此我们希望将其保持在最低限度。我们可以使用 az.plot_autocorr 直接检查自相关。


```python
az.plot_autocorr(chains, combined=True)
```

```{figure} figures/autocorrelation_plot.png
:name: fig:autocorrelation_plot
:width: 8.00in

自相关函数在 100 步窗口上的条形图。对于整个图，“good_chains”的条形高度接近于零（并且大部分在灰色带内），这表明自相关非常低。 `bad_chains0` 和 `bad_chains1` 中的高条表示自相关值较大，这是不可取的。灰色带代表 95% 置信区间。
```

在看到 `az.ess` 的结果后，我们在 {numref}`fig:autocorrelation_plot` 中看到的内容至少是定性的。

`good_chains` 显示出基本上为零的自相关，`bad_chains0` 高度相关，而 `bad_chains1` 并没有那么糟糕，但自相关仍然很明显并且是长期的，即它不会迅速下降。

(rank-plots)= 

### 2.4.6 秩图 

等级图是另一种可视化诊断，我们可以用来比较链内和链之间的采样行为。等级图，简单地说，是等级样本的直方图。通过首先组合所有链，然后为每个链分别绘制结果来计算排名。如果所有的链都针对相同的分布，我们希望排名具有均匀分布。此外，如果所有链的排名图看起来相似，这表明链的混合良好 {cite:p}`vehtari_rank_2019`。

```python 
az.plot_rank(chains, ax=ax[0], kind="bars")
```

```{figure} figures/rank_plot_bars.png
:name: fig:rank_plot_bars

使用“条形”表示对图进行排名。特别是，将条形的高度与表示均匀分布的虚线进行比较。理想情况下，条形应遵循均匀分布。
```

“bars”表示的一种替代方法是垂直线，缩写为“vlines”。

```python 
az.plot_rank(chains, kind="vlines")
```

```{figure} figures/rank_plot_vlines.png
:name: fig:rank_plot_vlines
:width: 8.00in

使用“vline”表示对图进行排名。垂直线越短越好。虚线上方的垂直线表示特定等级的采样值过多，下方的垂直线表示缺少采样值。
```

我们可以在图 {numref}`fig:rank_plot_bars` 和 {numref}`fig:rank_plot_vlines` 中看到，`good_chains` 的排名非常接近 Uniform，并且两条链看起来彼此相似，没有明显的模式。这与 `bad_chains0` 的结果形成鲜明对比，其中链偏离了一致性，并且他们正在探索两组不同的价值，在中间排名上有一些重叠。请注意，这与我们创建“bad_chains0”的方式以及我们在其跟踪图中看到的方式是一致的。 `bad_chains1` 在某种程度上是统一的，但到处都有一些大的偏差，反映出问题比来自 `bad_chains0` 的问题更*本地*。

秩图可能比跟踪图更敏感，因此我们推荐它们而不是后者。我们可以使用 `az.plot_trace(., kind="rank_bars")` 或 `az.plot_trace(., kind="rank_vlines")` 获得它们。这些函数不仅绘制等级，还绘制后验的边缘分布。这种图有助于快速了解后验*看起来像什么*，这在许多情况下可以帮助我们发现采样或模型定义的问题，尤其是在建模的早期阶段，我们很可能不会确定我们真正想做的事情，因此，我们需要探索许多不同的选择。随着我们的进步和模型开始变得更有意义，我们可以检查 ESS、$\hat R$ 和 MCSE 是否正常，如果不正常，则知道我们的模型需要进一步改进。

(divergences)= 

### 2.4.7 散度 

到目前为止，我们一直在通过研究生成的样本来诊断采样器的工作情况。执行诊断的另一种方法是监视采样方法的内部工作行为。这种诊断的一个突出示例是一些**Hamiltonian Monte Carlo** (HMC) 方法中存在的散度概念 [^11]。散度，或更准确地说是散度转变，是一种强大而灵敏的样本诊断方法，可作为我们在前几节中看到的诊断的补充。

让我们在一个非常简单的模型的背景下讨论散度，我们将在本书后面找到更现实的例子。我们的模型由一个参数 $\theta2$ 组成，该参数遵循区间 $[-\theta1, \theta1]$ 中的均匀分布，并且 $\theta1$ 是从正态分布中采样的。当$\theta1$ 很大时，$\theta2$ 将遵循一个跨越大范围的均匀分布，当$\theta1$ 接近于零时，$\theta2$ 的宽度也将接近于零。使用 PyMC3，我们可以将此模型编写为：


```{code-block} ipython3
:name: divm0
:caption: divm0

with pm.Model() as model_0:
    θ1 = pm.Normal("θ1", 0, 1, testval=0.1)
    θ2 = pm.Uniform("θ2", -θ1, θ1)
    idata_0 = pm.sample(return_inferencedata=True)
```

::: {admonition} The ArviZ InferenceData format 

`az.InferenceData` 是一种专门为 MCMC 贝叶斯用户设计的数据格式。

它基于 xarray {cite:p}`hoyer2017`，一个灵活的 N 维数组包。 InferenceData 对象的主要目的是提供一种方便的方法来存储和操作贝叶斯工作流程期间生成的信息，包括来自分布的样本，如后验、先验、后验预测、先验预测以及采样期间生成的其他信息和诊断。 InferenceData 对象使用称为组的概念来组织所有这些信息。

在本书中，我们广泛使用了`az.InferenceData`。我们使用它来存储贝叶斯推断结果、计算诊断、生成绘图以及从磁盘读取和写入。有关完整的技术规范和 API，请参阅 ArviZ 文档。

::: 

请注意代码块 [divm0](divm0) 中的模型如何不以任何观测为条件，这意味着“model_0”指定了由两个未知数（“θ1”和“θ2”）参数化的后验分布。

你可能还注意到我们已经包含了参数 `testval=0.1`。我们这样做是为了指示 PyMC3 从特定值（本例中为 $0.1$）开始采样，而不是从其默认值开始采样。

默认值为 $\theta1 = 0$，对于该值，$\theta2$ 的概率密度函数是一个狄拉克增量 [^12]，这将产生错误。使用 `testval=0.1` 只会影响采样的初始化方式。

 In {numref}`fig:divergences_trace` we can see vertical bars at the bottom of the KDEs for `model0`. Each one of these bars represents a divergence, indicating that something went wrong during sampling. We can see something similar using other plots, like with `az.plot_pair(., divergences=True)` as shown in {numref}`fig:divergences_pair`, here the divergences are the blue dots, which are everywhere! 

在 {numref}`fig:divergences_trace` 中，我们可以在“model0”的 KDE 底部看到竖线。这些条形中的每一个都代表一个散度，表明在采样过程中出现了问题。我们可以使用其他图看到类似的东西，例如 {numref} `fig:divergences_pair` 中所示的 `az.plot_pair(.,divergences=True)`，这里的散度是无处不在的蓝点！

```{figure} figures/divergences_trace.png
:name: fig:divergences_trace
:width: 8.00in

模型 0 代码 [divm0](divm0)、模型 1 [divm1](divm1) 和模型 1bis 的 KDE 和等级图，与模型 1 [divm1](divm1) 相同，但带有 `pm.sample(., target_accept =0.95)`。黑色竖条代表散度。
```

```{figure} figures/divergences_pair.png
:name: fig:divergences_pair
:width: 8.00in

来自代码 [divm0](divm0) 的模型 0、代码块 [divm1](divm1) 的模型 1 和模型 1bis 的后验样本散点图，与代码块 [divm1](divm1) 中的模型 1 相同但使用`pm.sample(., target_accept=0.95)`。蓝点代表散度。
```

`model0` 肯定有问题。通过检查代码块 [divm0](divm0) 中的模型定义，我们可能会意识到我们以一种奇怪的方式定义它。$\theta1$ 是一个以 0 为中心的正态分布，因此我们应该期望一半的值是负数，但是对于负值 $\theta2$ 将定义在区间 $[\theta1, -\theta1]$ 中，这至少有点奇怪。

因此，让我们尝试**重新参数化**模型，即以不同但在数学上等效的方式表达模型。例如，我们可以这样做：

```{code-block} ipython3
:name: divm1
:caption: divm1
with pm.Model() as model_1:
    θ1 = pm.HalfNormal("θ1", 1 / (1-2/np.pi)**0.5)
    θ2 = pm.Uniform("θ2", -θ1, θ1)
    idata_1 = pm.sample(return_inferencedata=True)
```

现在 $\theta1$ 将始终提供合理的值，我们可以将其输入到 $\theta2$ 的定义中。请注意，我们将 $\theta1$ 的标准差定义为 $\frac{1}{\sqrt{(1-\frac{2}{\pi})}}$ 而不是 1。这是因为标准半法线的偏差是 $\sigma \sqrt{(1-\frac{2}{\pi})}$ 其中 $\sigma$ 是半法线的尺度参数。换句话说，$\sigma$ 是*展开*正态分布的标准差，而不是半正态分布。

无论如何，让我们看看这些重新参数化的模型如何处理散度。 {numref}`fig:divergences_trace` 和 {numref}`fig:divergences_pair` 表明，`model1` 的散度数量已大大减少，但我们仍然可以看到其中的一些。我们可以尝试减少散度的一个简单选择是增加 `target_accept` 的值，如代码块 [divm2](divm2) 所示，默认情况下此值为 0.8，最大有效值为 1（请参阅第 {ref}`hmc` 了解详情）。

```{code-block} ipython3
:name: divm2
:caption: divm2
with pm.Model() as model_1bis:
    θ1 = pm.HalfNormal("θ1", 1 / (1-2/np.pi)**0.5)
    θ2 = pm.Uniform("θ2", -θ1, θ1)
    idata_1bis = pm.sample(target_accept=.95, return_inferencedata=True)
```

{numref}`fig:divergences_trace` 和 {numref}`fig:divergences_pair` 中的`model1bis` 与 `model1` 相同，但我们更改了采样参数之一的默认值 `pm.sample(., target_accept =0.95)` 。我们可以看到，最终我们消除了所有的散度。这已经是个好消息，但为了信任这些样本，我们仍然需要检查 $\hat R$ 和 ESS 的值，如前几节所述。

::: {admonition} Reparameterization 

重新参数化对于将难以采样的后验几何体转换为更容易的几何体很有用。这可能有助于消除散度，但即使不存在散度，它也会有所帮助。例如，我们可以使用它来加快采样速度或增加有效样本的数量，而无需增加计算成本。此外，重新参数化还可以帮助更好地解释或交流模型及其结果（参见第 {ref}`conjugate_priors` 节中的 Alice 和 Bob 示例）。

::: 

(sampler-parameters-and-other-diagnostics)= 

### 2.4.8 采样器参数和其他诊断 

大多数采样器方法都有影响采样器性能的超参数。虽然大多数 PPL 尝试使用合理的默认值，但在实践中，它们并不适用于数据和模型的所有可能组合。正如我们在前几节中看到的，有时可以通过增加参数“target_accept”来消除散度，例如，如果散度源于数值不精确。还有其他采样器参数也可以帮助解决采样问题，例如，我们可能希望增加用于调整 MCMC 采样器的迭代次数。在 PyMC3 中，我们默认有 `pm.sample(.,tune=1000)`。在调整阶段，采样器参数会自动调整。有些模型更复杂，需要更多交互才能让采样器学习更好的参数。因此增加转弯步数有助于增加 ESS 或降低 $\hat R$。增加抽签次数也有助于收敛，但总的来说其他路线更有成效。如果一个模型在数千次绘制时未能收敛，它通常仍会在 10 倍以上的绘制时失败，或者轻微的改进并不能证明额外的计算成本是合理的。

重新参数化、改进模型结构、提供更多信息的先验，甚至更改模型通常会更有效 [^13]。

我们要注意的是，在建模的早期阶段，我们可以使用相对较少的绘制次数来测试模型是否运行，我们实际上已经编写了预期的模型，我们大致得到了合理的结果。对于这个初始检查，大约 200 或 300 次通常就足够了。然后，当我们对模型更有信心时，我们可以将绘制次数增加到几千次，也许大约 2000 或 4000 次。

除了本章中显示的诊断之外，还存在其他诊断，例如平行图和分离图。所有这些诊断都是有用的并且有它们的位置，但是为了本文的简洁，我们在本节中省略了它们。要查看其他内容，我们建议你访问包含更多示例的 ArviZ 文档和绘图库。

(model_cmp)= 

## 2.5 模型比较 

通常，我们希望构建的模型既不会太简单以至于错过数据中的有价值信息，也不会太复杂以至于无法适应数据中的噪声。找到这个*甜蜜点*是一项复杂的任务。部分原因是没有一个单一的标准来定义最佳解决方案，部分原因是可能不存在这样的最佳解决方案，部分原因是在实践中我们需要从有限数据集上评估的有限模型集中进行选择。

尽管如此，我们仍然可以尝试找到好的通用策略。一种有用的解决方案是计算泛化误差，也称为样本外预测精度。这是对模型在预测未用于拟合的数据方面表现如何的估计。理想情况下，任何预测准确性的度量都应该考虑到我们试图解决的问题的细节，包括与模型预测相关的收益和成本。也就是说，我们应该应用决策理论方法。但是，我们也可以依赖适用于广泛模型和问题的通用设备。这种设备有时被称为评分规则，因为它们帮助我们对模型进行评分和排名。从许多可能的评分规则中可以看出，对数评分规则具有非常好的理论性质 {cite:p}`gneiting_2007`，因此被广泛使用。在贝叶斯设置下，日志评分规则可以计算为。

```{math} 
:label: eq:elpd

\text{ELPD} = \sum_{i=1}^{n} \int p_t(\tilde y_i) \; \log p(\tilde y_i \mid y_i) \; d\tilde y_i 
```

其中 $p_t(\tilde y_i)$ 是 $\tilde y_i$ 的真实数据生成过程的分布，$p(\tilde y_i \mid y_i)$ 是后验预测分布。公式 {eq}`eq:elpd` 中定义的量被称为 **expected log pointwise predict density** (ELPD)。

预期是因为我们正在整合真正的数据生成过程，即整合所有可能从该过程生成的数据集，并且逐点整合，因为我们在 $n$ 观测上执行每个观测 ($y_i$) 的计算。为简单起见，我们将术语密度用于连续和离散模型 [^14]。

对于实际问题，我们不知道 $p_t(\tilde y_i)$，因此公式 {eq}`eq:elpd` 中定义的 ELPD 没有立即使用，实际上我们可以计算：

```{math} 
:label: eq:elpd_practice

\sum_{i=1}^{n} \log \int \ p(y_i \mid \boldsymbol{\theta}) \; p(\boldsymbol{\theta} \mid y) d\boldsymbol{\theta} 
```

由公式 {eq}`eq:elpd_practice` 定义的量（或乘以某个常数的量）通常称为偏差，它用于贝叶斯和非贝叶斯上下文 [^15]。当似然度为高斯时，公式 {eq}`eq:elpd_practice` 将与二次平均误差成正比。

为了计算公式 {eq}`eq:elpd_practice`，我们使用了用于拟合模型的相同数据，因此平均而言，我们会高估 ELPD（公式 {eq}`eq:elpd`），这将导致我们选择模型容易过拟合。幸运的是，有几种方法可以更好地估计 ELPD。其中之一是交叉验证，我们将在下一节中看到。

(CV_and_LOO)= 

### 2.5.1 交叉验证和留一法 

交叉验证 (CV) 是一种估计样本外预测准确性的方法。这种方法需要多次重新拟合模型，每次都排除数据的不同部分。然后使用排除的部分来测量模型的准确性。此过程重复多次，模型的估计准确度将是所有运行的平均值。然后使用整个数据集再次拟合模型，这是用于进一步分析和/或预测的模型。我们可以将 CV 视为一种模拟或近似样本外统计数据的方法，同时仍使用所有数据。

当排除的数据是单个数据点时，留一法交叉验证 (LOO-CV) 是一种特殊类型的交叉验证。使用 LOO-CV 计算的 ELPD 为 $\text{ELPD}_\text{LOO-CV}$：

```{math} 
:label: eq:elpd_loo_cv

\text{ELPD}_\text{LOO-CV} = \sum_{i=1}^{n} \log   \int \ p(y_i \mid \boldsymbol{\theta}) \; p(\boldsymbol{\theta} \mid y_{-i}) d\boldsymbol{\theta} 
```

Computing Equation {eq}`eq:elpd_loo_cv` can easily become too costly as in practice we do not know $\boldsymbol{\theta}$ and thus we need to compute $n$ posteriors, i.e. as many values of $\boldsymbol{\theta_{-i}}$ as observations we have in our dataset.

计算公式 {eq}`eq:elpd_loo_cv` 很容易变得过于昂贵，因为实际上我们不知道 $\boldsymbol{\theta}$，因此我们需要计算 $n$ 后验，即 $\boldsymbol{ 的值一样多\theta_{-i}}$ 作为我们在数据集中的观测结果。

幸运的是，我们可以通过使用一种称为 Pareto 平滑重要性抽样留一法交叉验证 PSIS-LOO-CV 的方法从单一拟合数据中近似 $\text{ELPD}_\text{LOO-CV}$（有关详细信息，请参阅第 {ref}`loo_depth` 部分）。为简洁起见，为了与 ArviZ 保持一致，在本书中我们将此方法称为 LOO。重要的是要记住我们谈论的是 PSIS-LOO-CV，除非我们另有说明，否则当我们提到 ELPD 时，我们谈论的是通过这种方法估计的 ELPD。

ArviZ 提供了许多与 LOO 相关的函数，使用起来非常简单，但理解结果可能需要一点点小心。因此，为了说明如何解释这些函数的输出，我们将使用 3 个简单模型。模型在代码块 [pymc3_models_for_loo](pymc3_models_for_loo) 中定义。

```{code-block} ipython3
:name: pymc3_models_for_loo
:caption: pymc3_models_for_loo

y_obs = np.random.normal(0, 1, size=100)
idatas_cmp = {}

# Generate data from Skewnormal likelihood model
# with fixed mean and skewness and random standard deviation
with pm.Model() as mA:
    σ = pm.HalfNormal("σ", 1)
    y = pm.SkewNormal("y", 0, σ, alpha=1, observed=y_obs)
    idataA = pm.sample(return_inferencedata=True)

# add_groups modifies an existing az.InferenceData
idataA.add_groups({"posterior_predictive":
                  {"y":pm.sample_posterior_predictive(idataA)["y"][None,:]}})
idatas_cmp["mA"] = idataA

# Generate data from Normal likelihood model
# with fixed mean with random standard deviation
with pm.Model() as mB:
    σ = pm.HalfNormal("σ", 1)
    y = pm.Normal("y", 0, σ, observed=y_obs)
    idataB = pm.sample(return_inferencedata=True)

idataB.add_groups({"posterior_predictive":
                  {"y":pm.sample_posterior_predictive(idataB)["y"][None,:]}})
idatas_cmp["mB"] = idataB

# Generate data from Normal likelihood model
# with random mean and random standard deviation
with pm.Model() as mC:
    μ = pm.Normal("μ", 0, 1)
    σ = pm.HalfNormal("σ", 1)
    y = pm.Normal("y", μ, σ, observed=y_obs)
    idataC = pm.sample(return_inferencedata=True)

idataC.add_groups({"posterior_predictive":
                  {"y":pm.sample_posterior_predictive(idataC)["y"][None,:]}})
idatas_cmp["mC"] = idataC
```

要计算 LOO，我们只需要来自后验 [^16] 的样本。然后我们可以调用 `az.loo(.)`，它允许我们计算单个模型的 LOO。

在实践中，为两个或多个模型计算 LOO 是很常见的，因此常用的函数是`az.compare(.)`。

{numref}`table:compare_00` 是使用 `az.compare(idatas_cmp)` 生成的。

```{list-table} Summary of model comparison. Models are ranked from lowest to highest ELPD values (loo column).
:name: table:compare_00
* -
  - **rank**
  - **loo**
  - **p_loo**
  - **d_loo**
  - **weight**
  - **se**
  - **dse**
  - **warning**
  - **loo_scale**
* - mB
  - 0
  - -137.87
  - 0.96
  - 0.00
  - 1.0
  - 7.06
  - 0.00
  - False
  - log
* - mC
   - 1
   - -138.61
   - 2.03
   - 0.74
   - 0.0
   - 7.05
   - 0.85
   - False
   - log
*  - mA
   - 2
   - -168.06
   - 1.35
   - 30.19
   -  0.0
   - 10.32
   - 6.54
   - False
   - log
```

{numref}`table:compare_00`中有很多列，让我们一一详细说明它们的含义：

1. 第一列是索引，它列出了从传递给 `az.compare(.)` 的字典的键中获取的模型名称。

2. `rank`：模型上从0（预测准确率最高的模型）到模型个数的排名。

3. `loo`：ELPD 值列表。 DataFrame 总是从最好的 ELPD 到最差的排序。

4. `p_loo`：惩罚项的列表值。我们可以粗略地认为这个值是估计的有效参数数量（但不要太认真）。此值可能低于模型中*具有更多结构*（如分层模型）的实际参数数量，或者当模型具有非常弱的预测能力并可能表明严重的模型错误指定时，该值可能远高于实际数量。

5. `d_loo`：排名靠前的模型的 LOO 值与每个模型的 LOO 值之间的相对差异列表。出于这个原因，我们将始终为第一个模型获得 0 值。

6. `weight`：分配给每个模型的权重。这些权重可以粗略地解释为给定数据的每个模型（在比较模型中）的概率。有关详细信息，请参阅第 {ref}`model_averaging` 部分。

7. `se`：ELPD 计算的标准误差。

8. `dse`：ELPD 两个值之差的标准误。 `dse` 不一定与 `se` 相同，因为关于 ELPD 的不确定性可以在模型之间关联。对于排名靠前的模型，`dse` 的值始终为 0。

9. `warning`：如果`True`，这是一个警告，LOO 近似可能不可靠（详见{ref}`k-paretto` 节）。

10. `loo_scale`：报告值的比例。默认值为对数比例。其他选项是偏差，这是对数分数乘以 -2（这将颠倒顺序：较低的 ELPD 会更好）。负对数，这是对数分数乘以 -1，与偏差量表一样，值越低越好。

我们还可以在 {numref}`fig:compare_dummy` 中以图形方式表示 {numref}`table:compare_00` 中的部分信息。

模型的预测精度也从高到低排列。空心点代表“loo”的值，黑点是没有“p_loo”惩罚项的预测精度。黑色部分代表 LOO 计算“se”的标准误差。以三角形为中心的灰色部分表示每个模型的 LOO 值与排名最佳的模型之间的差异“dse”的标准误差。我们可以看到`mB` $\approx$ `mC` $>$ `mA`。

从 {numref}`table:compare_00` 和 {numref}`fig:compare_dummy` 我们可以看到模型 `mA` 排名最低，并且与其他两个明显分开。我们现在将讨论另外两个，因为它们的差异更加微妙。 `mB` 是预测准确率最高的一种，但与`mC` 相比差异可以忽略不计。根据经验，低于 4 的 LOO (`d_loo`) 差异被认为很小。

这两个模型之间的区别在于，对于“mB”，均值固定为 0，而对于“mC”，均值具有先验分布。 LOO 会惩罚添加此先验，由 `p_loo` 的值表示，`p_loo` 的值对于 `mC` 大于 `mB`，以及黑点（未惩罚的 ELPD）和开放点之间的距离（$\text{ELPD}_ \text{LOO-CV}$) 对于 `mC` 比 `mB` 大。我们还可以看到，这两个模型之间的 `dse` 远低于它们各自的 `se`，表明它们的预测高度相关。

鉴于“mB”和“mC”之间的微小差异，预计在稍微不同的数据集下，这些模型的排名可能会交换，“mC”成为排名最高的模型。此外，权重的值预计会发生变化（参见第 {ref}`model_averaging` 部分）。我们可以通过更改随机种子并重新拟合模型几次来轻松检查这一点。

```{figure} figures/compare_dummy.png
:name: fig:compare_dummy
:width: 8.00in

使用 LOO 进行模型比较。空心点代表“loo”的值，黑点是没有“p_loo”惩罚项的预测精度。黑色部分代表 LOO 计算“se”的标准误差。以三角形为中心的灰色部分表示每个模型的 LOO 值与排名最佳的模型之间的差异“dse”的标准误差。

```

(elpd_plots)= 

### 2.5.2 预期对数预测密度


在上一节中，我们计算了每个模型的 ELPD 值。

由于这是一个*全局*比较，它会将模型和数据简化为一个数字。但是从公式 {eq}`eq:elpd_practice` 和 {eq}`eq:elpd_loo_cv` 我们可以看到 LOO 是作为逐点值的总和计算的，每个观测值一个。因此，我们还可以执行 *local* 比较。我们可以将 ELPD 的各个值视为模型预测特定观测值的难度的指标。

为了比较基于每次观测 ELPD 的模型，ArviZ 提供了 `az.plot_elpd(.)` 函数。 {numref}`fig:elpd_dummy` 以成对方式显示模型 `mA`、`mB` 和 `mC` 之间的比较。

正值表示第一个模型比第二个模型更好地解决了观测结果。例如，如果我们观测第一个图（`mA-mB`），模型`mA`比模型`mB`更好地解决了观测49和72，而观测75和95则相反。我们可以看到前两个图`mA- mB`和`mA- mC`非常相似，原因是模型`mB`和模型`mC`实际上彼此非常相似。 {numref}`fig:elpd_and_khat` 表明观测 34、49、72、75 和 82 实际上是五个最*极端*的观测。

```{figure} figures/elpd_dummy.png
:name: fig:elpd_dummy
:width: 8.00in

逐点 ELPD 差异。注释点对应于 ELPD 差异为计算 ELPD 差异标准差 2 倍的观测值。所有 3 个示例中的差异都很小，尤其是在 `mB` 和 `mC` 之间。正值表示第一个模型比第二个模型更好地解析观测结果。
```

(k-paretto)= 

### 2.5.3 帕累托形状参数 

正如我们已经提到的，我们使用 LOO 来近似 $\text{ELPD}_\text{LOO-CV}$。这种近似涉及帕累托分布的计算（参见第 {ref}`loo_depth` 节中的详细信息），主要目的是获得更稳健的估计，这种计算的副作用是 $\hat \kappa$ 参数这样的帕累托分布可以用来检测有很大影响的观测，即
被排除在外时对预测分布有很大影响的观测值。通常，较高的 $\hat \kappa$ 值可能表明数据或模型存在问题，尤其是当 $\hat \kappa > 0.7$ {cite:p}`vehtari_pareto_2019, gabry_visualization_2017` 时。

在这种情况下，建议是 {cite:p}`loo_glossary`：

- 使用匹配矩方法 {cite:p}`Paananen2020` [^17]。通过一些额外的计算，可以将 MCMC 从后验分布中抽取，以获得更可靠的重要性采样估计。

- 对有问题的观测结果执行精确的留一交叉验证或使用 k 折交叉验证。

- 使用对异常观测更稳健的模型。

当我们得到至少一个 $\hat \kappa > 0.7$ 的值时，我们会在调用 `az.loo(.)` 或 `az.compare(.)` 时收到警告。 {numref}`table:compare_00` 中的 `warning` 列只有 `False` 值，因为 $\hat \kappa$ 的所有计算值都是 $< 0.7$，可以通过 {numref}`fig:loo_k_dummy` 自行验证。

我们在 {numref}`fig:loo_k_dummy` 中使用 $\hat \kappa > 0.09$ 的值对观测结果进行了注释，$0.09$ 只是任意选择的数字，也可以尝试使用其他截止值。比较 {numref}`fig:elpd_dummy` 和 {numref}`fig:loo_k_dummy` ，可以看到 $\hat \kappa$ 的最高值不一定是 ELPD 的最高值，反之亦然。

```{figure} figures/loo_k_dummy.png
:name: fig:loo_k_dummy
:width: 8.00in

$\hat \kappa$ 值。注释点对应于 $\hat \kappa > 0.09$ 的观测值，这是一个完全任意的阈值。
```

```{figure} figures/elpd_and_khat.png
:name: fig:elpd_and_khat
:width: 4.5in


符合“mA”、“mB”和“mC”的观测值的核密度估计。黑线代表每个观测值。注释观测与 {numref}`fig:elpd_dummy` 中突出显示的观测相同，但观测 78 以粗体注释，仅在 {numref}`fig:loo_k_dummy` 中突出显示。
```

(interpreting-p_loo-when-pareto-hat-kappa-is-large)= 

### 2.5.4 当帕累托 $\hat \kappa$ 比较大时解释 `p_loo`

如前所述，p_loo 可以粗略地解释为模型中估计的有效参数数量。然而，对于具有较大 $\hat\kappa$ 值的模型，我们可以获得一些额外的附加信息。如果 $\hat \kappa > 0.7$，那么将 p_loo 与 $p$ 的参数数量进行比较可以为我们提供一些额外的信息 {cite:p}`loo_glossary`：

 - 如果$p\_loo << p$，那么模型很可能被错误指定。你通常还会在后验预测检查中看到后验预测样本与观测结果匹配不佳的问题。

- 如果 $p\_loo < p$ 并且 $p$ 与观测次数相比相对较大（例如，$p > \frac{N}{5}$，其中 $N$ 是观测总数），这通常表明模型过于灵活或先验信息太少。因此，很难预测遗漏的观测结果。

- 如果 $p\_loo > p$，那么模型也很可能被严重错误指定。如果参数的数量是 $p << N$，那么后验预测检查也可能已经揭示了一些问题 [^18]。但是，如果 $p$ 与观测次数相比相对较大，例如 $p > \frac{N}{5}$，则你可能在后验预测检查中看不到问题。

你可以尝试修复模型错误指定的一些启发式方法：为模型添加更多结构，例如，添加非线性组件；使用不同的可能性，例如，使用像 NegativeBinomial 这样的过度分散的可能性而不是 Poisson 分布，或者使用混合可能性。

(loo-pit)= 

### 2.5.5 留一法--概率积分变换（ LOO-PIT ）

正如我们刚刚在 {ref}`elpd_plots` 和 {ref}`k-paretto` 模型比较部分中看到的，特别是 LOO，可以用于声明模型比另一个模型*更好* 之外的目的。我们可以比较模型来更好地理解它们。随着模型复杂性的增加，仅通过查看其数学定义或我们用来实现它的代码来理解它变得更加困难。因此，使用 LOO 或其他工具（如后验预测检查）比较模型可以帮助我们更好地理解它们。

对后验预测检查的一种评判是我们使用了两次数据，一次是为了拟合模型，一次是为了评判它。 LOO-PIT 图为这个问题提供了答案。主要思想是我们可以使用 LOO 作为交叉验证的快速可靠的近似值，以避免重复使用数据。 “PIT 部分”代表概率积分变换 [^19]，它是一维变换，如果我们变换该随机变量，我们可以从任何连续随机变量中得到 $\mathcal{U}(0, 1)$ 分布变量使用其自己的 CDF（有关详细信息，请参阅第 {ref}`loo_depth` 节）。在 LOO-PIT 中，我们不知道真正的 CDF，但我们用经验 CDF 来近似它。暂时搁置这些数学细节，带回家的信息是，对于一个经过良好校准的模型，我们应该期望一个近似均匀的分布。如果你正在经历似曾相识，请不要担心你没有超感官能力，这也不是矩阵中的故障。这可能听起来很熟悉，因为这实际上与我们在 {ref}`posterior_pd` 中使用函数 `az.plot_bpv(idata, kind="u_value")` 讨论的想法完全相同。

LOO-PIT 是通过将观测数据 $y$ 与后验预测数据 $\tilde y$ 进行比较来获得的。比较是逐点进行的。我们有：

```{math} 
p_i = P(\tilde y_i \leq y_i \mid y_{-i})
```

直观地说，当我们移除 $i$ 观测值时，LOO-PIT 正在计算后验预测数据 $\tilde y_i$ 的值低于观测数据 $y_i$ 的概率。因此，`az.plot_bpv(idata, kind="u_value")` 和 LOO-PIT 之间的区别在于，对于后者，我们几乎避免使用数据两次，但对图的总体解释是相同的。

{numref}`fig:loo_pit_dummy` 显示模型 `mA`、`mB` 和 `mC` 的 LOO-PIT。我们可以观测到，从模型“mA”的角度来看，低值的观测数据比预期的多，高值的数据少，即模型有偏差。相反，模型“mB”和“mC”似乎校准得很好。

```{figure} figures/loo_pit_dummy.png
:name: fig:loo_pit_dummy
:width: 8.00in

黑线是 LOO-PIT 的 KDE，即小于或等于观测数据的预测值的比例，根据每次观测计算。白线表示预期的均匀分布，灰带表示与所用数据集大小相同的数据集的预期偏差。
```

(model_averaging)= 

### 2.5.6 模型平均 

模型平均可以证明是关于模型不确定性的贝叶斯，因为我们是关于参数不确定性的贝叶斯。如果我们不能绝对确定 *a* 模型是 *the* 模型（通常我们不能），那么我们应该以某种方式将这种不确定性考虑到我们的分析中。考虑模型不确定性的一种方法是对所有考虑的模型进行加权平均，为似乎更好地解释或预测数据的模型赋予更大的权重。

对贝叶斯模型进行加权的*自然*方法是通过它们的边缘可能性，这被称为贝叶斯模型平均 {cite:p}`hoeting_bayesian_1999`。虽然这在理论上很有吸引力，但在实践中却存在问题（有关详细信息，请参阅第 {ref}`marginal_likelihood` 部分）。另一种方法是使用 LOO 的值来估计每个模型的权重。我们可以使用以下公式来做到这一点：

```{math} 
:label: eq_pseudo_avg

w_i = \frac {e^{-\Delta_i }} {\sum_j^k e^{-\Delta_j }} 
```

其中 $\Delta_i$ 是 LOO 的 $i$ 值与最高值之间的差异，假设我们使用的是对数尺度，这是 ArviZ 中的默认值。

这种方法称为伪贝叶斯模型平均，或类似 Akaike 的 [^20] 加权，是一种从 LOO[^21] 计算每个模型（给定一组固定模型）的相对概率的启发式方法。

看看分母如何只是一个标准化项，以确保权重总和为 $1$ 。公式 {eq}`eq_pseudo_avg` 提供的用于计算权重的解决方案是一种非常好的和简单的方法。一个主要的警告是它没有考虑 LOO 值计算中的不确定性。我们可以假设高斯近似计算标准误差，并相应地修改公式 {eq}`eq_pseudo_avg`。或者我们可以做一些更强大的事情，比如使用贝叶斯引导。

模型平均的另一个选择是堆叠预测分布 {cite:p}`yao_stacking_2018`。主要思想是将多个模型组合在一个元模型中，以使我们最小化元模型和*真实*生成模型之间的分歧。当使用对数评分规则时，这等效于计算：

```{math} 
:label: eq_stacking 

\max_{n} \frac{1}{n} \sum_{i=1}^{n}log\sum_{j=1}^{k} w_j p(y_i \mid y_{-i}, M_j) 
```
其中 $n$ 是数据点的数量，$k$ 是模型的数量。为了执行解决方案，我们将 $w$ 限制为 $w_j \ge 0$ 和 $\sum_{j=1}^{k} w_j = 1$。数量 $p(y_i \mid y_{-i}, M_j)$ 是 $M_j$ 模型的留一法预测分布。正如我们已经说过的计算它可能太昂贵了，因此在实践中我们可以使用 LOO 来近似它。

堆叠比伪贝叶斯模型平均具有更有趣的特性。我们可以从他们的定义中看出这一点，公式 {eq}`eq_pseudo_avg` 只是对每个模型独立于其余模型计算的权重的归一化。相反，在等式 {eq}`eq_stacking` 中，权重是通过最大化组合对数得分来计算的，即即使在伪贝叶斯模型平均中独立拟合模型时，权重的计算也会同时考虑所有模型。这有助于解释为什么模型“mB”的权重为 1，而“mC”的权重为 0（参见 {numref}`table:compare_00`），即使它们是非常相似的模型。为什么每个人的权重都不在 $0.5$ 左右？原因是根据堆叠过程，一旦 `mB` 包含在我们的比较模型集中，`mC` 不会提供新信息。换句话说，包括它将是多余的。

函数 `pm.sample_posterior_predictive_w(.)` 接受轨迹列表和权重列表，使我们能够轻松生成加权后验预测样本。权重可以从任何地方获取，但使用通过 `az.compare(., method="stacking")` 计算的权重很有意义。

(exercises2)= 
## 2.6 练习 

**2E1.** Using your own words, what are the main differences between prior predictive checks and posterior predictive checks? How are these empirical evaluations related to Equations [eq:prior_pred_dist](eq:prior_pred_dist) and [eq:post_pred_dist](eq:post_pred_dist).

**2E2.** Using your own words explain: ESS, $\hat R$ and MCSE. Focus your explanation on what these quantities are measuring and what potential issue with MCMC they are identifying.

**2E3.** ArviZ includes precomputed InferenceData objects for a few models. We are going to load an InferenceData object generated from a classical example in Bayesian statistic, the eight schools model {cite:p}`rubin_1981`. The InferenceData object includes prior samples, prior predictive samples and posterior samples. We can load the InferenceData object using the command `az.load_arviz_data("centered_eight")`. Use ArviZ to: 

1.  List all the groups available on the InferenceData object.

2. Identify the number of chains and the total number of posterior   samples.

3. Plot the posterior.

4. Plot the posterior predictive distribution.

5. Calculate the estimated mean of the parameters, and the Highest   Density Intervals.

If necessary check the ArviZ documentation to help you do these tasks <https://arviz-devs.github.io/arviz/> 

**2E4.** Load `az.load_arviz_data("non_centered_eight")`, which is a reparametrized version of the "centered_eight" model in the previous exercise. Use ArviZ to assess the MCMC sampling convergence for both models by using: 

1.  Autocorrelation plots 

2.  Rank plots.

3. $\hat R$ values.

Focus on the plots for the mu and tau parameters. What do these three different diagnostics show? Compare these to the InferenceData results loaded from `az.load_arviz_data("centered_eight")`. Do all three diagnostics tend to agree on which model is preferred? Which one of the models has better convergence diagnostics? 

**2E5.** InferenceData object can store statistics related to the sampling algorithm. You will find them in the `sample_stats` group, including divergences (`diverging`): 

1.  Count the number of divergences for "centered_eight" and   "non_centered_eight" models.

2. Use `az.plot_parallel` to identify where the divergences tend to   concentrate in the parameter space.

**2E6.** In the GitHub repository we have included an InferenceData object with a Poisson model and one with a NegativeBinomial, both models are fitted to the same dataset. Use `az.load_arviz_data(.)` to load them, and then use ArviZ functions to answer the following questions: 

1.  Which model provides a better fit to the data? Use the functions   `az.compare(.)` and `az.plot_compare(.)` 

2.  Explain why one model provides a better fit than the other. Use   `az.plot_ppc(.)` and `az.plot_loo_pit(.)` 

3.  Compare both models in terms of their pointwise ELPD values.

Identify the 5 observations with the largest (absolute) difference.

Which model is predicting them better? For which model p_loo is   closer to the actual number of parameters? Could you explain why?   Hint: the Poisson model has a single parameter that controls both   the variance and mean. Instead, the NegativeBinomial has two   parameters.

4. Diagnose LOO using the $\hat \kappa$ values. Is there any reason to   be concerned about the accuracy of LOO for this particular case? 

**2E7.** Reproduce {numref}`fig:posterior_predictive_many_examples`, but using `az.plot_loo(ecdf=True)` in place of `az.plot_bpv(.)`. Interpret the results. Hint: when using the option `ecdf=True`, instead of the LOO-PIT KDE you will get a plot of the difference between the LOO-PIT Empirical Cumulative Distribution Function (ECDF) and the Uniform CDF. The ideal plot will be one with a difference of zero.

**2E8.** In your own words explain why MCMC posterior estimation techniques need convergence diagnostics. In particular contrast these to the conjugate methods described in Section {ref}`conjugate_priors` which do not need those diagnostics. What is different about the two inference methods? 

**2E9.** Visit the ArviZ plot gallery at <https://arviz-devs.github.io/arviz/examples/index.html>. What diagnoses can you find there that are not covered in this chapter? From the documentation what is this diagnostic assessing? 

**2E10.** List some plots and numerical quantities that are useful at each step during the Bayesian workflow (shown visually in {numref}`fig:BayesianWorkflow`). Explain how they work and what they are assessing. Feel free to use anything you have seen in this chapter or in the ArviZ documentation.

1. Prior selection.

2. MCMC sampling.

3. Posterior predictions.

**2M11.** We want to model a football league with $N$ teams. As usual, we start with a simpler version of the model in mind, just a single team. We assume the scores are Poisson distributed according to a scoring rate $\mu$. We choose the prior $\text{Gamma}(0.5, 0.00001)$ because this is sometimes recommend as an objective prior.

+++

```{code-block} ipython3
:name: poisson_football
:caption: poisson_football

with pm.Model() as model:
    μ = pm.Gamma("μ", 0.5, 0.00001)
    score = pm.Poisson("score", μ)
    trace = pm.sample_prior_predictive()
```

+++

1.  Generate and plot the prior predictive distribution. How reasonable   it looks to you? 

2.  Use your knowledge of sports in order to refine the prior choice.

3. Instead of soccer you now want to model basketball. Could you come with a reasonable prior for that instance? Define the prior in a model and generate a prior predictive distribution to validate your intuition.

Hint: You can parameterize the Gamma distribution using the rate and   shape parameters as in Code Block   [poisson_football](poisson_football) or alternatively   using the mean and standard deviation.

**2M12.** In Code Block [metropolis_hastings](metropolis_hastings) from Chapter [1](chap1), change the value of `can_sd` and run the Metropolis sampler. Try values like 0.2 and 1.

1. Use ArviZ to compare the sampled values using diagnostics such as   the autocorrelation plot, trace plot and the ESS. Explain the   observed differences.

2. Modify Code Block [metropolis_hastings](metropolis_hastings) so you get more than one   independent chain. Use ArviZ to compute rank plots and $\hat R$.

**2M13.** Generate a random sample using `np.random.binomial(n=1, p=0.5, size=200)` and fit it using a Beta-Binomial model.

Use `pm.sample(., step=pm.Metropolis())` (Metropolis-Hastings sampler) and `pm.sample(.)` (the standard sampler). Compare the results in terms of the ESS, $\hat R$, autocorrelation, trace plots and rank plots.

Reading the PyMC3 logging statements what sampler is autoassigned? What is your conclusion about this sampler performance compared to Metropolis-Hastings? 

**2M14.** Generate your own example of a synthetic posterior with convergence issues, let us call it `bad_chains3`.

1. Explain why the synthetic posterior you generated is "bad". What   about it would we not want to see in an actual modeling scenario? 

2.  Run the same diagnostics we run in the book for `bad_chains0` and   `bad_chains1`. Compare your results with those in the book and   explain the differences and similarities.

3. Did the results of the diagnostics from the previous point made you   reconsider why `bad_chains3` is a "bad chain"? 

**2H15.** Generate a random sample using `np.random.binomial(n=1, p=0.5, size=200)` and fit it using a Beta-Binomial model.

1. Check that LOO-PIT is approximately Uniform.

2. Tweak the prior to make the model a bad fit and get a LOO-PIT that   is low for values closer to zero and high for values closer to one.

  Justify your prior choice.

3. Tweak the prior to make the model a bad fit and get a LOO-PIT that   is high for values closer to zero and low for values closer to one.

  Justify your prior choice.

4. Tweak the prior to make the model a bad fit and get a LOO-PIT that   is high for values close to 0.5 and low for values closer to zero   and one. Could you do it? Explain why.

**2H16.** Use PyMC3 to write a model with Normal likelihood. Use the following random samples as data and the following priors for the mean. Fix the standard deviation parameter in the likelihood at 1.

1. A random sample of size 200 from a $\mathcal{N}(0,1)$ and prior   distribution $\mathcal{N}(0,20)$ 

2.  A random sample of size 2 from a $\mathcal{N}(0,1)$ and prior   distribution $\mathcal{N}(0,20)$ 

3.  A random sample of size 200 from a $\mathcal{N}(0,1)$ and prior   distribution $\mathcal{N}(20 1)$ 

4.  A random sample of size 200 from a $\mathcal{U}(0,1)$ and prior   distribution $\mathcal{N}(10, 20)$ 

5.  A random sample of size 200 from a $\mathcal{HN}(0,1)$ and a prior   distribution $\mathcal{N}(10,20)$ 

Assess convergence by running the same diagnostics we run in the book for `bad_chains0` and `bad_chains1`. Compare your results with those in the book and explain the differences and similarities.

**2H17.** Each of the four sections in this chapter, prior predictive checks, posterior predictive checks, numerical inference diagnostics, and model comparison, detail a specific step in the Bayesian workflow. In your own words explain what the purpose of each step is, and conversely what is lacking if the step is omitted. What does each tell us about our statistical models? 

+++

[^1]: <https://www.countbayesie.com/blog/2015/2/18/bayes-theorem-with-lego> 

[^2]: We are omitting tasks related to obtaining the data in the first   place, but experimental design can be as critical if not more than   other aspects in the statistical analysis, see Chapter   [9](chap10).

[^3]: <https://arviz-devs.github.io/arviz/> 

[^4]: This example has been adapted from   <https://mc-stan.org/users/documentation/case-studies/golf.html> and   <https://docs.pymc.io/notebooks/putting_workflow.html> 

[^5]: The example has been adapted from {cite:p}`Gelman2020`.

[^6]: See Chapter [3](chap2) for details of the logistic   regression model.

[^7]: Posterior predictive checks are a very general idea. These figures   do not try to show the only available choices, just some of the   options offered by ArviZ.

[^8]: Unless you realize you need to collect data again, but that is   another story.

[^9]: Try <https://www.timeanddate.com/sun/ecuador/quito> 

[^10]: Do not confuse with the standard deviation of the MCSE for the   mean.

[^11]: Most useful and commonly used sampling methods for Bayesian   inference are variants of HMC, including for example, the default   method for continuous variables in PyMC3. For more details of this   method, see Section {ref}`hmc`).

[^12]: A function which is zero everywhere and infinite at zero.

[^13]: For a sampler like Sequential Monte Carlo, increasing the number   of draws also increases the number of particles, and thus it could   actually provide better convergence. See Section {ref}`smc_details`.

[^14]: Strictly speaking we should use probabilities for discrete   models, but that distinction rapidly becomes annoying in practice.

[^15]: In non-Bayesians contexts $\boldsymbol{\theta}$ is a point   estimate obtained, for example, by maximizing the likelihood.

[^16]: We are also computing samples from the posterior predictive   distribution to use them to compute LOO-PIT.

[^17]: At time of writing this book the method has not been yet   implemented in ArviZ, but it may be already available by the time   you are reading this.

[^18]: See the case study   <https://avehtari.github.io/modelselection/roaches.html> for an   example.

[^19]: A deeper give into Probability Integral Transform can be found in Section   {ref}`probability-integral-transform-pit`.

[^20]: The Akaike information criterion (AIC) is an estimator of the   generalization error, it is commonly used in frequentists   statistics, but their assumptions are generally not adequate enough   for general use with Bayesians models.

[^21]: This formula also works for WAIC [^22] and other information   criteria

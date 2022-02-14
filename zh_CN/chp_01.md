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

(chap1)=

# 第一章: 贝叶斯推断 

<style>p{text-indent:2em;2}</style>

现代贝叶斯统计主要使用计算机代码执行，这极大地改变了贝叶斯统计的执行方式。我们能够建立的模型的复杂性越来越大，必要的数学和计算技能障碍逐步降低。此外，迭代式建模过程也变得比以往更容易执行和更有价值。计算机方法的普及和流行确实很棒，但也需要更高的责任感。现在表达统计方法比以往任何时候都容易，但统计是一个微妙的领域，强大的计算方法并不会使统计神奇地消失。因此，具有良好理论背景知识（尤其是那些与实践相关的知识），对于有效应用统计方法非常重要。在本章中，我们介绍了其中一些基础概念和方法，还有很多内容将在本书其余部分进一步探索和扩展。

(bayesian_modeling)=

## 1.1 贝叶斯建模 

概念模型是对一个系统的表征，它由若干概念组合而成，用于帮助人们了解、理解或者模拟该模型所代表的对象或过程 {cite:p}`wikipedia_model_2020`。此外，模型是人为设计的表示，具有非常具体的目标，因此，讨论模型对给定问题的充分性通常比讨论其内在正确性更方便。模型的存在仅仅是为了帮助实现进一步的目标。

在设计新车时，汽车公司会制作一个物理模型，以帮助人们理解产品在制造时的外观。此时，一位具有汽车先验知识并且对如何使用模型具有很好估计的雕刻家，会寻找所需粘土等原材料，并使用手工工具雕刻物理模型。此物理模型能够帮助其他人了解设计的各个方面，例如：外观是否美观、汽车形状是否符合空气动力学等。这样的模型需要同时结合领域专业知识和雕刻知识才能得到想要的结果。此外，建模过程通常需要构建多个模型，这样做有可能是为了探索不同的选择，或者是因为需要与其他汽车设计团队成员互动来获得迭代式改进和扩展。如今，除了上述实体汽车模型外，通过计算机辅助设计软件制作的数字模型也很常见。

这种计算机模型相比物理模型存在一些优势。例如：与在实体汽车模型上进行测试相比，使用数字模型进行碰撞模拟更简单、更便宜；与团队内的同事共享模型也更加容易。

贝叶斯建模的思想与上述汽车建模非常相似。构建一个贝叶斯模型，需要结合领域专业知识和统计技能，以将知识整合到一些可计算的目标中，并确定结果的可用性。在贝叶斯建模场景中，*数据* 是“原材料”，而 *统计分布* 则是塑造模型的主要数学工具。需要同时结合领域专业知识和统计知识才能获得有用的结果。此外，贝叶斯实践者同样会以迭代方式构建多个模型，其中第一个基础模型主要用于帮助自身识别与思维的差距或模型存在的缺陷，然后将这些模型用于构建后续的改进模型和扩展模型。

此外，使用一种推断机制并不意味着阻碍其他推断机制发挥作用，就像汽车的物理模型不会阻碍数字模型发挥效用一样。同样的，现代贝叶斯实践者也有很多方式来表达想法、生成结果和分享输出，从而允许实践者及同行能够更广泛地推广其积极成果。

(bayesian-models)=

### 1.1.1 贝叶斯模型及其工作流程 

贝叶斯模型，无论是可计算模型还是其他模型，都有两个基本定义特征：

- 一是使用概率分布描述未知量 [^1] ，我们称这些未知量为参数 [^2]。

- 二是采用贝叶斯定理，以数据为条件更新参数的值，此过程也可以被视为概率的重新分配。

在高层次上，可以将贝叶斯模型的构建过程分为三个步骤：

1. **模型设计**：给定一些数据,以及关于如何生成这些数据的一些假设，我们通过 *组合（ Combing ）* 和 *转换（ Transforming ）* 随机变量来设计模型。

2. **模型推断**：利用贝叶斯定理，使所设计的模型能够适应数据。我们称此过程为*推断（ Inference ）*，其结果是获得了后验分布。我们希望数据能够减少所有可能参数值的不确定性，尽管这并不是任何贝叶斯模型都能保证的。

3. **模型评判**：我们根据不同标准来检查模型是否有意义，进而对模型进行评判，这些标准包括数据以及我们的专业领域知识等。模型来自于我们的设计，因此其自身天然具有不确定性，通过比较多个模型，在理论上会减少模型的不确定性。

如果你熟悉其他形式的建模工作，你就会认识到模型评判的重要性，以及迭代式地执行此三个步骤的必要性。例如：我们可能需要在任何给定点回溯历史步骤。这也许是因为我们引入了一个愚蠢的编程错误，或者在一些挑战之后我们找到了改进模型的方法，或者我们发现数据不像最初想象的那样可用，以至于我们需要收集更多数据甚至是不同类型的数据。
 
在本书中，我们将讨论这三个步骤中每一步的各种执行方法，并将学习如何将其扩展为更复杂的**贝叶斯工作流（ Bayesian Workflow ）**。我们认为“贝叶斯工作流”非常重要，以至于本书用了完整的一章（ [第 9 章](chap9) ）来审视和讨论此主题。

(Bayesian_inference)=

### 1.1.2 贝叶斯推断 

通俗地说，推断与 “根据证据和推理得出结论” 有关。贝叶斯推断是一种特殊形式的统计推断，它通过组合概率分布来获得其他概率分布。当我们已经观测到一些数据 $\boldsymbol{Y}$ 时，贝叶斯定理提供了用于估计参数 $\boldsymbol{\theta}$ 值的通用方法：

```{math} 
:label: eq:posterior_dist 

\underbrace{p(\boldsymbol{\theta} \mid \boldsymbol{Y})}_{\text{posterior}} = \frac{\overbrace{p(\boldsymbol{Y} \mid \boldsymbol{\theta})}^{\text{likelihood}}\; \overbrace{p(\boldsymbol{\theta})}^{\text{prior}}}{\underbrace{{p(\boldsymbol{Y})}}_{\text{marginal likelihood}}}
``` 

似然函数将观测数据与未知参数链接起来，而先验分布表示在观测到数据 $\boldsymbol{Y}$ 之前的参数不确定性 [^3]。通过将两者相乘，可以得到后验分布，即模型中所有未知参数的联合分布（ 以观测数据为条件 ）。 

{numref}`fig:bayesian_triad` 展示了一个任意的先验分布、似然函数，以及两者产生的后验分布 [^4] 。

```{figure} figures/bayesian_triad.png
:name: fig:bayesian_triad
:width: 8.00in 
 
左子图：一个假想的先验（黑色曲线）表明， $\theta = 0.5$ 的可能性更大，而其余值则呈现出线性对称的下降趋势；似然函数（灰色曲线）则表明， $\theta = 0.2$ 的值能更好地解释数据；而结果后验（蓝色曲线）先验和似然之间的折衷。图中省略了 $y$ 轴的值，因为我们只关心相对值。右子图：其功能与左子图相同，但 $y$ 轴采用了对数尺度。请注意，对数尺度能够保留相对性质，例如，两个子图中最大值和最小值所在位置并没有改变。对数尺度由于数值计算更稳定而成为计算机实现的首选。

``` 

请注意，虽然 $\boldsymbol{Y}$ 是观测数据，但它也被视为一个随机向量，因为它的值取决于特定实验结果 [^5]。为了获得后验分布，我们会将该随机向量的值视为固定在实际观测值上不变，出于此原因，一个常见的替代符号是使用 $y_{obs}$ ，而不是 $\boldsymbol{Y}$。

正如你看到的，在每个特定点上评估后验，在概念上非常简单，我们只需将 *先验* 乘以 *似然* 即可。然而，这不足以告诉我们后验概率的全部，因为我们不仅需要特定点处的绝对后验值，还需要其与周围点的相对值。后验分布的这种全局信息由归一化常数来表示，而且不幸的是，计算归一化常数 $p(\boldsymbol{Y})$ 非常困难。将边缘似然写成如下公式可能更容易理解这一点：

```{math} 
:label: eq:marginal_likelihood 

{p(\boldsymbol{Y}) = \int_{\boldsymbol{\Theta}} p(\boldsymbol{Y} \mid \boldsymbol{\theta})p(\boldsymbol{\theta}) d\boldsymbol{\theta}} 
``` 

其中 $\Theta$ 表示我们正在积分 $\theta$ 的所有可能值。

像这样计算积分非常困难（ 参见 {ref}`marginal_likelihood` 和一个有趣的 `XKCD 漫画` [^6] ）。尤其是对于大多数问题而言，根本无法给出边缘似然的封闭解，计算积分就更困难了。幸运的是，有一些数值方法可以帮助我们应对这一挑战。

在实践中，很多问题的解决并不需要计算边缘似然，此时将贝叶斯定理表示为比率形式比较常见：

```{math} 
:label: eq:proportional_bayes 

\underbrace{p(\boldsymbol{\theta} \mid \boldsymbol{Y})}_{\text{posterior}} \propto \overbrace{p(\boldsymbol{Y} \mid \boldsymbol{\theta})}^{\text{likelihood}}\; \overbrace{p(\boldsymbol{\theta})}^{\text{prior}}
```

::: {admonition} 关于符号的说明 

在本书中，我们使用相同的符号 $p(\cdot)$ 来表示不同的量，例如：似然函数和先验概率分布。这是对符号的轻微滥用（ 似然函数并非一定是某种概率分布 ），但这样做很有用。这种符号表示为贝叶斯公式中的所有量提供了相同的认识论地位。此外，它还反映出：即便似然不是严格意义上的概率密度函数，我们也不在乎，因为我们只考虑先验背景下的似然，反之亦然。换句话说，为了计算后验分布，我们将这两个量视为模型的同等必要元素。

::: 

贝叶斯统计的特点之一是：**后验（总是）是一个概率分布**。这使我们能够对参数做出概率性表述，例如参数 $\boldsymbol{\tau}$ 为正的概率是 $0.35$ 。或者 $\boldsymbol{\phi}$ 的最可能值是 $12$，并且有 $50\%$ 的机会介于 $10$ 和 $15$ 之间。

此外，可以将后验分布视为将模型与数据相结合的逻辑性结果，因此由其得出的概率陈述保证了在数学上的一致性。我们只需记住，所有这些好的数学性质只在柏拉图式的思想世界中有效。当我们从数学纯粹性转向现实世界中应用数学的混杂性时，必须始终牢记：**我们的结果不仅取决于数据，还取决于模型**。因此，不良数据和/或不良模型均可能导致无意义的陈述，即使其在数学上是一致的。我们必须始终对数据、模型和结果保持健康的怀疑精神。

为了使这一点更明确，我们可以更准确地表示贝叶斯定理如下：

```{math} 
p(\boldsymbol{\theta} \mid  \boldsymbol{Y}, M) \propto  p(\boldsymbol{Y} \mid \boldsymbol{\theta}, M) \; p(\boldsymbol{\theta}, M)
```

上式强调了我们的推断总是依赖于模型 $M$ 所做的假设。

一旦有了后验分布，我们就可以用它来推导其他感兴趣的量，而这通常以计算期望的方式实现，例如：

```{math} 
:label: eq:posterior_expectation 

J = \int f(\boldsymbol{\theta}) \; 
p(\boldsymbol{\theta} \mid \boldsymbol{Y}) \; 
d\boldsymbol{\theta} 
``` 

如果 $f$ 为恒等函数，则 $J$ 是 $\boldsymbol{\theta}$ 的均值 [^7]：

```{math} 
\bar{\boldsymbol{\theta}} = \int_{\boldsymbol{\Theta}} \boldsymbol{\theta}  p(\boldsymbol{\theta} \mid \boldsymbol{Y})  d\boldsymbol{\theta}
```

后验分布是贝叶斯统计的核心对象，但不是唯一重要的。除了对参数值进行推断外，我们可能还想对数据做出推断。这可以通过计算**先验预测分布**来完成：

```{math} 
:label: eq:prior_pred_dist 

p(\boldsymbol{Y}^\ast) =  \int_{\boldsymbol{\Theta}} p(\boldsymbol{Y^\ast} \mid \boldsymbol{\theta}) \; p(\boldsymbol{\theta}) \; d\boldsymbol{\theta} 

``` 

这是根据模型（先验和似然）做出的预期数据分布。给定模型，这是在实际看到任何观测数据 $\boldsymbol{Y}^\ast$ 之前所预期的数据。

请注意，公式 {eq}`eq:marginal_likelihood`（边缘似然）和公式 {eq}`eq:prior_pred_dist`（先验预测分布）看起来有些相似，但其含义是截然不同的。在边缘似然公式中，我们以观测数据 $Y$ 为条件，而在先验预测分布公式中，我们并不以观测数据为条件。最终，边缘似然表现为一个数字，而先验预测分布则是一个概率分布。
 
我们可以使用先前预测分布的样本作为评估和校准模型的一种方式。例如，我们可能会问 “人类身高的模型能否将人类身高预测为 $-1.5$ 米？”之类的问题。而这种问题在测量身高之前，我们就能认识其荒谬性。在本书后面章节中，我们将看到许多使用先验预测分布进行模型评估的示例，以及先验预测分布如何为后续建模选择提供有效或无效信息。

::: {admonition} 贝叶斯模型作为生成式模型 

采用概率视角进行建模导致了一个口头禅：*模型生成数据* {cite:p}`WestfallUnderstandingAdvancedStatistical2013`。我们认为此概念至关重要。一旦你将它内化，所有统计模型都会变得更加清晰，甚至是非贝叶斯模型。

此口头禅可以帮助我们创建新模型；如果数据是由模型生成的，那么我们可以 *仅* 通过考虑如何生成数据，来为数据创建适合的模型！此外，此口头禅并不是一个抽象概念，我们可以用先验预测分布作为其具体表示。如果重新审视贝叶斯建模的三个步骤，我们可以将它们重新调整为：编写先验预测分布 --> 添加数据以对其进行约束 ---> 检查结果是否有意义。

当然，必要时同样需要进行迭代。

::: 

另一个需要计算的量是**后验预测分布**：

```{math} 
:label: eq:post_pred_dist 

p(\tilde{\boldsymbol{Y}} \mid \boldsymbol{Y}) = \int_{\boldsymbol{\Theta}} p(\tilde{\boldsymbol{Y}} \mid \boldsymbol{\theta}) \, p(\boldsymbol{\theta} \mid \boldsymbol{Y}) \, d\boldsymbol{\theta} 

``` 

这是根据后验 $p(\boldsymbol{\theta} \mid \boldsymbol{Y})$ 预期的未来数据 $\tilde{\boldsymbol{Y}}$ 的分布；而后验又是模型（先验和似然）和观测数据的结果。用更常见的术语来说，这是模型在看到数据集 $\boldsymbol{Y}$ 后期望看到的未来数据，即这些是模型的预测。

从公式 {eq}`eq:post_pred_dist` 可以看到，预测是通过对参数的后验分布进行积分（或边缘化）来计算的。因此，以这种预测包含了估计的不确定性。

::: {admonition} 频率主义者眼中的贝叶斯后验

因为后验仅来自于模型和观测数据，所以我们并不是基于未观测到的事情做出陈述，而是基于内蕴数据生成过程得到的潜在观测做出陈述。

对未观测的事物做出推断通常是频率主义者的方法。但如果使用后验预测样本来检查模型，我们其实（部分）接受了频率主义者关于“未观测但潜在可观测的数据”的思想。

我们不仅对该想法满意，而且将在本书中看到此过程的多处示例。我们认为这是一个很棒的主意！

::: 

(sampling_methods_intro)=

## 1.2 一个自制的采样器 

公式 {eq}`eq:marginal_likelihood` 中的积分很多情况下没有封闭形式解，因此现代贝叶斯推断大多使用被称为 **通用推断引擎（ Universal Inference Engines ）** 的数值方法来实现（ 参见 {ref}`inference_methods` ) 。有许多经过良好测试的 Python 库能够提供此类数值方法，因此一般来说，贝叶斯实践者不太可能需要编写自己的通用推断引擎。

目前，编写自己的引擎通常只有两个理由：一是设计一个能够改进旧引擎的新引擎；二是正在学习当前引擎的工作原理。本章出于学习目的，我们将编写一个代码，但对于本书其余部分，主要使用 Python 库中的可用引擎。

有许多算法可以用于通用推断引擎，其中使用最广泛、功能最强的算法是`马尔可夫链蒙特卡洛方法（ MCMC ）` 。在较高层次上，所有 `MCMC 方法` 都使用样本来近似后验分布，而这些样本大多通过接受或拒绝来自某个提议分布的样本来生成。通过遵循某些规则和假设，我们有理论上的保证，能够获得非常近似后验分布的样本 [^8] 。因此，MCMC 方法也称为**采样器 （ Sampler ）**。所有这些方法都需要具有估计给定参数值时的先验和似然的能力。也就是说，即使不知道完整的后验形态，我们也能够逐点获取其概率密度。

此类算法之一是 `Metropolis-Hastings` {cite:p}`Metropolis1953, Hastings1970, Rosenbluth2003`。这并不是一个非常现代或有效的算法，但很容易被理解，因此为理解更复杂、更强大的其他方法奠定了基础 [^9] 。

`Metropolis-Hasting` 算法定义如下：

1. 在 $x_i$ 处初始化参数 $\boldsymbol{X}$ 的值

2. 使用提议分布 $q(x_{i + 1} \mid x_i)$ 从旧值 $x_i$ 生成新值 $x_{i + 1}$  [^10]。

3. 计算新值被接受的概率：

   ```{math} 
   :label: acceptance_prob
   
   p_a (x_{i + 1} \mid x_i) = \min \left (1, \frac{p(x_{i + 1}) \; q(x_i \mid x_{i + 1})} {p(x_i) \; q (x_{i + 1} \mid x_i)} \right)

   ``` 
   
4. 如果 $p_a > R$ 其中 $R \sim \mathcal{U}(0, 1)$，则保留新值，否则保留旧值。

5. 迭代 $2$ 到 $4$ 直到生成 *足够多* 的样本点。

`Metropolis 算法` 非常通用，可以在非贝叶斯应用中使用，但对于本书的内容，$p(x_i)$ 是在参数值 $x_i$ 处估计的后验密度。请注意，如果 $q$ 是对称分布，则 $q(x_i \mid x_{i + 1})$ 和 $q(x_{i + 1} \mid x_i)$ 将被消掉（ 从概念上讲，这意味着从 $x_{i+1}$ 转移到 $x_i$ 或从 $x_{i}$ 到 $x_{i+1}$ 具有相同的可能性 )，只留下在两个点处估计的后验比率。从公式 {eq}`acceptance_prob` 可以看到该算法将始终接受从低概率区到较高概率区的转移，并且将按照概率接受从高概率区到低概率区的移动。

另一个重要说明是: `Metropolis-Hastings 算法`不是一种优化方法！我们并不关心最大概率的点在哪儿，而是想探索 $p$ 分布（ 贝叶斯统计中主要指后验分布 ）。如果你稍加注意，就会发现此方法在达到最大概率区后并不会停止，而是在后续步骤中继续转移到较低概率区。

为了使事情更具象，让我们尝试求解 `Beta-Binomial 模型`。这可能是贝叶斯统计中最常见的示例，它用于对二值、互斥的结果进行建模，例如 `0` 或 `1`、`正`或`负`、`正面`或`反面`、`垃圾邮件`或`正常邮件`、`热狗`或`非热狗`、`健康`或`不健康`等。 `Beta-Binomial 模型` 经常被用作介绍贝叶斯统计基础知识的第一个示例，因为它是一个简单的模型，可以轻松求解和计算。在统计符号中 `Beta-Binomial 模型` 记为：

```{math} 
:label: eq:beta_binomial 

\begin{split}
\theta \sim &\; \text{Beta}(\alpha, \beta) \\
Y \sim &\; \text{Bin}(n=1, p=\theta) 
\end{split}
```

在公式 {eq}`eq:beta_binomial` 中，未知参数为 $\theta$ ，其先验分布为贝塔分布 $\text{Beta}(\alpha, \beta)$ ；我们假设数据的似然函数为二项分布 $\text{Bin}(n=1, p=\theta)$ 。在此模型中，成功的数量 $\theta$ 可以代表抛硬币时的正面比例、病亡率等量。该模型实际上存在解析形式解（ 参见 {ref}`conjugate_priors` 了解详细信息 ），但为了讲解，我们假设现在并不知道如何计算后验。因此我们将在 Python 代码中实现 `Metropolis-Hastings 算法`来以获得近似解。我们将在 SciPy 的统计函数支持下实现：

```{code-block} ipython3
:name: metropolis_hastings_sampler
:caption: Metropolis_Hhastings_采样器

def post(θ, Y, α=1, β=1):
    if 0 <= θ <= 1:
        prior = stats.beta(α, β).pdf(θ)
        like  = stats.bernoulli(θ).pmf(Y).prod()
        prob = like * prior
    else:
        prob = -np.inf
    return prob
```

实现后验推断还需要数据，为此可以随机生成一些合成数据（也称伪数据）。

```{code-block} ipython3 
:name: metropolis_hastings_sampler_rvs 
:caption: metropolis_hastings_sampler_rvs 

Y = stats.bernoulli(0.7).rvs(20)
```

运行 `Metropolis-Hastings 算法`的实现代码：


```{code-block} ipython3 
:name: metropolis_hastings 
:caption: metropolis_hastings 
:linenos: 

n_iters = 1000
can_sd = 0.05
α = β =  1
θ = 0.5
trace = {"θ":np.zeros(n_iters)}
p2 = post(θ, Y, α, β)

for iter in range(n_iters):
    θ_can = stats.norm(θ, can_sd).rvs(1)
    p1 = post(θ_can, Y, α, β)
    pa = p1 / p2

    if pa > stats.uniform(0, 1).rvs(1):
        θ = θ_can
        p2 = p1

    trace["θ"][iter] = θ
```

在代码 [metropolis_hastings](metropolis_hastings) 中，第 $9$ 行从标准差为 `can_sd` 的正态分布中采样来生成提议分布。第 $10$ 行在新生成的值 `θ_can` 处估计后验，第 $11$ 行计算接受概率。第 $17$ 行在 `trace` 数组中保存 `θ` 的值。

此值是新值还是重复上一个值，取决于第 $13$ 行的比较结果。

::: {admonition} 模棱两可的 MCMC 术语 

当使用马尔可夫链蒙特卡洛方法进行贝叶斯推断时，我们通常将其称为**MCMC 采样器**。在每次迭代中，我们从采样器中抽取一个随机样本，因此很自然地将 MCMC 的结果称为 *样本（ Samples ）* 或 *抽取（Draws）*。有些人将*样本*视为由一组*抽取*组成，而另外有一些人倾向于两者可以互换。
 
由于 MCMC 是按顺序抽取样本的，因此也会说*我们得到了一个抽取结果的 *链（ Chain ）*，或者简称为 MCMC 链。出于计算和诊断的原因，通常需要抽取许多链（ 我们在[第 2 章](chap1bis) 中讨论了如何做到这一点 ）。所有输出链，无论是单数还是复数，通常都被称为轨迹或简单地称为后验。不幸的是，口语并不精确，如果需要精确，最好的方法是查看代码以准确了解正在发生的事情。

::: 

请注意，代码 [metropolis_hastings](metropolis_hastings) 中的实现并非旨在提高效率，实际上生产级代码中会出现许多调整，例如计算对数规模的概率以避免溢出问题等（ 参见 {ref}`log_probabilities` 部分 ），或预先计算提议分布值。这些都是需要改变数学纯粹性以适应计算机实现的地方，也解释了为什么最好让专家来构建这些引擎。

同样， `can_sd` 的值是 `Metropolis-Hastings 算法` 的参数，而不是贝叶斯模型的参数。理论上该参数不应该影响算法的正确行为，但在实践中它又非常重要，因为方法效率肯定会受到其值的影响（ 参阅 {ref}`inference_methods`  ）。

回到示例，现在有了 MCMC 样本，我们想了解其形态。检查贝叶斯推断结果的一种常用方法是将每次迭代得到的采样值通过直方图或其他可视化工具绘制出来，以表示分布。例如，可以使用代码 [diy_trace_plot](diy_trace_plot) 中的代码来绘制 {numref}`fig:traceplot` [^11]：


```{code-block} ipython3 
:name: diy_trace_plot
:caption: diy_trace_plot

_, axes = plt.subplots(1,2, sharey=True) 
axes[1].hist(trace["θ"], color="0.5", orientation="horizontal", density=True)
```

```{figure} figures/traceplot.png
:name: fig:traceplot
:width: 8.00in 
 
左图中，每次迭代都会产生参数 $\theta$ 的采样值。右图为 $\theta$ 采样值的直方图。该直方图经过了旋转，以便更容易看出两个图之间的密切关系。左图显示了采样值的顺序序列，该序列其实就是所谓的马尔可夫链，而右图则显示了采样值的分布情况。
``` 

通常，计算一些数字汇总信息也很有用。我们将使用名为 `ArviZ` 的 Python 软件包 {cite:p}`Kumar2019` 来计算这些统计信息：

```{code-block} ipython3 
az.summary(trace, kind="stats", round_to=2)
```

```{list-table} 后验分布的汇总信息
:name: tab:posterior_summary 

* -
  - **mean**
  - **sd**
  - **hdi_3%**
  - **hdi_97%**
* - $\theta$
  - 0.69
  - 0.01
  - 0.52
  - 0.87
```

ArviZ 的函数 `summary` 计算参数 $\theta$ 的均值、标准差和 $94\%$ 最高密度区间 (HDI)。 最高密度区间（ HDI ）是包含给定概率密度（此处为 $94\%$ ）的最短区间 [^12] 。 {numref}`fig:plot_posterior` 为采用 `az.plot_posterior(trace)` 生成，其与 {numref}`tab:posterior_summary` 中的汇总信息非常相似。我们可以在代表整个后验分布的曲线顶部看到均值和最高密度区间。该曲线是使用 **核密度估计（ Kernel Density Estimator, KDE ）** 计算的，类似于直方图的平滑版本。

ArviZ 在许多绘图函数中都会使用 KDE，甚至在内部进行一些计算。

```{figure} figures/plot_posterior.png
:name: fig:plot_posterior
:width: 4in 
 
用后验图对代码 [metropolis_hastings](metropolis_hastings) 生成的样本进行可视化。后验分布使用 KDE 表示，均值和 $94\%$ 最高密度区间均在图中有所展示。

``` 

最高密度区间（ HDI ）是贝叶斯统计中的常见选择，像 $50\%$ 或 $95\%$ 这样的边界值也很常见。但是 ArviZ 的默认值为 $94\%$（ 或 $0.94$ ），如 {numref}`tab:posterior_summary` 和 {numref}`fig:plot_posterior` 中所示。这种选择的原因是 $94$ 接近广泛使用的 $95$，而同时这种差别可以友好地提醒观众，边界值的选择并没有什么特别之处 {cite:p}`mcelreath_2020` 。理想情况下，你应该选择一个适合需要的值 {cite:p}`Lakens2018`，或者至少承认你使用的是默认值。

(Automating_inference)=

## 1.3 人工建模与自动推断

我们可以利用 **概率编程语言 ( Probabilistic Programming Languages, PPL )** 的帮助，而不是编写自己的采样器从头定义自己的模型。概率编程语言允许用户使用代码表达贝叶斯模型，然后借助通用推断引擎以相当自动化的方式执行贝叶斯推断。简而言之，PPL 帮助贝叶斯实践者更多地关注模型构建本身，而不是数学和计算细节。

在过去的几十年中，此类工具的可用性大大提升了贝叶斯方法的普及度和实用性。不幸的是，这些通用推断引擎方法并不是真正通用的，因为它们无法有效地解决所有贝叶斯模型。现代贝叶斯实践者的部分工作是能够理解并且9999解决这些限制。

在本书中，我们将使用 `PyMC3` {cite:p}`Salvatier2016` 和 `TensorFlow Probability` {cite:p}`dillon2017tensorflow`。让我们使用 PyMC3 为公式 {eq}`eq:beta_binomial` 编写模型：

```{code-block} ipython3 
:name: beta_binom 
:caption: beta_binom 

# Declare a model in PyMC3 
with pm.Model() as model:     
   # Specify the prior distribution of unknown parameter
    θ = pm.Beta("θ", alpha=1, beta=1) 
   # Specify the likelihood distribution and condition on the observed data     
    y_obs = pm.Binomial("y_obs", n=1, p=θ, observed=Y) 

# Sample from the posterior distribution     
idata = pm.sample(1000, return_inferencedata=True) 
``` 
你可以自己检查一下这段代码的结果是否与之前自制采样器的结果一致，但在 PPL 支持下，工作量要少得多。如果你不熟悉 PyMC3 语法，现阶段只需关注代码注释中表达的每一行的意图。

由于我们已经在 PyMC3 语法中定义了模型，因此可以利用 `pm.model_to_graphviz(model)` 在代码 [beta_binom](beta_binom) 中生成模型的概率图表示（参见 {numref}`fig:BetaBinomModelGraphViz`）。

```{figure} figures/BetaBinomModelGraphViz.png
:name: fig:BetaBinomModelGraphViz
:width: 2in 
 
公式 {eq}`eq:beta_binomial` 和代码 [beta_binom](beta_binom) 中定义的贝叶斯模型的概率图表示。椭圆代表先验和似然，而 $20$ 表示观测次数。

```

概率编程语言不仅可以估计随机变量的对数概率以获得后验分布，还可以模拟各种预测分布。例如，代码 [predictive_distributions](predictive_distributions) 展示了如何使用 PyMC3 获得先验预测分布的 $1000$ 个样本，以及后验预测分布的 $1000$ 个样本。请注意，第一个函数有一个 `model` 参数，而第二个函数必须同时传递 `model` 和 `trace` 参数，这反映了先验预测分布仅需要模型，而后验预测分布不仅需要模型还需要后验分布。从先验预测分布和后验预测分布生成的样本分别在 {numref}`fig:quartet` 的顶部和底部子图中表示。


```{code-block} ipython3
:name: predictive_distributions
:caption: predictive_distributions

pred_dists = (pm.sample_prior_predictive(1000, model)["y_obs"],
              pm.sample_posterior_predictive(idata, 1000, model)["y_obs"])
```

公式 {eq}`eq:posterior_dist` 、 {eq}`eq:prior_pred_dist` 和 {eq}`eq:post_pred_dist` 清楚地将后验分布、先验预测分布和后验预测分布定义为三个不同的数学对象。后面的两个是数据分布，而第一个是参数的分布。{numref}`fig:quartet` 帮助我们可视化了这种差异，为了完整，该图还包括了先验分布。

::: {admonition} 用多种方式表示模型 

有许多方法可以表示统计模型的架构。以下没有特定的顺序：

- 口语和书面语言

- 概念图：{numref}`fig:BetaBinomModelGraphViz`。

- 数学符号：公式 {eq}`eq:beta_binomial`

- 计算机代码：代码 [beta_binom](beta_binom)

对于现代贝叶斯实践者来说，了解所有这些媒介很有用。它们是你在会谈、科学论文、与同事讨论时的手绘草图、互联网上的代码示例等中经常看到的格式。熟练地使用这些媒介，你将能够更好地理解以某种方式呈现的概念，然后将其应用到另一种方式中。例如，阅读一篇论文然后实现一个模型，或者在演讲中听到一种技术，然后能够为其写一篇博客。对个人而言，熟练程度会加快你的学习速度并提高与他人交流的能力。

::: 

```{figure} figures/Bayesian_quartet_distributions.png
:name: fig:quartet
:width: 8.00in 

自上至下，我们展示了：（1）参数 $\theta$ 的先验分布样本； (2) 成功总数的先验预测分布样本； (3) 参数 $\theta$ 的后验分布样本； (4) 成功总数的后验预测分布样本。在第一个和第三个图、第二个和第四个图之间分别共享 $x$ 轴和 $y$ 轴的坐标尺度。
``` 

正如我们已经提到的，后验预测分布考虑了估计结果的不确定性。

{numref}`fig:predictions_distributions` 表明根据均值得到的预测，比根据后验预测分布得到的预测更窄。该结果不仅对均值有效，对于其他任何点估计，会得到类似的图。

```{figure} figures/predictions_distributions.png
:name: fig:predictions_distributions
:width: 8.00in 
 
`Beta-Binomial 模型`的预测结果，使用后验均值所做的预测表示为灰色直方图，使用完整后验所做的预测（ 即后验预测分布 ）表示为蓝色直方图。
``` 

(make_prior_count)=

## 1.4 量化先验信息的几种选择

在贝叶斯统计中，不得不选择先验分布这件事，既是一种负担又是一种祝福，而我们认为这是必要的。如果你没有选择先验，那么最大可能是别人正在为你做这件事。当然，让别人替你做决定并不总是一件坏事。如果在正确的场景中应用并且意识到其局限性，许多非贝叶斯方法可能非常有用和有效。然而，我们坚信：**了解模型假设并灵活调整它们是有优势的，先验只是假设的一种形式**。

我们也明白，对于许多实践者来说，先验选择可能是怀疑、焦虑甚至沮丧的根源，尤其是对于新手，但不仅限于新人。寻找给定问题的最佳先验，是一个常见且完全有效的问题。但是除了没有最佳先验此结论之外，很难给出一个令人满意的答案。好在我们有一些默认值，可以作为迭代式建模工作流的起点。

在本节中，我们将讨论一些选择先验的一般性方法。此讨论或多或少遵循一个信息性的阶梯，从不包含任何信息的“空白”先验，到信息丰富的、信息尽可能多的先验。

本章关于先验的讨论更多是在理论方面的。在后续章节中，我们将讨论如何在更实际的环境中选择先验。

(conjugate_priors)=

### 1.4.1 共轭先验 

如果后验与先验属于同一分布族，则先验与似然共轭，或称其为似然的共轭先验。例如，如果似然是 `Poisson` 的并且先验为 `Gamma` 分布，那么后验也会是 `Gamma` 分布 [^13]。

从纯数学角度来看，**共轭先验**是最有利的选择，因为共轭先验和似然能够得到后验的封闭形式表达式，这允许我们用“纸笔”就可以分析和计算后验分布 [^14]。然而，从现代计算角度来看，共轭先验通常并不比其他方法好，主要原因是现代计算方法允许我们使用几乎任何先验进行推断，而不仅仅是那些在数学上方便的有限选择。尽管如此，共轭先验在学习贝叶斯推断时、以及在某些需要对后验使用解析表达式的情况下，可能仍然非常有用（ 参阅 {ref}`conjugate_case_study` 中的示例 ）。因此，我们会简要地讨论 `贝塔-二项式（ Beta-Binomial ）模型`中的解析共轭先验。顾名思义，该模型的似然为二项分布，其共轭先验为贝塔分布：

```{math} 
p(\theta \mid Y) \propto \overbrace{\frac{N!}{y!(N-y)!} \theta^y (1 - \theta)^{N-y}}^{\text{binomial-likelihood}} \: \overbrace{\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\, \theta^{\alpha-1}(1-\theta)^{\beta-1}}^{\text{beta.prior}} 
```

式中所有不包含 $\theta$ 的项都是关于 $\theta$ 不变的，可以省略它们，进而得到：

```{math} 
p(\theta \mid Y) \propto \overbrace{\theta^y (1 - \theta)^{N-y}}^{\text{binomial-likelihood}} \: \overbrace{ \theta^{\alpha-1}(1-\theta)^{\beta-1}}^{\text{beta.prior}}
```

重新组织公式，得到：

```{math} 
:label: eq:kernel_beta 
p(\theta \mid Y) \propto \theta^{\alpha-1+y}(1-\theta)^{\beta-1+N-y} 
``` 

如果想确保后验是一个正确的概率分布函数，我们需要添加一个归一化常数，以确保密度函数的积分为 $1$（参见 {ref}`cont_rvs` ）。请注意，公式 {eq}`eq:kernel_beta` 看起来像贝塔分布的核，由此添加贝塔分布的归一化常数后，得出 `Beta-Binomial 模型` 的后验分布为：

```{math} 
:label: eq:beta_posterior 
p(\theta \mid Y) \propto \frac{\Gamma(\alpha_{post}+\beta_{post})}{\Gamma(\alpha_{post})\Gamma(\beta_{post})} \theta^{\alpha_{post}-1}(1-\theta)^{\beta_{post}-1} = \text{Beta}(\alpha_{post}, \beta_{post}) 
``` 

其中 $\alpha_{post} = \alpha+y$ 和 $\beta_{post} = \beta+N-y$。

由于 `Beta-Binomial 模型`的后验也是贝塔分布，我们可以使用其作为下一步贝叶斯分析的先验。这意味着，*一次使用整个数据集* 和 *一次使用一个数据点* 将获得相同的结果。例如，{numref}`fig:beta_binomial_update` 的前四个子图显示了从 $0$ 到 $1$、$2$ 和 $3$ 试验时，不同的先验是如何更新的。如果遵循此顺序，或者如果从 $0$ 次试验跳到 $3$ 次试验（ 或者，实际上是 $n$ 次试验 ），得到的结果是相同的。

从 {numref}`fig:beta_binomial_update` 中还可以看到很多其他有趣的事情。例如，随着试验次数增加，后验的宽度越来越小，即不确定性越来越低。子图 $3$ 和 $5$ 显示了 $2$ 次试验成功 $1$ 次和 $12$ 次试验成功 $6$ 次的结果。两种情况的采样比例估计 $\hat \theta = \frac{y}{n}$（ 黑点 ）相同，均为 $0.5$（ 后验分布的众数也是 $0.5$ ），不过子图 5 中中的后验宽度相对聚集，这反映了观测数量更大，不确定性更低。最后，我们可以观察到：随着观测次数增加，不同先验最终可以收敛到相同的后验分布。在无限数据的情况下，后验与用于计算它的先验无关，不同先验得到的后验最终将在 $\hat \theta = \frac{y}{n}$ 处具有所有密度。

```{code-block} ipython3
:name: binomial_update
:caption: binomial_update

_, axes = plt.subplots(2,3, sharey=True, sharex=True)
axes = np.ravel(axes)

n_trials = [0, 1, 2, 3, 12, 180]
success = [0, 1, 1, 1, 6, 59]
data = zip(n_trials, success)

beta_params = [(0.5, 0.5), (1, 1), (10, 10)]
θ = np.linspace(0, 1, 1500)
for idx, (N, y) in enumerate(data):
    s_n = ("s" if (N > 1) else "")
    for jdx, (a_prior, b_prior) in enumerate(beta_params):
        p_theta_given_y = stats.beta.pdf(θ, a_prior + y, b_prior + N - y)

        axes[idx].plot(θ, p_theta_given_y, lw=4, color=viridish[jdx])
        axes[idx].set_yticks([])
        axes[idx].set_ylim(0, 12)
        axes[idx].plot(np.divide(y, N), 0, color="k", marker="o", ms=12)
        axes[idx].set_title(f"{N:4d} trial{s_n} {y:4d} success")
```

```{figure} figures/beta_binomial_update.png
:name: fig:beta_binomial_update
:width: 8.00in 
 
从 $3$ 个不同的先验开始连续更新先验并增加试验次数。黑点代表采样比例的估计 $\hat \theta = \frac{y}{n}$。
``` 

贝塔分布的均值是 $\frac{\alpha}{\alpha + \beta}$ ，因此先验均值是：

```{math} 
\mathbb{E}[\theta]  = \frac{\alpha}{\alpha + \beta}
```

后验均值为：

```{math} 
:label: eq:beta_binom_mean 

\mathbb{E}[\theta \mid Y]  = \frac{\alpha + y}{\alpha + \beta + n}
``` 

可以看到，如果 $n$ 的值相对于 $\alpha$ 和 $\beta$ 的值较小，那么后验均值将更接近于先验均值。也就是说，先验对结果的贡献大于数据。如果出现相反的情况，则后验均值将更接近采样比例的估计 $\hat \theta = \frac{y}{n}$ ，实际上在 $n \rightarrow \infty$ 的情况下，后验均值将与 $\alpha$ 和 $\beta$ 的先验无关，最终都完美匹配样本比例。

对于 `Beta-Binomial 模型`，后验众数为：

```{math}
:label: eq:beta_binom_mode

\operatorname*{argmax}_{\theta}{[\theta \mid Y]}  = \frac{\alpha + y - 1}{\alpha + \beta + n - 2}

```

可以看到，当先验为 $\text{Beta}(\alpha\!=\!1, \beta\!=\!1)$ ( 即均匀分布 ) 时，后验众数在数值上等价于采样比例的估计 $\hat \theta = \frac{y}{n}$。后验众数通常被称为**最大后验( MAP )**  值。此结果并非 `Beta-Binomial 模型`独有。事实上，许多非贝叶斯方法的结果都可以被理解为贝叶斯方法在特定先验条件下的最大后验 [^15]。

将公式 {eq}`eq:beta_binom_mean` 与采样比例 $\frac{y}{n}$ 进行比较。贝叶斯估计器将 $\alpha$ 添加到成功次数，将 $\alpha + \beta$ 添加到试验次数。这使得 $\beta$ 成为失败的次数。从此意义上说，我们可以将先验参数视为*伪计数*。先验 $\text{Beta}(1, 1)$ 等价于进行两次试验，$1$ 次成功，$1$ 次失败。从概念上讲，贝塔分布的形状由参数 $\alpha$ 和 $\beta$ 控制，观测数据会更新先验，以便使贝塔分布的形状更接近、更窄地移动到大多数观测值。对于 $\alpha < 1$ 和/或 $\beta < 1$ 的值，先验的解释变得有点奇怪，因为字面解释会说先验 $\text{Beta}(0.5, 0.5)$ 对应于一次试验，半次失败，半次成功，或者可能是一次结果未定的试验。诡异！

(objective-priors)=

### 1.4.2 客观先验 

在没有先验信息的情况下，遵循 *无差别原则* 听起来似乎更合理。此原则基本上是说，如果你没有关于某个问题的信息，那么你没有任何理由相信一个结果会比任何其他结果更有可能发生。

在贝叶斯统计背景下，这一原则激发了 **客观先验（ Objective Priors ）** 的研究和应用。这些是生成“对给定分析影响最小的先验”的系统方法。有些统计学者偏爱客观先验，因为此类先验消除了先验的主观性。但其实这并没有消除其他来源的主观性，例如：似然的选择、数据的选择、问题的选择等等。

获得客观先验的一种方法是 `Jeffreys 先验 ( JP )`。此类先验通常被称为*非信息性*，即便其总是以某种方式提供了信息。更好的描述是说： `Jeffreys 先验 ( JP )`具有在**重参数化**下保持不变性的性质，即以不同但在数学上等效的方式编写表达式。让我们用一个例子来解释这究竟意味着什么。

假设 `Alice` 具有参数为 $\theta$ 的二项似然，她选择某种先验并计算后验。而她的朋友 `Bob`，对同一问题感兴趣，但不关心成功的次数 $\theta$ ，而关心成功的**赔率（ odds ）**，即 $\kappa$ ，$\kappa = \frac{\ theta}{1-\theta}$ 。 `Bob` 有两种选择：一是使用 `Alice` 在 $\theta$ 上的后验来计算 $\kappa$ [^16] ，二是选择 $\kappa$ 上的先验来计算自己的后验。 `Jeffreys 先验` 保证：如果 `Alice` 和 `Bob` 都使用了 Jeffreys 先验，那么无论 `Bob` 做出哪种选择，都会得到相同的结果。从此意义上说，最终结果相对于所选择的参数具有不变性。此解释的一个推论是，除非使用 `Jeffreys 先验`，否则无法保证模型的两种（或更多）参数化必然会导致一致的后验。

对于一维 $\theta$ 的情况， `Jeffreys 先验` 为：

```{math} 
:label: eq:Jeffreys_prior0 

p(\theta) \propto \sqrt{I(\theta)} 
``` 

其中 $I(\theta)$ 为预期 `Fisher 信息`：

```{math} 
:label: eq:Jeffreys_prior 
I(\theta) = - \mathbb{E_{Y}}\left[\frac{d^2}{d\theta^2} \log p(Y \mid \theta)\right] 
``` 

一旦实践者确定了似然函数 $p(Y \mid \theta)$ ，那么 `Jeffreys 先验` 就会自动确定，从而消除了对先前选择的任何讨论。

有关 `Alice` 和 `Bob` 问题 `Jeffreys 先验` 的详细推导，请参阅 {ref}`Jeffreys_prior_derivation` 。

如果想在这里跳过细节，则 `Alice` 的 `Jeffreys 先验` 为：

```{math} 
\begin{aligned}
 p(\theta) \propto \theta^{-0.5} (1-\theta)^{-0.5}
\end{aligned}
```

这变成了 $\text{Beta}(0.5, 0.5)$ 分布的核。这是一个 $U$ 形分布，如 {numref}`fig:Jeffrey_priors` 的左上子图所示。

对于 `Bob` 来说， `Jeffreys 先验` 为：

```{math}
:label: fig:bob_prior
p(\kappa) \propto \kappa^{-0.5} (1 + \kappa)^{-1}

```

这是一个半 $U$ 形分布，定义在 $[0, \infty)$ 区间中，参见 {numref}`fig:Jeffrey_priors` 中的右上角子图。称其为半 $U$ 形可能有点奇怪，但实际上，它是贝塔分布的近亲，即参数为 $\alpha=\beta=0.5$ 时的 `Beta-prime 分布` 核。

```{figure} figures/Jeffrey_priors.png
:name: fig:Jeffrey_priors
:width: 8.00in 
 
上图：根据成功次数 $\theta$（ 左 ）或几率 $\kappa$（ 右 ）参数化的二项似然的 Jeffreys 先验（ 未归一化 ）。下图：根据成功次数 $\theta$（ 左 ）或几率 $\kappa$（ 右 ）参数化的二项似然的 Jeffreys 后验（ 未归一化 ）。后验之间的箭头表示，通过应用变量规则变化，后验之间可相互转换（ 详细信息参阅 {ref}`transformations` ）。
``` 

请注意，公式 {eq}`eq:Jeffreys_prior` 中的期望是关于 $Y \mid \theta$ 的，这是在样本空间上的期望。这意味着为了获得 `Jeffreys 先验`，我们需要对所有可能的实验结果求平均，这违反了似然原则 [^17]，因为关于 $\theta$ 的推断不仅取决于手头数据，还取决于潜在（但尚未）观测的数据集。

`Jeffreys 先验` 可以是不恰当的先验，即它可能不会积分为 $1$ 。例如，已知方差的高斯分布，其均值参数的 `Jeffreys 先验` 在整个实数轴上均匀分布。但只要我们能够验证，这些不恰当的先验与似然组合后，能够产生恰当（ 即积分为 $1$ ）的后验，则这些不恰当的先验就是可以使用的。

另外还要注意，我们不能从不恰当的先验中抽取随机样本（ 即此类先验是非生成式的 ），这会造成许多能够帮助我们做模型推断的工具失效。

`Jeffreys 先验` 并不是获得客观先验的唯一方法。另一种途径是通过最大化先验和后验之间的`预期 KL 散度`来获得先验（ 参见 {ref}`DKL` ）。此类先验被称为 `Bernardo 参考先验`，之所以是客观性的，是因为这些先验是 “允许数据将最大量信息带入后验” 的先验。 `Bernardo 参考先验`和 `Jeffreys 先验` 不必一致。

此外，对于某些复杂模型，可能不存在客观先验或难以推导出客观先验。


(maximum-entropy-priors)=

### 1.4.3 最大熵先验 

另一种证明先验选择合理性的方法是选择具有最大熵的先验。此时，如果我们对参数可取的合理值完全无区别对待，那么此先验的结果就是合理值范围内的均匀分布 [^18]。但是，当我们对可取的合理值并非无动于衷呢？例如，我们可能知道参数仅限于 $[0, \infty)$ 区间。我们能否得到具有最大熵同时满足给定约束的先验？是的，这正是最大熵先验背后的思想。在文献中，当人们谈论最大熵原理时，通常会找到 `MaxEnt` 这个词。

为了获得最大熵先验，需要求解一个包含一组约束条件的优化问题。从数学上讲，这可以使用`拉格朗日乘数`来实现。我们会使用几个代码示例来感性认识它，而不是采用形式化的证明。

{numref}`fig:max_entropy` 展示了通过最大化熵获得的 $3$ 个分布。紫色分布是在没有约束条件下获得的，我们会发现这确实是 {ref}`entropy` 中所预期的均匀分布。如果我们对问题一无所知，那么所有事件都是同样可能的。第二个青色的分布是在知道分布均值（ 本例中为 $1.5$ ）的约束下获得的，这是一个类似指数的分布。最后一个黄绿色的分布是在已知 $3$ 和 $4$ 的出现概率为 $0.8$ 这个约束下获得的。

> 注意：如果检查代码 [max_ent_priors](max_ent_priors)，你会看到所有分布都是在两个基本约束条件下计算的，一是概率只能在 $[0, 1]$ 区间中取值，二是总概率必须为 $1$ 。由于它们是有效概率分布的通用约束，因此可以被视为固有约束。出于此原因，我们经常默认它们，进而称 {numref}`fig:max_entropy` 中的紫色分布是在无约束条件下获得的。
 

```{code-block} ipython3
:name: max_ent_priors
:caption: max_ent_priors
cons = [[{"type": "eq", "fun": lambda x: np.sum(x) - 1}],
        [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
         {"type": "eq", "fun": lambda x: 1.5 - np.sum(x * np.arange(1, 7))}],
        [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
         {"type": "eq", "fun": lambda x: np.sum(x[[2, 3]]) - 0.8}]]

max_ent = []
for i, c in enumerate(cons):
    val = minimize(lambda x: -entropy(x), x0=[1/6]*6, bounds=[(0., 1.)] * 6,
                   constraints=c)['x']
    max_ent.append(entropy(val))
    plt.plot(np.arange(1, 7), val, 'o--', color=viridish[i], lw=2.5)
plt.xlabel("$t$")
plt.ylabel("$p(t)$")
```

```{figure} figures/max_entropy.png
:name: fig:max_entropy
:width: 8.00in 

在不同约束下通过最大化熵获得的离散分布。我们使用 `scipy.stats` 的函数 `entropy` 来估计这些分布。请注意添加约束极大地改变了分布。
``` 

我们可以将最大熵原理视为**在给定约束下选择最平坦分布的过程**，在贝叶斯统计中扩展为**在给定约束下选择最平坦先验分布的过程**。在 {numref}`fig:max_entropy` 中，均匀分布是最平坦的分布，但请注意，一旦引入 “ $3$ 和 $4$ 的出现概率为 80% ” 的约束，绿色分布就变成了最平坦的分布。

请注意本例中 $3$ 和 $4$ 的出现概率都是 $0.4$ ，尽管要获得 $0.8$ 的目标值，有很多种选择，例如： $0+0.8$ 、 $0.7+0.1$ 、 $0.312+0.488$ 等。另请注意，值 $1$ 、 $2 、 $5$ 和 $6$ 也有类似情况，它们的总概率为 $0.2$ ，而本例中也仅使用了均匀分布（ 每个值的概率均为 $0.05$ ）。

现在看一下类似指数的曲线，它看起来肯定不是很平坦，但再次注意到其他选择将更不平坦且更集中，例如，分别以 $50\%$ 的概率获得 $1$ 和 $2$（ 因此 $3$ 至 $6$ 的概率为 $0$ ），这也将有 $1.5$ 的期望值。

```python
ite = 100_000
entropies = np.zeros((3, ite))
for idx in range(ite):
    rnds = np.zeros(6)
    total = 0
    x_ = np.random.choice(np.arange(1, 7), size=6, replace=False)
    for i in x_[:-1]:
        rnd = np.random.uniform(0, 1-total)
        rnds[i-1] = rnd
        total = rnds.sum()
    rnds[-1] = 1 - rnds[:-1].sum()
    H = entropy(rnds)
    entropies[0, idx] = H
    if abs(1.5 - np.sum(rnds * x_)) < 0.01:
        entropies[1, idx] = H
    prob_34 = sum(rnds[np.argwhere((x_ == 3) | (x_ == 4)).ravel()])
    if abs(0.8 - prob_34) < 0.01:
        entropies[2, idx] = H
```

{numref}`fig:max_entropy_vs_random_dist` 显示了在与 {numref}`fig:max_entropy` 中 $3$ 个分布完全相同条件下，为随机生成的样本计算的熵的分布。垂直虚线表示 {numref}`fig:max_entropy_vs_random_dist` 中曲线的熵。虽然这不是一个证明，但实验似乎表明没有分布会比 {numref}`fig:max_entropy_vs_random_dist` 中的分布具有更高的熵，这与理论告诉我们的完全一致。

```{figure} figures/max_entropy_vs_random_dist.png
:name: fig:max_entropy_vs_random_dist
:width: 8.00in 
 
一组随机生成的分布的熵的分布。垂直虚线表示具有最大熵分布的值，使用代码 [max_ent_priors](max_ent_priors) 计算。可以看到，没有一个随机生成的分布的熵大于具有最大熵分布的熵，尽管这不是形式化的证明，但结果是令人放心的。
``` 

在特定约束下，具有最大熵的分布封闭是 [^19]：

- 无约束时：均匀分布（连续的或离散的，取决于随机变量类型）

- 正均值，支持 $[0, \infty)$ ：指数分布

- 绝对值均值，支持 $(-\infty, \infty)$：拉普拉斯（也称为双指数）

- 给定均值和方差，支持 $(-\infty, \infty)$：正态分布

- 给定均值和方差，支持 $[-\pi, \pi]$：Von Mises

- 只有两个无序结果和一个恒定均值：二项分布，或者存在罕见事件的泊松分布（ 泊松可以看作二项分布的特例 ）

有趣的是，考虑到模型约束，许多传统上的广义线性模型（ 如 [第 3 章](chap2) 中描述的模型 ）都是使用最大熵分布来定义的。与客观先验类似，`MaxEnt 先验`可能不存在或难以推导。


(weakly-informative-priors-and-regularization-priors)=

### 1.4.4 弱信息性先验和正则化先验

在前面部分中，我们使用一般性过程来生成模糊的、无信息的先验，旨在不将*太多*信息放入分析中。这些一般性过程还提供了“以某种方式”自动生成先验的方法。这两点特征听起来非常具有吸引力，而且在实际上被大量贝叶斯实践者和理论家所采用。
 
但在本书中，我们不会过分依赖这些先验。我们认为先验选择应该取决于上下文，这意味着来自特定问题的细节甚至给定科学领域的特质，都可以为选择先验提供信息。虽然 `MaxEnt 先验`能够包含其中一些约束，但我们还可以更加靠近信息性先验频谱的信息端，我们可以用弱信息先验来实现这一点。

构造弱信息先验的方法通常在数学上没有像 `Jeffreys 先验` 或 `MaxEnt 先验` 那样的良好定义。相反，它们更多是*经验主义的*和*模型驱动的*，也就是说，它们是通过领域专业知识和模型本身的组合来定义的。对于很多问题，我们经常有关于参数可取值的信息，这些信息往往来自于参数的物理意义，例如身高必须是正数。我们甚至可以从以前实验或观测中得到参数的合理取值范围。我们或许有充分理由证明一个值应该接近零或高于某个预定义的下界。我们可以使用这些信息来为分析提供微弱的信息，同时保持一定程度的无知。

再次使用 `Beta-Binomial` 示例，{numref}`fig:prior_informativeness_spectrum` 显示了四个可选先验。其中两个是 `Jeffreys 先验` 和最大熵先验。另外一种是弱信息先验，它优先考虑 $\theta=0.5$ 的值，同时保持对其他值很宽泛或相对模糊。最后一个是信息丰富的先验，以 $\theta=0.8$ 为中心 [^20] 。如果从理论、之前的实验、观测数据等中能够获得高质量信息，则信息性先验是一个有效的选择。信息性先验可以传达大量信息，因此它们通常需要比其他先验更强的理由。正如 `Carl Sagan` 常说的 “非凡主张需要非凡的证据” {cite:p}`Deming2016`。重要的：先验的信息量取决于模型和模型上下文。一个在某种情境中的无信息先验，可能在另一个情境中变得非常有用 {cite:p}`LikehoodandPrior` 。例如，如果以米为单位对成年人的平均身高进行建模，则 $\mathcal{N}(2,1)$ 的先验可以被认为是无信息的，但如果估计长颈鹿的高度，则该先验变得信息量非常大，因为现实中的长颈鹿的高度与人类身高相差很大。

```{figure} figures/prior_informativeness_spectrum.png
:name: fig:prior_informativeness_spectrum
:width: 8.00in 
 
先验信息谱：虽然 `Jeffrey 先验`和 `MaxEnt 先验`是为二项似然唯一定义的，但弱信息先验和信息性先验不是，而是取决于之前的信息和实践者的建模决策。
``` 

因为弱信息先验可以将后验分布保持在一定的合理范围内，所以它们也被称为正则化先验。正则化是一种添加信息的过程，目的是解决不适定问题或减少过拟合的机会，先验提供了一种执行正则化的原则方法。
 
在本书中，我们经常使用弱信息先验。有时会在没有太多理由的情况下在模型中使用先验，仅仅是因为示例的重点可能与贝叶斯建模工作流程的其他方面有关。但我们也会展示一些使用先验预测检查来校准先验分布的例子。

::: {admonition} 过拟合（ Overfitting ） 

当模型生成的预测非常接近用于拟合它的有限数据集时，就会发生过拟合，但它无法拟合新数据和/或不能很好地预测未来的观测结果。也就是说，它未能将其预测推广到更广泛的可观测结果。过拟合的对应物是欠拟合，即模型未能充分捕捉数据的底层结构。我们将在 {ref}`model_cmp` 和 {ref}`information_criterion` 部分讨论此主题的更多信息。

::: 

(using-prior-predictive-distributions-to-assess-priors)=

### 1.4.5 使用先验预测分布评估先验 

在评估先验选择时，{ref}`Automating_inference` 中显示的先验预测分布是一个方便的工具。

通过从先验预测分布中采样，计算机将*在参数空间中的选择*转换为*在观测空间中的样本*。考虑观测值通常比考虑模型参数更容易，这使模型评估变得更容易。遵循`Beta-Binomial` 模型，而不是判断 $\theta$ 的特定值是否合理，先验预测分布允许我们判断特定数量的成功是否合理。这对于参数通过许多数学运算或多个先验相互交互的复杂模型更加有用。

最后，计算先验预测分布可以帮助我们确保模型已经正确编写，并且能够在概率编程语言中运行，甚至可以帮助我们调试模型。

在接下来的章节中，我们将看到更具体的示例，说明如何推断先验预测样本并使用它们来选择合理的先验。

(exercises1)=

## 1.5 练习

Problems are labeled Easy (E), Medium (M), and Hard (H).

**1E1.** As we discussed, models are artificial representations used to help define and understand an object or process.

However, no model is able to perfectly replicate what it represents and thus is deficient in some way. In this book we focus on a particular type of models, statistical models. What are other types of models you can think of? How do they aid understanding of the thing that is being modeled? How are they deficient? 

**1E2.** Match each of these verbal descriptions to their corresponding mathematical expression: 

1.  The probability of a parameter given the observed data 

2.  The distribution of parameters before seeing any data 

3.  The plausibility of the observed data given a parameter value 

4.  The probability of an unseen observation given the observed data 

5.  The probability of an unseen observation before seeing any data 

**1E3.** From the following expressions, which one corresponds to the sentence, The probability of being sunny given that it is July 9th of 1816? 

1.  $p(\text{sunny})$ 

2.  $p(\text{sunny} \mid \text{July})$ 

3.  $p(\text{sunny} \mid \text{July 9th of 1816})$ 

4.  $p(\text{July 9th of 1816} \mid \text{sunny})$ 

5.  $p(\text{sunny}, \text{July 9th of 1816}) / p(\text{July 9th of 1816})$ 

**1E4.** Show that the probability of choosing a human at random and picking the Pope is not the same as the probability of the Pope being human. In the animated series Futurama, the (Space) Pope is a reptile. How does this change your previous calculations? 

**1E5.** Sketch what the distribution of possible observed values could be for the following cases: 

1.  The number of people visiting your local cafe assuming Poisson   distribution 

2.  The weight of adult dogs in kilograms assuming a Uniform   distribution 

3.  The weight of adult elephants in kilograms assuming Normal   distribution 

4.  The weight of adult humans in pounds assuming skew Normal   distribution 

**1E6.** For each example in the previous exercise, use SciPy to specify the distribution in Python. Pick parameters that you believe are reasonable, take a random sample of size 1000, and plot the resulting distribution. Does this distribution look reasonable given your domain knowledge? If not adjust the parameters and repeat the process until they seem reasonable.

**1E7.** Compare priors $\text{Beta}(0.5, 0.5)$, $\text{Beta}(1, 1)$, $\text{Beta}(1, 4)$. How do the priors differ in terms of shape? 

**1E8**. Rerun Code Block [binomial_update](binomial_update) but using two Beta-priors of your choice. Hint: you may what to try priors with $\alpha \neq \beta$ like $\text{Beta}(2, 5)$.

**1E9.** Try to come up with new constraints in order to obtain new Max-Ent distributions (Code Block [max_ent_priors](max_ent_priors)) 

**1E10.** In Code Block [metropolis_hastings](metropolis_hastings), change the value of `can_sd` and run the Metropolis-Hastings sampler. Try values like 0.001 and 1.

1. Compute the mean, SD, and HDI and compare the values with those in   the book (computed using `can_sd=0.05`). How different are the   estimates? 

2.  Use the function `az.plot_posterior`.

**1E11.** You need to estimate the weights of blue whales, humans, and mice. You assume they are normally distributed, and you set the same prior $\mathcal{HN}(200\text{kg})$ for the variance. What type of prior is this for adult blue whales? Strongly informative, weakly informative, or non-informative? What about for mice and for humans? How does informativeness of the prior correspond to our real world intuitions about these animals? 

**1E12.** Use the following function to explore different combinations of priors (change the parameters `a` and `b`) and data (change heads and trials). Summarize your observations.

```python
def posterior_grid(grid=10, a=1, b=1, heads=6, trials=9):
    grid = np.linspace(0, 1, grid)
    prior = stats.beta(a, b).pdf(grid)
    likelihood = stats.binom.pmf(heads, trials, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    _, ax = plt.subplots(1, 3, sharex=True, figsize=(16, 4))
    ax[0].set_title(f"heads = {heads}\ntrials = {trials}")
    for i, (e, e_n) in enumerate(zip(
            [prior, likelihood, posterior],
            ["prior", "likelihood", "posterior"])):
        ax[i].set_yticks([])
        ax[i].plot(grid, e, "o-", label=e_n)
        ax[i].legend(fontsize=14)


interact(posterior_grid,
    grid=ipyw.IntSlider(min=2, max=100, step=1, value=15),
    a=ipyw.FloatSlider(min=1, max=7, step=1, value=1),
    b=ipyw.FloatSlider(min=1, max=7, step=1, value=1),
    heads=ipyw.IntSlider(min=0, max=20, step=1, value=6),
    trials=ipyw.IntSlider(min=0, max=20, step=1, value=9))
```

**1E13.** Between the prior, prior predictive, posterior, and posterior predictive distributions which distribution would help answer each of these questions. Some items may have multiple answers.

1. How do we think is the distribution of parameters values before   seeing any data? 

2.  What observed values do we think we could see before seeing any   data? 

3.  After estimating parameters using a model what do we predict we will   observe next? 

4.  What parameter values explain the observed data after conditioning   on that data? 

5.  Which can be used to calculate numerical summaries, such as the   mean, of the parameters? 

6.  Which can can be used to to visualize a Highest Density Interval? 

**1M14.** Equation {eq}`eq:posterior_dist` contains the marginal likelihood in the denominator, which is difficult to calculate.

In Equation {eq}`eq:proportional_bayes` we show that knowing the posterior up to a proportional constant is sufficient for inference.

Show why the marginal likelihood is not needed for the Metropolis-Hasting method to work. Hint: this is a pen and paper exercise, try by expanding Equation {eq}`acceptance_prob`.

**1M15.** In the following definition of a probabilistic model, identify the prior, the likelihood, and the posterior: 

```{math} 
\begin{split} 
Y \sim \mathcal{N}(\mu, \sigma)\\ 
\mu \sim \mathcal{N}(0, 1)\\ 
\sigma \sim \mathcal{HN}(1)\\ 
\end{split}
```

**1M16.** In the previous model, how many parameters will the posterior have? Compare your answer with that from the model in the coin-flipping problem in Equation {eq}`eq:beta_binomial`.

**1M17.** Suppose that we have two coins; when we toss the first coin, half of the time it lands tails and half of the time on heads. The other coin is a loaded coin that always lands on heads. If we choose one of the coins at random and observe a head, what is the probability that this coin is the loaded one? 

**1M18.** Modify Code Block [metropolis_hastings_sampler_rvs](metropolis_hastings_sampler_rvs) to generate random samples from a Poisson distribution with parameters of your choosing.

Then modify Code Blocks [metropolis_hastings_sampler](metropolis_hastings_sampler) and [metropolis_hastings](metropolis_hastings) to generate MCMC samples estimating your chosen parameters. Test how the number of samples, MCMC iterations, and initial starting point affect convergence to your true chosen parameter.

**1M19.** Assume we are building a model to estimate the mean and standard deviation of adult human heights in centimeters. Build a model that will make these estimation. Start with Code Block [beta_binom](beta_binom) and change the likelihood and priors as needed. After doing so then 

1.  Sample from the prior predictive. Generate a visualization and   numerical summary of the prior predictive distribution 

2.  Using the outputs from (a) to justify your choices of priors and   likelihoods 

**1M20.** From domain knowledge you have that a given parameter can not be negative, and has a mean that is roughly between 3 and 10 units, and a standard deviation of around 2. Determine two prior distribution that satisfy these constraints using Python. This may require trial and error by drawing samples and verifying these criteria have been met using both plots and numerical summaries.

**1M21.** A store is visited by $n$ customers on a given day.

The number of customers that make a purchase $Y$ is distributed as $\text{Bin}(n, \theta)$, where $\theta$ is the probability that a customer makes a purchase. Assume we know $\theta$ and the prior for $n$ is $\text{Pois}(4.5)$.

1. Use PyMC3 to compute the posterior distribution of $n$ for all   combinations of $Y \in {0, 5, 10}$ and $\theta \in {0.2, 0.5}$. Use   `az.plot_posterior` to plot the results in a single plot.

2. Summarize the effect of $Y$ and $\theta$ on the posterior 

**1H22.** Modify Code Block [metropolis_hastings_sampler_rvs](metropolis_hastings_sampler_rvs) to generate samples from a Normal Distribution, noting your choice of parameters for the mean and standard deviation. Then modify Code Blocks [metropolis_hastings_sampler](metropolis_hastings_sampler) and [metropolis_hastings](metropolis_hastings) to sample from a Normal model and see if you can recover your chosen parameters.

 **1H23.** Make a model that estimates the proportion of the number of sunny versus cloudy days in your area. Use the past 5 days of data from your personal observations. Think through the data collection process. How hard is it to remember the past 5 days. What if needed the past 30 days of data? Past year? Justify your choice of priors. Obtain a posterior distribution that estimates the proportion of sunny versus cloudy days. Generate predictions for the next 10 days of weather.

Communicate your answer using both numerical summaries and visualizations.

 **1H24.** You planted 12 seedlings and 3 germinate. Let us call $\theta$ the probability that a seedling germinates. Assuming $\text{Beta}(1, 1)$ prior distribution for $\theta$.

1. Use pen and paper to compute the posterior mean and standard   deviation. Verify your calculations using SciPy.

2. Use SciPy to compute the equal-tailed and highest density $94\%$   posterior intervals.

3. Use SciPy to compute the posterior predictive probability that at   least one seedling will germinate if you plant another 12 seedlings.

 After obtaining your results with SciPy repeat this exercise using PyMC3 and ArviZ 

[^1]: If you want to be more general you can even say that everything is   a probability distribution as a quantity you assume to know with   arbitrary precision that can be described by a Dirac delta function.

[^2]: Some authors call these quantities latent variables and reserve   the name parameter to identify fixed, but unknown, quantities.

[^3]: Alternatively you can think of this in terms of certainty or   information, depending if you are a glass half empty or glass half   full person.

[^4]: Sometimes the word *distribution* will be implicit, this commonly occurs when discussing these topics.

[^5]: Here we are using experiment in the broad sense of any procedure   to collect or generate data.

[^6]: <https://xkcd.com/2117/> 

[^7]: Technically we should talk about the expectation of a random   variable. See Section {ref}`expectations` for details.

[^8]: See detailed balance at Sections {ref}`markov_chains` and {ref}`sec_metropolis_hastings`.

[^9]: For a more extensive discussion about inference methods you should   read Section [{ref}`inference_methods` and references   therein.

[^10]: This is sometimes referred to as a kernel in other Universal   Inference Engines.

[^11]: You can use ArviZ `plot_trace` function to get a similar plot.  This is how we will do in the rest of the book.

[^12]: Notice that in principle the number of possible intervals   containing a given proportion of the total density is infinite.

[^13]: For more examples check   <https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions> 

[^14]: Except, the ones happening in your brain.

[^15]: For example, a regularized linear regression with a L2   regularization is the same as using a Gaussian prior on the   coefficient.

[^16]: For example, if we have samples from the posterior, then we can   plug those samples of $\theta$ into   $\kappa = \frac{\theta}{1-\theta}$.

[^17]: <https://en.wikipedia.org/wiki/Likelihood_principle> 

[^18]: See Section {ref}`entropy` for more details.

[^19]: Wikipedia has a longer list at   <https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution#Other_examples> 

[^20]: Even when the definition of such priors will require more context   than the one provided, we still think the example conveys a useful   intuition, that will be refined as we progress through this book.

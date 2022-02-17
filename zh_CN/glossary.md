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

(glossary)= 
# 词汇表

**自相关（ Autocorrelation ）**：自相关是信号与其自身滞后副本的相关性。从概念上讲，您可以将其视为观察结果之间的时间滞后的相似程度。大自相关是 MCMC 样本中的一个问题，因为它会减少有效样本量。

 Autocorrelation is the correlation of a signal with a lagged copy of itself. Conceptually, you can think of it as how similar observations are as a function of the time lag between them.Large autocorrelation is a concern in MCMC samples as it reduces the effective sample size.

**任意不确定性（ Aleatoric Uncertainty ）**：任意不确定性与存在一些影响测量或观察的量的概念有关，这些量本质上是不可知的或随机的。例如，即使我们能够准确地复制用弓射箭时的方向、高度和力量等条件。箭仍然不会击中同一点，因为还有其他我们无法控制的条件，例如大气波动或箭杆的振动，它们是随机的。

Aleatoric uncertainty is related to the notion that there are some quantities that affect a measurement or observation that are intrinsically unknowable or random. For example, even if we were able to exactly replicate condition such as direction, altitude and force when shooting an arrow with a bow. The arrow will still not hit the same point, because there are other conditions that we do not control like fluctuations of the atmosphere or vibrations of the arrow shaft, that are random.

**贝叶斯推断（ Bayesian Inference ）**： 贝叶斯推理是一种特殊形式的统计推理，它基于组合概率分布以获得其他概率分布。换句话说，是条件概率或概率密度的公式和计算，$p(\boldsymbol{\theta} \mid \boldsymbol{Y}) \propto p(\boldsymbol{Y} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})$ 。

Bayesian Inference is a particular form of statistical inference based on combining probability distributions in order to obtain other probability distributions. In other words is the formulation and computation of conditional probability or probability densities, $p(\boldsymbol{\theta} \mid \boldsymbol{Y}) \propto p(\boldsymbol{Y} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})$.

**贝叶斯工作流（ Bayesian workflow ）**： 为给定问题设计一个足够好的模型需要大量的统计和领域知识专业知识。这种设计通常通过称为贝叶斯工作流的迭代过程来执行。此过程包括模型构建 {cite:p}`Gelman2020` 的三个步骤：推理、模型检查/改进和模型比较。在这种情况下，模型比较的目的不一定限于选择*最佳*模型，更重要的是更好地理解模型。

Designing a good enough model for a given problem requires significant statistical and domain knowledge expertise. Such design is typically carried out through an iterative process called Bayesian workflow. This process includes the three steps of model building {cite:p}`Gelman2020`: inference, model checking/improvement, and model comparison. In this context the purpose of model comparison is not necessarily restricted to pick the *best* model, but more importantly to better understand the models.

**因果推断（ Causal inference ）**：那些在不测试干预的情况下，用于估计某些系统中的治疗（或干预）效果的过程和工具。也就是说，推断来自与观测数据而非实验数据。

The procedures and tools used to estimate the impact of a treatment (or intervention) in some system without testing the intervention. That is from observational data instead of experimental data.

**协方差矩阵与精度矩阵（ Covariance Matrix and Precision Matrix ）**： 协方差矩阵是一个方阵，包含随机变量集合的每对元素之间的协方差。协方差矩阵的对角线是随机变量的方差。精度矩阵是协方差矩阵的逆矩阵。

The covariance matrix is a square matrix that contains the covariance between each pair of elements of a collection of random variable. The diagonal of the covariance matrix is the variance of the random variable. The precision matrix is the matrix inverse of the covariance matrix.

**设计矩阵（ Design Matrix ）**： 在回归分析中，设计矩阵是解释变量值的矩阵。每行代表一个单独的对象，连续的列对应于该观察的变量及其特定值。它可以包含指示组成员身份的指示变量（一和零），也可以包含连续值。

In the context of regression analysis a design matrix is a matrix of values of the explanatory variables. Each row represents an individual object, with the successive columns corresponding to the variables and their specific values for that observation. It can contain indicator variables (ones and zeros) indicating group membership, or it can contain continuous values.

**决策树（ Decision tree ）**： 决策树是一个类似流程图的结构，其中每个内部节点代表一个属性的“测试”（例如抛硬币是正面还是反面），每个分支代表测试的结果，每个叶节点代表一个类标签（在计算所有属性后做出的决定）。从根到叶的路径代表分类规则。如果树用于回归，则叶节点处的值可以是连续的。
A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes). The paths from root to leaf represent classification rules. The values at the leaf nodes can be continuous if the tree is used for regression.

**dse**: 两个模型之间“elpd_loo”的组件差异的标准误差。此误差小于单个模型的标准误差（`az.compare` 中的`se`）。原因是通常某些观察结果对于所有模型都一样容易/难以预测，因此这会引入相关性。

The standard error of component-wise differences of `elpd_loo` between two models. This error is smaller than the standard error (`se` in `az.compare`) for individual models. The reason being that generally some observations are as easy/hard to predict for all models and thus this introduce correlations.

**d_loo**: 两种模型的 `elpd_loo` 差异。如果比较两个以上的模型，则相对于具有最高 `elpd_loo` 的模型计算差异）。

The difference in `elpd_loo` for two models. If more than two models are compared, the difference is computed relative to the model with highest `elpd_loo`).

**认知不确定性（ Epistimic Uncertainty ）**： 认知不确定性与某些观察者缺乏对系统状态的知识有关。它与我们在原则上但在实践中可能拥有的知识有关，与自然的内在不可知量无关（与偶然的不确定性相反）。例如，我们可能不确定一件物品的重量，因为我们手头没有秤，所以我们通过举起它来估计重量，或者我们可能有一个秤但精度限制在公斤。如果我们设计实验或执行忽略因素的计算，我们也可能存在认知不确定性。例如，为了估计我们必须开车去另一个城市需要多少时间，我们可能会忽略在收费站上花费的时间，或者我们可能会假设天气或道路状况良好等。换句话说，认知不确定性是关于无知和反对偶然性，不确定性，原则上我们可以通过获取更多信息来减少它。

Epistemic uncertainty is related to the lack of knowledge of the states of a system by some observer. It is related to the knowledge that we could have in principle but not in practice and not about the intrinsic unknowable quantity of nature (contrast with aleatory uncertainty). For example, we may be uncertain of the weight of an item because we do not have an scale at hand, so we estimate the weight by lifting it, or we may have one scale but with a precision limited to the kilogram. We could also have epistemic uncertainty if we design an experiment or perform a computation ignoring factors. For example, to estimate how much time we will have to drive to another city, we may omit the time spent at tolls, or we may assume excellent weather or road conditions etc. In other words, epistemic uncertainty is about ignorance and in opposition to aleatoric, uncertainty, we can in principle reduce it by obtaining more information.

**统计量（ Statistic ）**： 统计量（不是复数）或样本统计量是从样本中计算出的任何数量。样本统计的计算有多种原因，包括估计总体（或数据生成过程）参数、描述样本或评估假设。样本均值（也称为经验均值）是一个统计量，样本方差（或经验方差）是另一个例子。当统计量用于估计总体（或数据生成过程）参数时，该统计量称为估计量。因此，样本均值可以是一个估计量，而后验均值可以是另一个估计量。

A statistic (not plural) or sample statistic is any quantity computed from a sample. Sample statistics are computed for several reasons including estimating a population (or data generating process) parameter, describing a sample, or evaluating a hypothesis. The sample mean (also known as empirical mean) is a statistic, the sample variance (or empirical variance) is another example. When a statistic is used to estimate a population (or data generating process) parameter, the statistic is called an estimator. Thus, the sample mean can be an estimator and the posterior mean can be another estimator.

**（ ELPD ）**： 预期对数逐点预测密度（或离散模型的预期对数逐点预测概率）。这个数量通常通过交叉验证或使用诸如 WAIC (`elpd_waic`) 或 LOO (`elpd_loo`) 等方法来估计。由于概率密度可以小于或大于 1，因此对于连续变量，ELPD 可以是负数或正数，而对于离散变量，ELPD 可以是非负数。

Expected Log-pointwise Predictive Density (or expected log pointwise predictive probabilities for discrete model). This quantity is generally estimated by cross-validation or using methods such as WAIC (`elpd_waic`) or LOO (`elpd_loo`). As probability densities can be smaller or larger than 1, the ELPD can be negative or positive for continuous variables and non-negative for discrete variables.

**可交换性（ Exchangeability ）**： 如果随机变量的联合概率分布在序列中的位置改变时不改变，则随机变量序列是可交换的。可交换的随机变量不一定是 iid，但 iid 是可交换的。

A sequence of Random variables is exchangeable if their joint probability distribution does not change when the positions in the sequence is altered. Exchangeable random variables are not necessarily iid, but iid are exchangeable.

**贝叶斯模型的探索性分析（ Exploratory Analysis of Bayesian Models ）**： 执行成功的贝叶斯数据分析所需的任务集合，而不是推理本身。这包括：诊断，使用数值方法获得的推理结果的质量；模型批评，包括对模型假设和模型预测的评估；模型比较，包括模型选择或模型平均；为特定受众准备结果。

The collection of tasks necessary to perform a successful Bayesian data analysis that are not the inference itself. This includes. Diagnosing the quality of the inference results obtained using numerical methods. Model criticism, including evaluations of both model assumptions and model predictions.
Comparison of models, including model selection or model averaging.
Preparation of the results for a particular audience.

**汉密尔顿蒙特卡洛（ Hamiltonian Monte Carlo ）**：汉密尔顿蒙特卡罗 (HMC) 是一种马尔可夫链蒙特卡罗 (MCMC) 方法，它使用梯度有效地探索概率分布函数。在贝叶斯统计中，这最常用于从后验分布中获取样本。
HMC 方法是 Metropolis--Hastings 算法的实例，其中提出的新点是从汉密尔顿量计算的，这使得提出新状态的方法远离当前具有高接受概率的状态。系统的演化是使用时间可逆和保体积的数值积分器（最常见的是蛙式积分器）来模拟的。 HMC 方法的效率高度依赖于该方法的某些超参数。因此，贝叶斯统计中最有用的方法是 HMC 的自适应动态版本，它可以在预热或调整阶段自动调整这些超参数。

Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC) method that uses the gradient to efficiently explore a probability distribution function. In Bayesian statistics this is most commonly used to obtain samples from the posterior distribution.
HMC methods are instances of the Metropolis--Hastings algorithm, where the proposed new points are computed from a Hamiltonian, this allows the methods to proposed new states to be far from the current one with high acceptance probability. The evolution of the system is simulated using a time-reversible and volume-preserving numerical integrator (most commonly the leapfrog integrator). The efficiency of the HMC method is highly dependant on certain hyperparameters of the method. Thus, the most useful methods in Bayesian statistics are adaptive dynamics versions of HMC that can adjust those hyperparameters automatically during the warm-up or tuning phase.

**异方差性（ Heteroscedasticity ）**： 如果随机变量的随机变量不具有相同的方差，即如果它们不是同方差的，则随机变量序列是异方差的。这也称为方差异质性。

A sequence of random variables is heteroscedastic if its random variables do not have the same variance, i.e. if they are not homoscedastic. This is also known as heterogeneity of variance.

**同方差性（ Homoscedasticity ）**： 如果一个随机变量序列的所有随机变量具有相同的有限方差，则该序列是同方差的。这也称为方差同质性。互补的概念称为异方差。

A sequence of random variables is homoscedastic if all its random variables have the same finite variance. This is also known as homogeneity of variance. The complementary notion is called heteroscedasticity.

**独立同分布（ iid ）**： 独立同分布。如果每个随机变量与其他随机变量具有相同的概率分布并且都相互独立，则随机变量的集合是独立且同分布的。如果随机变量的集合是独立同分布的，它也是可交换的，但反过来不一定是正确的。

Independent and identically distributed. A collection of random variables is independent and identically distributed if each random variable has the same probability distribution as the others and all are mutually independent. If a collection of random variables is iid it is also exchangeable, but the converse is not necessarily true.

**个体条件期望（ Individual Conditional Expectation, ICE ）** 
ICE 显示结果变量和感兴趣的协变量之间的依赖关系。这是对每个样本分别进行的，每个样本一行。这与 PDP 的对比,其中表示协变量的平均效应。

An ICE shows the dependence between the response variable and a covariate of interest. This is done for each sample separately with one line per sample. This contrast to PDPs 
where the average effect of the covariate is represented.

**推断（ Inference ）**： 通俗地说，推理是根据证据和推理得出结论。在本书中，我们通常指的是贝叶斯推理，它具有更严格和更精确的定义。贝叶斯推理是将模型调整为可用数据并获得后验分布的过程。因此，为了基于证据和推理得出结论，我们需要执行更多的步骤，而不仅仅是贝叶斯推理。因此，在贝叶斯模型的探索性分析方面或更一般地在贝叶斯工作流程方面讨论贝叶斯分析的重要性。

Colloquially, inference is reaching a conclusion based on evidence and reasoning. In this book refer to inference we generally mean about Bayesian Inference, which has a more restricted and precise definition. Bayesian Inference is the process of conditioning models to the available data and obtaining posterior distributions. Thus, in order to reach a conclusion based on evidence and reasoning, we need to perform more steps that mere Bayesian inference. Hence the importance of discussing Bayesian analysis in terms of exploratory analysis of Bayesian models or more generally in term of Bayesian workflows.

**填充（ Imputation ）**： 通过选择的方法替换缺失的数据值。常用方法可能包括最常见的事件或基于其他（当前）观测数据的插值。

Replacing missing data values through a method of choice. Common methods may include most common occurrence or interpolation based on other (present) observed data.

**核密度估计（ Kernel Density Estimation, KDE ）**：核密度估计。一种从有限样本集中估计随机变量概率密度函数的非参数方法。我们经常使用术语 KDE 来谈论估计密度而不是方法。

A non-parametric method to estimate the probability density function of a random variable from a finite set of samples. We often use the term KDE to talk about the estimated density and not the method.

**LOO**: 在本文中为帕累托平滑重要性采样留一交叉验证法 (PSIS-LOO-CV) 的简写。但在留一法交叉验证文献中，`LOO` 缩写可能会被限制为单纯的留一交叉验证法。

Short for Pareto smoothed importance sampling leave one out cross-validation (PSIS-LOO-CV). In the literature "LOO" may be restricted to leave one out cross-validation.

**最大后验估计（ Maximum a Posteriori, MAP ）** ：未知量的估计量，等于后验分布的模式。 MAP 估计器需要对后验进行优化，这与需要积分的后验均值不同。如果先验是平坦的，或者在无限样本大小的限制下，MAP 估计量相当于最大似然估计量。

An estimator of an unknown quantity, that equals the mode of the posterior distribution. The MAP estimator requires optimization of the posterior, unlike the posterior mean which requires integration. If the priors are flat, or in the limit of infinite sample size, the MAP estimator is equivalent to the Maximum Likelihood estimator.

**赔率（ Odds ）** ：衡量特定结果的似然性。它们被计算为产生该结果的事件数与不产生该结果的事件数之比。赔率通常用于赌博。

A measure of the likelihood of a particular outcome. They are calculated as the ratio of the number of events that produce that outcome to the number that do not. Odds are commonly used in gambling.

**过拟合（ Overfitting ）**： 当产生的预测与用于拟合模型的数据集过于接近而无法拟合新数据集时，模型就会过度拟合。就参数数量而言，过拟合模型包含的参数比数据所能证明的要多。任意的过于复杂的模型不仅会拟合数据，还会拟合噪声，从而导致预测不佳。

A model overfits when produces predictions too closely to the dataset used for fitting the model failing to fit new datasets.In terms of the number of parameters an overfitted model contains more parameters than can be justified by the data. An arbitrary over-complex model will fit not only the data but also the noise, leading to poor predictions.

**部分依赖图（ Partial Dependence Plots, PDP ）** ： PDP 显示结果变量和一组感兴趣的协变量之间的依赖关系，这是通过边缘化所有其他协变量的值来完成的。直观地说，我们可以将部分相关性解释为结果变量的期望值作为感兴趣的协变量的函数。

A PDP shows the dependence between the response variable and a set of covariates of interest, this is done by marginalizing over the values of all other covariates. Intuitively, we can interpret the partial dependence as the expected value of the response variable as function of the covariates of interest.

**帕累托 k 估计（ Pareto k estimates ）** $\hat k$:  `LOO` 使用的帕累托平滑重要性采样 (PSIS) 诊断。 帕累托 `k` 诊断估计单个留一观察与完整分布的距离。如果遗漏观察值对后验的影响太大，那么重要性采样就无法给出可靠的估计。如果 $\hat \kappa < 0.5$，则 `elpd_loo` 的对应分量被高精度估计。如果 $0.5< \hat \kappa <0.7$ 则准确度较低，但在实践中仍然有用。如果 $\hat \kappa > 0.7$，则重要性采样无法为该观察提供有用的估计。 $\hat \kappa$ 值也可用于衡量观察的影响。具有高度影响力的观测值具有较高的 $\hat \kappa$ 值。非常高的 $\hat \kappa$ 值通常表明模型错误指定、异常值或数据处理中的错误。

A diagnostic for Pareto smoothed importance sampling (PSIS), which is used by LOO. The Pareto k diagnostic estimates how far an individual leave-one-out observation is from the full distribution. If leaving out an observation changes the posterior too much then importance sampling is not able to give reliable estimates. If $\hat \kappa < 0.5$, then the corresponding component of `elpd_loo` is estimated with high accuracy. If $0.5< \hat \kappa <0.7$ the accuracy is lower, but still useful in practice. If $\hat \kappa > 0.7$, then importance sampling is not able to provide a useful estimate for that observation. The $\hat \kappa$ values are also useful as a measure of influence of an observation. Highly influential observations have high $\hat \kappa$ values. Very high $\hat \kappa$ values often indicate model misspecification, outliers, or mistakes in the data processing.

**点估计（ Point estimate ）** ：单个值，通常但不一定在参数空间中，用作未知量的*最佳估计*的摘要。点估计可以与区间估计（如最高密度区间）进行对比，后者提供描述未知量的值的范围或区间。我们还可以将点估计与分布估计进行对比，例如后验分布或其边缘。

A single value, generally but not necessarily in parameter space, used as a summary of *best estimate* of an unknown quantity. A point estimate can be contrasted with an interval estimate like highest density intervals, which provides a range or interval of values describing the unknown quantity. We can also contrast a point estimate with distributional estimates, like the posterior distribution or its marginals.

**p_loo**: `elpd_loo` 之间的差：和非交叉验证的对数后验预测密度。它描述了预测未来数据比预测数据困难得多。在某些正则条件下渐近，`p_loo` 可以解释为参数的有效数量。在表现良好的情况下，`p_loo` 应该低于模型中的参数数量并且低于数据中的观察值数量。如果不是，这表明模型的预测能力非常弱，因此可能表明模型严重错误。参见高 Pareto k 诊断值。

The difference between `elpd_loo`: and the non-cross-validated log posterior predictive density. It describes how much more difficult it is to predict future data than the observed data.Asymptotically under certain regularity conditions, `p_loo` can be interpreted as the effective number of parameters. In well behaving cases `p_loo` should be lower than the number of parameters in the model and smaller than the number observations in the data. If not, this is an indication that the model has very weak predictive capability and may thus indicate a severe model misspecification. See high Pareto k diagnostic values.

**概率编程语言（ Probabilistic Programming Language ）**： 一种由原语组成的编程语法，允许定义贝叶斯模型并自动执行推理。通常，概率编程语言还包括生成先验或后验预测样本甚至分析推理结果的功能。

A programming syntax composed of primitives that allows one to define Bayesian models and perform inference automatically. Typically a Probabilistic Programming Language also includes functionality to generate prior or posterior predictive samples or even to analysis result from inference.

**先验预测分布（ Prior predictive distribution ）**： 根据模型（先验和可能性）的数据的预期分布。也就是说，模型在看到任何数据之前期望看到的数据。参见方程 [eq:prior_pred_dist](eq:prior_pred_dist) 。先验预测分布可用于先验启发，因为根据观测数据来考虑通常比根据模型参数来考虑更容易。

The expected distribution of the data according to the model (prior and likelihood). That is, the data the model is expecting to see before seeing any data. See Equation [eq:prior_pred_dist](eq:prior_pred_dist). The prior predictive distribution can be used for prior elicitation, as it is generally easier to think in terms of the observed data, than to think in terms of model parameters.

**后验预测分布（ Posterior predictive distribution ）**： 这是（未来）数据根据后验的分布，而后者又是模型（先验和可能性）和观察数据的结果。换句话说，这些是模型的预测。参见方程 [eq:post_pred_dist](eq:post_pred_dist)。除了生成预测之外，后验预测分布还可用于通过将模型与观测数据进行比较来评估模型的拟合度。

This is the distribution of (future) data according to the posterior, which in turn is a consequence of the model (prior and likelihood) and observed data. In other words, these are the model's predictions. See Equation [eq:post_pred_dist](eq:post_pred_dist). Besides generating predictions, the posterior predictive distribution can be used to asses the model fit, by comparing it with the observed data.

**残差（ Residuals ）**： 观察值与感兴趣数量的估计值之间的差异。如果一个模型假设方差是有限的并且对于所有残差都是相同的，我们说我们有同方差性。如果相反，方差可以改变，我们说我们有异方差。

The difference between an observed value and the estimated value of the quantity of interest. If a model assumes that the variance is finite and the same for all residuals, we say we have homoscedasticity. If instead the variance can change, we say we have heteroscedasticity.

**充分统计量（ Sufficient statistics ）**： 如果没有从同一样本计算的其他统计量提供有关该样本的任何附加信息，则对于模型参数的统计量就足够了。换句话说，该统计数据*足以*汇总您的样本而不会丢失信息。例如，给定来自具有期望值 $\mu$ 和已知有限方差的高斯分布的独立值样本，样本均值对于 $\mu$ 来说是足够的统计量。请注意，均值没有说明色散，因此仅就参数 $\mu$ 而言就足够了。众所周知，对于 iid 数据，具有足够统计量且维度等于 $\theta$ 维度的唯一分布是指数族的分布。
对于其他分布，充分统计量的维度随着样本量的增加而增加。

A statistic is sufficient with respect to a model parameter if no other statistic computed from the same sample provides any additional information about that sample. In other words, that statistic is *sufficient* to summarize your samples without losing information. For example, given a sample of independent values from a normal distribution with expected value $\mu$ and known finite variance the sample mean is sufficient statistics for $\mu$. Notice that the mean says nothing about the dispersion, thus it is only sufficient with respect to the parameter $\mu$. It is known that for iid data the only distributions with a sufficient statistic with dimension equal to the dimension of $\theta$ are the distributions from the exponential family.
For other distribution, the dimension of the sufficient statistic increases with the sample size.

**合成数据（ Synthetic data ）**： 也称为假数据，它是指从模型生成的数据，而不是从实验或观察中收集的数据。来自后验/先验预测分布的样本是合成数据的示例。

Also known as fake data it refers to data generated from a model instead of being gathered from experimentation or observation. Samples from the posterior/prior predictive distributions are examples of synthetic data.

**时间戳（ Timestamp ）**： 时间戳是用于识别特定事件何时发生的编码信息。通常时间戳以日期和时间的格式写入，必要时使用更精确的几分之一秒。

A timestamp is an encoded information to identify when a certain event happens. Usually a timestamp is written in the format of date and time of day, with more accurate fraction of a second when necessary.

**图灵完备（ Turing-complete ）**：在口语中，用于表示任何现实世界的通用计算机或计算机语言都可以近似地模拟任何其他现实世界的通用计算机或计算机语言的计算方面。

 In colloquial usage, is used to mean that any real-world general-purpose computer or computer language can approximately simulate the computational aspects of any other real-world general-purpose computer or computer language.

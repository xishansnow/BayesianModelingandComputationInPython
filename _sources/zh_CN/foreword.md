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

(foreword)= 
# 序言

<style>p{text-indent:2em;2}</style>

```{epigraph} 

贝叶斯建模为许多数据科学和决策问题提供了一种优雅的方法，但在实践中很难让它良好地运作。尽管有许多软件包可以轻松指定复杂的分层模型，例如 `Stan`、PYMC3、`TensorFlow Probability (TFP)` 和 `Pyro`，但用户仍然需要额外的工具来诊断其计算结果是否正确。他们非常需要 “在出现问题时该怎么做” 的一些实践建议。

本书重点介绍 `ArviZ` 软件库，该库使用户能够对贝叶斯模型进行探索性分析，例如对任何推断方法生成的后验样本做诊断，并可用于诊断贝叶斯推断中的各种故障模式。

本书还讨论了可用于消除许多常见问题的建模策略（ 例如中心化处理 ）。

本书中大多数示例都使用了 PYMC3 ，另外一些使用了 `TFP`；书中还包括一些对其他概率编程语言的简要比较。

本书的作者都是贝叶斯软件领域的专家，并且是 PYMC3、`ArviZ` 和 `TFP` 软件库的主要贡献者。他们在应用贝叶斯数据分析的实践中也拥有丰富经验，这反映在本书采用的各种实用方法中。

总体来说，本人认为本书是对现有文献的有益补充，有望进一步推动贝叶斯方法的应用。
 
-- Kevin P. Murphy

```


原本设计：
1. K TurboProd + V TurboProd
2. K TurboProd + V GroupedLinear
并实验测得第二种量化方案较优。
---
存在问题：
1. 之前的ppl是直接续写测定，不准确、全面
2. 缺少对grouped linear和turbo不同组合（共四种）的测试
---
进行了测试，结果显示...
```
=== Perplexity Summary ===
mode                   success ppl         mean_nll  tokens  ppl_ratio  delta_pct
---------------------------------------------------------------------------------
NoQuant                True    9.987067    2.301291  512     1.000000   +0.00%
K=Prod_V=Prod          True    12.121017   2.494941  512     1.213671   +21.37%
K=Prod_V=Grouped       True    10.417543   2.343491  512     1.043103   +4.31%
K=Grouped_V=Prod       True    10.201289   2.322514  512     1.021450   +2.14%
K=Grouped_V=Grouped    True    10.130833   2.315584  512     1.014395   +1.44%
```

---

分析原因：
（参考https://zhuanlan.zhihu.com/p/2027781641892807929，先分析对kvcache量化方案的要求，然后解释为什么每种方案会有各自的表现）

---

下一阶段研究计划：
1. 测试更大的模型和更长上下文的表现
2. ...（列出可能的）
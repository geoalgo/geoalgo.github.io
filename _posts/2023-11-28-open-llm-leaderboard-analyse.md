---
layout: post
title: Ranking LLM models by more than one metric
---
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

You may have seen `Yi-34B` model leading the [Open-llm leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for accuracy, is that all there is? In this blog, I discuss the ranking obtained when considering model-size in addition to accuracy. The code is available [here](https://github.com/geoalgo/analyse_llm_leaderboard) in case you want to play with it.

### Available LLMs benchmark

Currently, LLMs have taken the world by storm and are all over. One critical aspect is *evaluation*, how do we make sure that model A is better than model B?

As you may guess, there are currently many available benchmarks for LLMs. Some benchmarks focus on general abilities of LLMs such as [HELM](https://crfm.stanford.edu/helm/latest/#/leaderboard), on chat-bot LLMs with [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/) or on code LLMs with [Human-eval](https://github.com/openai/human-eval). For the first general category, Eleuther-AI compiled a large collection of benchmarks in [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Some of those benchmarks are evaluated in HuggingFace [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

In particular, Open LLM Leaderboard allows to schedule evaluation “for free” for researchers which is a pretty cool feature! Underneath, it uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) from Eleuther-AI. While it gives indication on how this is done (see `about` tab in the Open LLM leaderboard page), the exact scripts are not public [yet](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/367) although it may change at some point.

### A quick glance at Open LLM Leaderboard

At the time of this writing, the top model from Open LLM Leaderboard is `tigerbot-70b-chat-v2` a 70b parameters model follow by two 34B parameters models fine-tuned from `Yi-34B` outperforming even larger `llama-70B` models and the like.

Any pull-request made to Open LLM Leaderboard, automatically schedules an evaluation job on HuggingFace cluster. Models are evaluated on 8x A100 and model versions from HuggingFace hub are stored which makes sure that results are reproducible (only weights are public not the training code of models though).

Let us look at the data, how generous is this sponsoring exactly? 

```python
from utils import load_dataframe_results
df = load_dataframe_results(Path("results"))
# 101054
number_of_models = len(df)
# 346011
total_time_hour = df.total_evaluation_time_secondes.sum()
```

That is **346K** hour of compute on 8x A100 has been given to the community to evaluate models. This would cost approximately **2.7M$** using the cheapest options from [lambdalabs](https://lambdalabs.com/service/gpu-cloud), thank you HuggingFace for evaluating 100K models for the community!

Looking at the [leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), we already see the model sorted by average performance so let us look at something different and perhaps more interesting. Accurate LLMs are great, but we are all stunned by their cost so an important aspect are also latency and model-size (for memory requirements). But how do we go about ranking if we consider those objectives too?


## Considering other objectives

Let us consider average performance together with model-size. If we consider multiple objectives, there is no single minimum in general but a Pareto front of solutions, namely the solution that are not dominated by others [[wikipedia](https://en.wikipedia.org/wiki/Multi-objective_optimization)]. A solution is said to be dominated by another one if it is worse for all objectives. A solution not dominated by any other is such that no model evaluation yield better scores for all metrics.

If you are lost, let us see a graphical picture :-) 

Here is the scatter plot of all models in the Open-LLM-leaderboard (the plot is interactive, model names will pop up when you hover your mouse on the figure):


{% include open-llm-leaderboard-analyse/all.html %}

and next, the models that are on the Pareto front, you can think visually of models that are on the “bottom left”:

{% include open-llm-leaderboard-analyse/all-pareto.html %}


We see that `platypus-yi-34B` (fine-tuned from `Yi-34b`) outperforms many 70B+ models. This makes those models “useless” for model size and average error since `Yi-34B` variants outperforms them in both metrics! 

We also see many "vertical clusters" of models as many researchers try to fine-tune efficient models such as `Llama2-7B`, `Mistral-7B` or `Yi-34B` and obtain exactly the same model size as they are only fine-tuning the weights.


Interestingly, many of the models that are on the Pareto Front uses 4 or 8 bit precision as opposed to 16 bits (see the vertex color). This makes sense for two reasons. The first is this [nice paper](https://proceedings.mlr.press/v202/dettmers23a/dettmers23a.pdf) from Dettmers et al who showed that 4 bit precision is optimal for performance. In particular, the paper shows that given a fixed number of parameters it is better for accuracy to quantise a larger model with lower precision than a smaller model with higher precision (e.g. it is better for zero-shot accuracy to use `Llama2` 14B quantised in 4-bit than it is to use `Llama2` 7B quantised 8-bit). The second one is that researchers that use lower-precisions typically care about model size and are thus likely to produce efficient models for low size. 

If we report the configurations on the Pareto front, we do not get a single best solution but a **list** of configurations which are optimal. Let us take a look at this new leaderboard:


| model_name                                |   Average error |   model_size_GB |
|:------------------------------------------|----------------:|----------------:|
| tigerbot-70b-chat-v2                      |            0.22 |          129.43 |
| FashionGPT-70B-V1.2                       |            0.26 |          128.64 |
| Llama-2-70b-instruct                      |            0.28 |          128.56 |
| platypus-yi-34b                           |            0.28 |           64.17 |
| Airoboros-L2-70B-2.1-GPTQ                 |            0.30 |           38.03 |
| OpenHermes-2.5-neural-chat-7b-v3-1-7B     |            0.31 |           13.99 |
| Chupacabra-v3                             |            0.34 |           13.74 |
| openchat_3.5                              |            0.35 |            3.86 |
| llama-2-7b-chat-hf-phr_mental_health-2048 |            0.47 |            3.57 |
| llama7b-qlora                             |            0.48 |            3.54 |
| phi-1_5                                   |            0.50 |            2.64 |
| Guanaco-3B-Uncensored-v2-GPTQ             |            0.58 |            1.72 |
| pythia-2.8b-4bit-alpaca                   |            0.60 |            1.67 |
| Bloom_1b_Quantized                        |            0.65 |            1.35 |
| pythia-410m                               |            0.66 |            0.79 |
| GPTNeo350M-Instruct-SFT                   |            0.66 |            0.69 |
| Aira-2-355M                               |            0.67 |            0.68 |
| megatron-gpt2-345m-evol_instruct_v2       |            0.68 |            0.68 |
| gpt2-turkish-uncased                      |            0.68 |            0.24 |
| pythia-70m-deduped-cleansharegpt          |            0.69 |            0.14 |
| Flash-Llama-30M-20001                     |            0.69 |            0.08 |
| pythia-31m-goodwiki-deduped-2048-scratch  |            0.69 |            0.06 |


Congratulations to all model authors (many are hackers and not big labs),  you deserve the top position of the leaderboard as much as everyone else! 🥂🥳🎉


## Performance over time

Since the leaderboard provides access to all evaluations and metadata, we can also look at other interesting aspects. For instance, how much does the Pareto Front evolve through time? Let us have a look and plot the Pareto front of models trained over time for the last 5 months (the benchmark was introduced only at the end of July).

{% include open-llm-leaderboard-analyse/pareto-over-time.html %}

We see that the Pareto front has some gravity and the community has been steadily improving the performance on all budget! Great news for all, we need smaller and smarter models.

This is my first blogpost, feel free to reach out if you have feedback on how to improve it or suggestions for potential topics. Looking forward to hear what you say!

In a next post on the series, we will look at how to run those evaluation locally on your machine. Then we will see how to run on Hugging Face “free” cluster to save your bill! 🙃

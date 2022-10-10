# transformer-bayes

I put together this repository as part of a classroom presentation of [Transformers can do Bayesian inference](https://github.com/automl/TransformersCanDoBayesianInference).

# Bayesian Deep Learning

We want to approximate the distribution that underlies some dataset.

![image](https://user-images.githubusercontent.com/55119338/194781448-6c690620-f32d-49d9-8b18-38ea3a34ca47.png)

## PPD

## KL Divergence


# Overview of Paper
The paper introduces a prior-data-fitted-network, which works as follows:
![image](https://user-images.githubusercontent.com/55119338/194781533-d7447e4a-e7e4-4553-b53e-e8af58512641.png)


Using the above method, we can train a model to optimize the term on the left directly:
![image](https://user-images.githubusercontent.com/55119338/194918521-25685272-9dbd-41f3-8ca1-c7096f474436.png)

## Permutation Invariance

Transformers typically use positional embeddings to model sequence. In this case, we want to ignore the sequenceof the inputs, making them "permutation invariant." The authors accomplish this by modifying the transformer architecture:
![image](https://user-images.githubusercontent.com/55119338/194919717-e1dc0e02-0b1a-4fa9-b231-0549cedc6c84.png)


# Demo of Paper
The authors of the paper created a simple demonstration on  Huggiingface Spaces:
[Huggingface Spaces](https://huggingface.co/spaces/samuelinferences/transformers-can-do-bayesian-inference)

I ported some of their demonstration code into a Colab that you can run yourself:
[Colab link](https://colab.research.google.com/drive/1qn2hhzRfouo-F4iW7XnrB7948vOnS0tC#scrollTo=G1v6JK-j0Ium)

# Critique
The paper participated in an [open review process](https://openreview.net/forum?id=KSugKcbNf9).

# Question 1

# Question 2

# Extra Resources
1. [The official Github repository](https://github.com/automl/TransformersCanDoBayesianInference)
2. [A blog post](https://towardsdatascience.com/bayesian-inference-and-transformers-3dc473ac1af2) by Kaan Bıçakcı
3. [Video and Slidedeck of Paper Presentation](https://slideslive.com/38971570/transformers-can-do-bayesianinference-by-metalearning-on-priordata?ref=speaker-93687)
4. [3Blue1Brown on Bayes](https://www.youtube.com/watch?v=HZGCoVF3YvM)

# transformer-bayes

I put together this repository as part of a classroom presentation of [Transformers can do Bayesian inference](https://github.com/automl/TransformersCanDoBayesianInference).

# Bayesian Deep Learning

We want to approximate the distribution that underlies some dataset.

![image](https://user-images.githubusercontent.com/55119338/194781448-6c690620-f32d-49d9-8b18-38ea3a34ca47.png)

# Overview of Paper
The paper introduces a prior-data-fitted-network, which works as follows:
![image](https://user-images.githubusercontent.com/55119338/194781533-d7447e4a-e7e4-4553-b53e-e8af58512641.png)

Using the above method, we can train a model to optimize the term on the left directly:
![image](https://user-images.githubusercontent.com/55119338/194918521-25685272-9dbd-41f3-8ca1-c7096f474436.png)

This is called the PPD, posterior predictive distribution, and it is difficult to approximate directly using other methods such as Markov Chain Monte Carlo (MCMC).

## Permutation Invariance
Transformers typically use positional embeddings to model sequences. In this case, we want to ignore the sequence of the inputs, making them "permutation invariant." The authors accomplish this by modifying the transformer architecture:
![image](https://user-images.githubusercontent.com/55119338/194919717-e1dc0e02-0b1a-4fa9-b231-0549cedc6c84.png)

The input points attend to each other, and the queries attend to the input points.

In pseudocode:
Input: 𝒙 ∈ 𝑉∗, a sequence of token IDs.
Output: 𝑷 ∈ (0, 1)	<sup>𝑁<sub>V</sub>×length(𝒙)</sup>, where the 𝑡-th column of 𝑷 represents 𝑃ˆ𝜽(𝑥 [𝑡 + 1] | 𝒙[1 : 𝑡]).
Hyperparameters: *l*<sub>max</sub>, 𝐿, 𝐻, 𝑑<sub>e</sub>, 𝑑<sub>mlp</sub> ∈ ℕ
Parameters: 𝜽 includes all of the following parameters:
𝑾<sub>𝒆</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×𝑁<sub>V</sub></sup> , 𝑾<sub>𝒑</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×*l*<sub>max</sub></sup> , the token and positional embedding matrices.
For 𝑙 ∈ [𝐿]:
| W<sub>𝑙</sub>, multi-head attention parameters for layer 𝑙, see (4),
| 𝜸<sup>1</sup><sub>𝑙</sub>, 𝜷<sup>1</sup><sub>𝑙</sub>, 𝜸<sup>2</sup><sub>𝑙</sub>, 𝜷<sup>2</sup><sub>𝑙</sub>∈ ℝ<sup>𝑑<sub>e</sub></sup>, two sets of layer-norm parameters,
| 𝑾<sup>𝑙</sup><sub>mlp1</sub> ∈ ℝ<sup>𝑑<sub>mlp</sub>×𝑑<sub>e</sub></sup>, 𝒃<sup>𝑙</sup><sub>mlp1</sub> ∈ ℝ<sup>𝑑<sub>mlp</sub></sup>, 𝑾<sup>𝑙</sup><sub>mlp2</sub> ∈ ℝ<sup>𝑑<sub>e</sub>×𝑑<sub>mlp</sub></sup>, 𝒃<sup>𝑙</sup><sub>mlp2</sub> ∈ ℝ<sup>𝑑<sub>e</sub></sup>, MLP parameters.
𝜸, 𝜷 ∈ ℝ<sup>𝑑<sub>e</sub></sup>, final layer-norm parameters.
𝑾<sub>𝒖</sub> ∈ ℝ<sup>𝑁<sub>V</sub>×𝑑<sub>e</sub></sup>, the unembedding matrix.
1 *l* ← length(𝒙)
2 for 𝑡 ∈ [*l*] : 𝒆<sub>𝑡</sub> ← 𝑾<sub>𝒆</sub> [:, 𝑥 [𝑡]] + 𝑾<sub>𝒑</sub> [:, 𝑡]
3 𝑿 ← [𝒆<sub>1</sub>, 𝒆<sub>2</sub>, . . . 𝒆<sub>*l*</sub>]
4 for 𝑙 = 1, 2, . . . , 𝐿 do
5 | for 𝑡 ∈ [*l*] : 𝑿˜[:, 𝑡] ← layer_norm(𝑿[:, 𝑡] | 𝜸<sup>1</sup><sub>𝑙</sub>, 𝜷<sup>1</sup><sub>𝑙</sub>)
6 | 𝑿 ← 𝑿 + MHAttention(𝑿˜ |W<sub>𝑙</sub>, Mask[𝑡, 𝑡'] = [[𝑡 ≤ 𝑡']])
7 | for 𝑡 ∈ [*l*] : 𝑿˜[:, 𝑡] ← layer_norm(𝑿[:, 𝑡] | 𝜸<sup>2</sup><sub>𝑙</sub>, 𝜷<sup>2</sup><sub>𝑙</sub>)
8 | 𝑿 ← 𝑿 + 𝑾<sup>𝑙</sup><sub>mlp2</sub>GELU(𝑾<sup>𝑙</sup><sub>mlp1</sub>𝑿˜ + 𝒃<sup>𝑙</sup><sub>mlp1</sub>1<sup>T</sup>) + 𝒃<sup>𝑙</sup><sub>mlp2</sub>1<sup>T</sup>
9 end
10 for 𝑡 ∈ [*l*] : 𝑿[:, 𝑡] ← layer_norm(𝑿[:, 𝑡] | 𝜸, 𝜷)
11 return 𝑷 = softmax(𝑾<sub>𝒖</sub>𝑿)

## What does a prior Dataset look like?
The authors generate a prior dataset for a shared handwriting recognition task called Omniglot. The task looks like this:
![image](https://user-images.githubusercontent.com/55119338/194986884-59a8053b-d258-47d3-b03e-5a958652ff33.png)

The authors trained a model on synthetically generated data that looks like this:
![image](https://user-images.githubusercontent.com/55119338/194986437-b3d0f58e-1abe-43be-b845-5385dc781a18.png)


# Demo of Paper
The authors of the paper created a simple demonstration on Huggingface Spaces:
[Huggingface Spaces](https://huggingface.co/spaces/samuelinferences/transformers-can-do-bayesian-inference)

I ported some of their demonstration code into a Colab that you can run yourself:
[Colab link](https://colab.research.google.com/drive/1qn2hhzRfouo-F4iW7XnrB7948vOnS0tC#scrollTo=G1v6JK-j0Ium)

# Critique
The paper participated in an [open review process](https://openreview.net/forum?id=KSugKcbNf9).

Things that are still not clear to me that I think the paper could address more clearly:
 - It is not clear where this method would be most useful. What are some real world problems that might benefit from such an approach? They show that the method is more efficient than existing approaches for a diverse collection of tabular datasets (when run on a Tesla V100...), but they also reduce these datasets to binary classification problems and take other simplifying steps, so it is not clear to me how close we are to a 'market-ready' approach.
 - Speaking of the Tesla V100, it seems that the PFN approach should be compared to other GPU-accelerated approaches. I suspect there are other methods that could see massive speedups if they were also GPU-optimized.
 - The authors don't provide any intuitive explanation for why the method seems to work so well. In particular, I am confused about the transformer architecture. Why is the transformer architecture useful here? They do not use positional embeddings, and I do not understand how self-attention is contributing to their setup. Would this method work with a simpler NN architecture?
 - In terms of the writeup, it isn't clear where this contribution fits in to the literature or what issues should be addressed in future work.

# Question 1
 - Is it important to have accurate uncertainty estimates in your field? How important are confidence intervals in a predictive model?

# Question 2
 - A requirement to create a prior-data-fitted network (PFN) is to generate synthetic (labelled) data. Then, you perform Bayesian inference by learning on real data. Is this just weak supervision and transfer learning?

# Question 3
 - If the parameters of a transformer network can serve as a prior for Bayesian inference, could a sufficiently large language model serve as a universal prior?

# Extra Resources
1. [The official Github repository](https://github.com/automl/TransformersCanDoBayesianInference)
2. [A blog post](https://towardsdatascience.com/bayesian-inference-and-transformers-3dc473ac1af2) by Kaan Bıçakcı
3. [Video and Slidedeck of Paper Presentation](https://slideslive.com/38971570/transformers-can-do-bayesianinference-by-metalearning-on-priordata?ref=speaker-93687)
4. [3Blue1Brown on Bayes](https://www.youtube.com/watch?v=HZGCoVF3YvM)

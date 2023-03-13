---
title: Deep Black-Box Optimization with Influence Functions
author:
  - name: Jayanth Koushik
    affiliation:
    - 1
    email: jayanthkoushik@cmu.edu
    corresponding: true
  - name: Michael J. Tarr
    affiliation:
    - 1
    email: michaeltarr@cmu.edu
  - name: Aarti Singh
    affiliation:
    - 1
    email: aartisingh@cmu.edu
institute:
  - id: 1
    name: Carnegie Mellon University
abstract: Deep neural networks are increasingly being used to model black-box
  functions. Examples include modeling brain response to stimuli, material
  properties under given synthesis conditions, and digital art. In these
  applications, often the model is a surrogate and the goal is rather to optimize
  the black-box function to achieve the desired brain response, material property,
  or digital art characteristics. Moreover, resource constraints imply that,
  rather than training on a passive dataset, one should focus subsequent sampling
  on the most informative data points. In the Bayesian setting, this can be
  achieved by utilizing the ability of Bayesian models such as Gaussian processes
  to model uncertainty in observed data via posterior variance, which can guide
  subsequent sampling. However, uncertainty estimates for deep neural networks are
  largely lacking or are very expensive to compute. For example, bootstrap or
  cross-validation estimates require re-training the network several times which
  is often computationally prohibitive. In this work, we use influence functions
  to estimate the variance of neural network outputs, and design a black-box
  optimization algorithm similar to confidence bound-based Bayesian algorithms. We
  demonstrate the effectiveness of our method through experiments on synthetic and
  real-world optimization problems.
includes:
- commands.md
---

\newcommand{\II}{\mathcal{I}}

\acrodef{GP}{Gaussian process}
\acrodef{EI}{expected improvement}
\acrodef{TS}{Thompson sampling}
\acrodef{MPI}{maximum probability of improvement}
\acrodef{LCB}{lower confidence bound}
\acrodef{GP-LCB}{GP-LCB}
\acrodef{GP-INF}{GP-INF}
\acrodef{SVD}{singular value decomposition}
\acrodef{NN}{neural network}
\acrodef{INF}{INF}
\acrodef{NN-INF}{NN-INF}
\acrodef{MNIST}{MNIST}
\acrodef{CNN}{convolutional neural network}

# Introduction {#sec:introduction}

Black-box optimization, also known as zeroth order optimization, is the problem
of finding the global minima or maxima of a function given access to only
(possibly noisy) evaluations of the function. Perhaps the most popular black-box
optimization approach is in the Bayesian setting, such as \ac{GP} optimization,
which assumes that the black-box function is sampled from a \ac{GP}, and uses an
acquisition function such as \ac{LCB} to guide sampling and subsequently update
the posterior mean and variance of the \ac{GP} model in an iterative manner.
However, recently, deep neural networks are increasingly being used to model
black-box functions. Examples include modeling brain response to
stimuli[@yamins2014performance;@agrawal2014pixels;@kell2018task], material
properties under given synthesis conditions[@materialsAL], and digital
art[@manovich2015data]. Often the goal in these problems is optimization of the
black-box model, rather than learning the entire model. For example, a human
vision researcher might be interested in understanding which images cause
maximum activation in a specific brain
region[@ponce2019evolving;@bashivan2019neural]; a material scientist is
interested in finding optimal experimental conditions that yield a material with
desired properties[@materialsAL] or generate digital art with desired
characteristics[@manovich2015data]. While a simple approach is to learn a deep
model on passively acquired evaluations of the function, and then report its
optima, this is wasteful as often the black-box evaluations are expensive (c.f.,
subject time in a brain scanner is limited, material synthesis experiments are
expensive, etc.). Also, often pre-trained models of black-boxes need to be
updated subsequently to identify inputs that may lead to novel outputs not
explored in training set. For example, in material science a model trained to
predict the energy of a pure lattice may need to be updated to understand new
low-energy configurations achievable under defects, or deep neural net models of
images may need to be updated to achieve desired characteristics of synthetic
digital images. Thus, it is of interest to develop sequential optimization
methods akin to Bayesian optimization for deep neural networks.

Sequential optimization of neural network models requires an acquisition
function, similar to Bayesian optimization. However, popular acquisition
functions (\ac{LCB}, expectation maximization, Thompson sampling, etc.) are
mostly based on an uncertainty measure or confidence bound which characterizes
the variability of the predictions. Unfortunately, formal methods for
uncertainty quantification that are also computationally feasible for deep
neural network models are largely non-existent. For example, bootstrap or
cross-validation based estimates of uncertainty require re-training the network
several times, which is typically computationally prohibitive. In this paper, we
seek to investigate principled and computationally efficient estimators of
uncertainty measures (like variance of prediction), that can then be used to
guide subsequent sampling for optimization of black-box functions.

Specifically, we use the concept of influence
functions[@cook1980characterizations;@cook1982residuals] from classical
statistics to approximate the leave-one-out cross-validation estimate of the
prediction variance, _without having to re-train the model_. It is
known[@cook1982residuals] that if the loss function is twice-differentiable and
strictly convex, then the influence function has a closed-form approximation,
and the influence-function based estimate provides an asymptotic approximation
of the variance of prediction. Even though the loss function of neural networks
is non-differentiable and non-convex, it was recently
shown[@koh2017understanding] that in practice, the approximation continues to
hold for this case. However, @koh2017understanding used influence functions to
understand the importance of each input on the prediction of a passively trained
deep neural network, Influence functions were not investigated for uncertainty
quantification and estimation of prediction variance for use in subsequent
sampling.

A related line of work is activation maximization in \acp{NN} where the goal is
to find input that maximizes the output of a particular unit in the network.
However, since the corresponding target functions are known and differentiable,
gradient based optimization methods can be used. Still, obtaining results
suitable for visualization requires careful tuning and optimization
hacks[@nguyen2016synthesizing].  In this paper, we will consider the activation
maximization problem in a black-box setting, to mimic neuroscience and material
science experiments, where the brain is the black-box function. Furthermore,
prior work is passive requiring learning a good model for all inputs, while we
focus on  collecting new data to sequentially guide the model towards
identifying the input which leads to maximum output _without necessarily
learning a good model for all inputs_.

There have also been attempts to directly extend Bayesian optimization to neural
networks. @snoek2015scalable add a Bayesian linear layer to neural networks,
treating the network outputs as basis function. @springenberg2016bayesian focus
on scalability, and use a Monte Carlo approach, combined with scale adaptation.
However, our focus is to enable sequential optimization of existing NN models
being used in scientific domains. Our contributions can be summarized as
follows:

* We use influence functions to obtain a computationally efficient approximation
  of prediction variance in neural networks.
* We propose a computationally efficient method to compute influence functions
  for neural network predictions. Our approach uses a low-rank approximation of
  the Hessian, which is represented using an auxillary network, and trained
  along with the main network.
* We develop a deep black-box optimization method using these  influence
  function based uncertainty estimates that is valid in the non-Bayesian
  setting.
* We demonstrate the efficacy of our method on synthetic and real datasets. Our
  method can be comparable, and also outperform Bayesian optimization in
  settings where neural networks may be able to model the underlying function
  better than GPs.

The rest of the paper is organized as follows. In @sec:preliminaries, we
formally define the problem, and the Bayesian setting we build upon in this
work. Our proposed method is described in @sec:method, followed by results on
synthetic and real datasets in @sec:experiments. We conclude with discussion of
open problems in @sec:discussion.

# Preliminaries {#sec:preliminaries}

## Problem Setting {#sec:problem-setting}

We consider the problem of sequential optimization of a black-box function.
Specifically, let $f: \X \to \R$ be a cost function to be minimized. At each
step $t$, we select a point $x_t \in \X$, and observe a noisy evaluation $y_t =
f(x_t) + \e_t$, where $\e_t$ is independent 0-mean noise. This noisy evaluation
is the only way to interact with the function, and we don't assume any prior
knowledge of it. We will use $z$ to denote an input-output pair; $z \equiv (x,
y) \in \X \times \R$.

Practical functions in this category (like hyper-parameter optimization for
instance) generally tend to be "expensive", either in terms of time, or
resources, or both. This makes it impractical to do a dense "grid search" to
identify the minimum; algorithms must use as few evaluations as possible. With a
given time budget $T$, the objective is to minimize the simple regret,
$\min_{t=1 \dots T} {f(x_t) - f(x^\ast)}$ where $x^\ast \in \argmin_{x \in \X}
f(x)$ is a global minimum (not necessarily unique). This measures how close to
the optimum an algorithm gets in $T$ steps, and is equivalent to minimizing
$\min_{t=1 \dots T} f(x_t)$.

## Bayesian Optimization {#sec:bayesian-optimization}

Bayesian optimization is a popular method for solving black-box optimization
problems, which uses \ac{GP} models to estimate the unknown cost function. At
each step $T$, newly obtained data $(x_T, y_T)$ is used to update a \ac{GP}
prior, and the posterior distribution is used to define an acquisition function
$\alpha_T: \X \to \R$. The next point to query, $x_{T+1}$ is selected by
minimizing the acquisition function; $x_{T+1} = \argmin_{x \in \X} \alpha_T(x)$.
Popular acquisition functions are \ac{EI}, \ac{MPI}, and \ac{LCB}. Here, we will
particularly focus on \ac{LCB}, which provides the motivation for our method.

## GP-LCB {#sec:gp-lcb}

Consider a \ac{GP} model with mean function $0$, and covariance function
$k(\cdot, \cdot)$. After observing $T$ points, the model is updated to obtain a
posterior mean function $\mu_T$, and a posterior covariance function $k_T(\cdot,
\cdot)$. The \ac{LCB} acquisition function is $\alpha_T^{LCB}(x) = \mu_T(x) -
\beta_T^{1/2} \sigma_T(x)$, where $\sigma_T(x) \equiv k_T(x, x)$, and $\beta_T$
is a parameter for balancing exploration and exploitation. This expression is
easily interpretable; $\mu_T$ is an estimate of expected cost, and $\sigma_T$ is
an estimate of uncertainty. The next point is chosen taking both into
consideration. Points with low expected cost are exploited, and points with high
uncertainty are explored. $\alpha_T$ defines a "pessimistic" estimate (lower
confidence bound) for the cost, hence the name of the algorithm.

# Method {#sec:method}

Suppose we have a neural network $g: \X \to \R$ with parameters $\theta \in
\Theta$, trained using a convex loss function $L: (\X \times \R) \times \Theta
\to \R^+$. At a particular step $T$, we get an estimate of the parameters
$\shat{\theta}_T$, by minimizing $(1 / T)\sumnl_{t=1}^T L(z_t, \theta)$. As
noted earlier, $z_t \equiv (x_t, y_t)$, and we will use $g_\theta$ to denote the
network with a particular set of parameters. Now, for any point $x \in \X$, we
have a prediction for the cost function, i.e., $g_{\shat{\theta}_T}(x)$. So, if
we get an estimate $\shat{\sigma}_T(x)$, for the variance of
$g_{\shat{\theta}_T}(x)$, we can define an acquisition function $\alpha_T(x) =
g_{\shat{\theta}_T}(x) - \shat{\sigma}_T(x)$. Then optimization proceeds similar
to Bayesian optimization; where we select the next point for querying $x_{T +
1}$ by minimizing $\alpha_T(X)$. This auxiliary optimization problem is
non-convex, but local minima can be obtained through gradient based methods. In
practice, it is also common to use multiple restarts when solving this problem,
and select the best solution. We now describe our method for estimating the
variance using influence functions.

## Influence Functions {#sec:influence-functions}

Intuitively, influence functions measure the effect of a small perturbation at a
data point on the parameters of a model. We up-weight a particular point $z^+$
from the training set $\set{z_t}_{t=1}^T$, and obtain a new set of parameters
$\shat{\theta}_T^+(z^+, \nu)$ by minimizing the reweighted loss function, $(1 /
T)\sumnl_{t=1}^T L(z_t, \theta) + \nu L(z^+, \theta)$. We define the influence
of $z^+$ on $\shat{\theta}_T$ as the change $\shat{\theta}_T^+(z^+, \nu) -
\shat{\theta}_T$ caused by an adding an infinitesimal weight to $z^+$. Formally,
$$
  \II_{\shat{\theta}_T}(z^+)
= \lim_{\nu \to 0} \frac{\shat{\theta}_T^+(z^+, \nu) - \shat{\theta}_T}{\nu}
= \pder{\shat{\theta}_T^+(z^+, \nu)}{\nu}.
$$

Importantly, the influence function can be approximated using the following
result.
$$
\II_{\shat{\theta}_T}(z^+) \approx -H_{\shat{\theta}_T}^{-1} \nabla_\theta L(z^+, \shat{\theta}_T),
$$
where $\nabla_\theta L(z^+, \shat{\theta}_T)$ is the gradient of the loss with
respected to the parameters evaluated at $(z^+, \shat{\theta}_T)$, and
$H_{\shat{\theta}_T} \equiv (1 / T)\sumnl_{t=1}^T \nabla_\theta^2 L(z_t,
\shat{\theta}_T)$ is the Hessian. Now, we can use the chain rule to extend this
approximation to the influence on the prediction of $g_{\shat{\theta}_T}$. For a
test point $x^\dagger$, let $\II_{g_{\shat{\theta}_T}}(x^\dagger, z^+)$ be the
influence of $z^+$ on $g_{\shat{\theta}_T}(x^\dagger)$. So,

$$
\begin{aligned}
          \II_{g_{\shat{\theta}_T}}(x^\dagger, z^+)
      &=  \pder{g_{\shat{\theta}_T^+(z^+, \nu)}(x^\dagger)}{\nu} \\
      &=  \pder{g_{\shat{\theta}_T}(x^\dagger)}{\theta} \pder{\shat{\theta}_T^+(z^+, \nu)}{\nu} \\
      &=  \pder{g_{\shat{\theta}_T}(x^\dagger)}{\theta} \II_{\shat{\theta}_T}(z^+) \\
 &\approx -\pder{g_{\shat{\theta}_T}(x^\dagger)}{\theta} H_{\shat{\theta}_T}^{-1} \nabla_\theta L(z^+, \shat{\theta}_T).
\end{aligned}
$$

## Variance Estimation {#sec:variance-estimation}

Finally, we estimate the variance by computing the average squared influence
over the training points.  $$ \shat{\sigma}_T(x) = \frac{1}{T}\suml_{t=1}^T
\II_{g_{\shat{\theta}_T}}(x, z_t)^2. $$ In semi-parametric theory, influence is
formalized through the behavior of asymptotically linear estimators, and under
regularity conditions, it can be shown that  the average squared influence
converges to the asymptotic variance[@tsiatis2007semiparametric].

## Implementation {#sec:implementation}

The procedure described above cannot be directly applied to neural networks
since the Hessian is not positive-definite in the general case. We address this
issue by making a low-rank approximation, $H_{\shat{\theta}} \approx Q \equiv
PP^T$. Let $P = U \Sigma V^T$ be a \ac{SVD} of $P$. Then, $Q^\dagger \equiv U
\Sigma^{\dagger^2} U^T$ is the Moore-Penrose pseudoinverse of $Q$, where
$\Sigma^{\dagger^2}$ is a diagonal matrix with reciprocals of the squared
non-zero singular values. With this, for any vector $v$ we can approximate the
product with the inverse Hessian.
$$
H_{\shat{\theta}}^\dagger v \approx Q^\dagger v = U \Sigma^{\dagger^2} U^T v.
$$
We represent the low-rank approximation using a second neural network with a
single hidden layer. The network uses shared weights similar to an autoencoder,
and given input $v$, computes $PP^Tv$. We train this network to approximate
$H_{\shat{\theta}} \nabla_\theta L(z, \shat{\theta})$, using samples from the
training data. The Hessian vector product can be computed efficiently by
performing two backward passes through the network (Perlmutter's method). After
updating the network at each step $T$, the \ac{SVD} of $P$ is computed, which
allows efficient computation of $H_{\shat{\theta}_T}^{-1} \nabla_\theta L(z^+,
\shat{\theta}_T)$. The full algorithm (\acs{NN-INF}) is shown in @fig:algorithm.

![Algorithm](fig/alg){#fig:algorithm width=7in}

## GP-INF {#sec:gp-inf}

The influence approximation for variance can also be applied to \ac{GP} models,
by viewing them as performing kernel ridge regression. In this case, there is a
closed form expression for the influence[@ollerer2015influence], so we can
directly compute variance approximation. This gives a method similar to
\acs{GP-LCB}, where we use the influence approximation of variance instead of
the posterior variance. We term this method \acs{GP-INF}, and use it as an
additional baseline in our experiments.

# Experiments {#sec:experiments}

## Synthetic function maximization {#sec:synthetic-function-maximization}

First, we compare our method with \ac{GP} based algorithms using common test
functions used in optimization: five dimensional Ackley
function[@ackley2012connectionist], and ten dimensional Rastrigin
function[@rastrigin1974systems]. For the Ackley function, we use a network with
3 hidden layers, with 8, 8, and 4 hidden units respectively. And for the
Rastrigin function, we again use a network with 3 hidden layers, but with 16,
16, and 8 hidden units. In both cases, we approximate the Hessian with a rank 5
matrix. We report two sets of results, using different schemes for setting the
$\beta_t$ parameter (used in \acs{INF} and \ac{LCB} methods).

@fig:synth (1) shows the instantaneous regret over 500 iterations with $\beta_t
= c\sqrt{t}\log^{2}(10t)$ (based on the theoretical results presented by
@srinivas2009gaussian). We set $c=0.1$ for \ac{GP} methods, and $c=0.01$ for
\acs{NN-INF}. We did not find $c$ to have a significant effect on performance,
but for consistency, we used scaled $\beta_t$ for \acs{NN-INF} by 10 in all
cases. @fig:synth (2) shows the same results, but with $\beta_t$ held constant
throughout the experiment. We have $\beta_t = 2$ for \ac{GP} methods, and
$\beta_t = 0.2$ for \acs{NN-INF}.

![Optimization of synthetic functions](fig/synth){#fig:synth width=7in}

## Neural network output maximization {#sec:neural-network-output-maximization}

We now demonstrate results on a task inspired from neuroscience. To understand
the properties of a neuron, brain region etc., experimenters collect response
signals (neuron firing rate, increase in blood oxygenation etc.) to different
stimuli in order to identify maximally activating inputs. This is generally done
in an ad-hoc manner, whereby experimenters hand pick, or manually create a
restricted set of images designed to address a given theoretical question. This
can lead to biased results caused by insufficient exploration of the input
space. One way to address this issue is to perform adaptive stimulus selection
over the full input space.

To simulate the setting of stimulus selection in neuroscience, we first trained
a \ac{CNN} to classify images from the \acs{MNIST} dataset. The output layer of
this \ac{CNN} has 10 units, each corresponding to one of the \acs{MNIST} digits
(0 to 9). Given an input image, the output of each unit is proportional to the
probability (as predicted by the model), that the image belongs to the
particular class. With this, we can define an optimization task: find the image
that maximizes the output of a particular unit. This is similar to a
neuroscience visual stimulus selection experiment, where the output unit could
be a single neuron in the visual cortex.

Given the difficultly of this optimization problem, it is important to exploit
available prior knowledge. For a visual experiment, this could be in the form of
a pre-trained network. Here, we pre-train our \ac{CNN} model for binary
classification of two digits different from the target digit; for example
(classifying '5' vs. '6' when the target digit is '2'. For the model, we use a
smaller \ac{CNN} than the target; with two convolution layers, each with a
single filter. @fig:mnist shows the target neuron output for two different
settings. In @fig:mnist (a), the target digit is '2', and the \ac{CNN} is
pre-trained for classifying '5' vs. '6'. In @fig:mnist (b), the target digit is
'3', and the \ac{CNN} is pre-trained for classifying '1' vs. '8'. In both cases,
we see that the \ac{CNN} model is able to exploit the prior information, and
achieve better performance compared to the \acs{GP-LCB} baseline. This is a
promising result showing the feasibility of large scale adaptive sitmulus
selection.

![MNIST](fig/mnist){#fig:mnist width=7in}

# Discussion {#sec:discussion}

In this paper, we use the notion of influence functions from classical
statistics to estimate the variance of prediction made using a neural network
model, without having to retrain the model on multiple subsets of data as in
bootstrap or cross-validation based estimates. We additionally use these
uncertainty estimates to design a deep black-box optimization algorithm, that
can be used to optimize a black-box function such as brain response or desired
material property with sequentially collected data. We show the efficacy of our
algorithm on synthetic and real datasets.

There are several directions for future work. First, the uncertainty estimates
we propose are backed by theoretical underpinning under convexity assumptions
when the samples are assumed to be independent and it is of interest to develop
theoretical guarantees for the non-convex and sequentially dependent samples
setting which arises in optimization. The latter should be possible given
parallel analysis in the Bayesian setting. Such non-Bayesian confidence bounds
that are valid for sequential data can then also be used for active learning of
black-box functions or deep models. Second, while the method does not require
retraining the NN model at each iteration for variance estimation, the model
does require retraining as new data is collected. While this is inevitable in
optimization and active learning settings,the computational complexity can be
improved by not training the model to convergence at each iteration. For
example, in [@Awasthi:2017:PLE:3038256.3006384], and references therein,
computational efficiency is achieved for active learning of linear separators by
training the model to lower accuracy initially (e.g. it should be matched to the
lower statistical accuracy due to limited samples initially) and then increasing
the computational accuracy at subsequent iterations. Finally, we have only
explored the notion of uncertainty (coupled with prediction maximization) to
guide subsequent sampling. However, since neural networks learn a feature
representation, another way to guide sampling is via the notion of
expressiveness (c.f. [@sener2017active]) that selects data points which help
improve the learnt feature representation. It is interesting to compare and
potentially combine the notions of uncertainty and expressiveness to guide
sampling for optimization as well as active learning of black-box functions
modeled via deep neural networks.

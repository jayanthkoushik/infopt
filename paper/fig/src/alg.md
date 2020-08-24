\newcommand{\hidefrompandoc}[1]{#1}
\hidefrompandoc{
\algnewcommand{\HyperParameter}[2]{\State{\textbf{hyper-parameter} #1} \Comment{#2}}
\algnewcommand{\BlockComment}[1]{\Statex{$\blacktriangleright$ #1} \Statex{}}
}

\begin{algorithmic}[1]

\HyperParameter{$\{\beta_t\}_{t = 1 \dots \infty}$}{exploration-exploitation trade-off values}
\HyperParameter{$n_p$}{random samples used to pre-train the model}
\HyperParameter{$r$}{Hessian approximation rank}
\HyperParameter{$n_H$}{samples used for training Hessian approximation}
\HyperParameter{$n_I$}{samples used for computing influence}
\Statex

\Procedure{NNINF}{$f$, $\mathcal{X}$, $T$, $g_\theta$}
\BlockComment{Minimize $f$ over $\mathcal{X}$ for $T$ steps using the network $g_\theta$.}

  \State $D \gets \{(x, f(x)): x \in \Call{Sample}{\mathcal{X}, n_p}\}$ \Comment{samples for pre-training}
  \State $|\theta| \gets$ number of parameters in $\theta$
  \State $P \gets \Call{Matrix}{|\theta|, r}$ \Comment{for low rank Hessian approximation}
  \Statex

  \For{$t \gets 1 \dots T$}
    \State $\Call{TrainNetwork}{g_\theta, D}$
    \State $P, \mathcal{I} \gets \Call{IHVP}{g_\theta, D, P}$
    \State $x_t \gets \argmin_{x \in \mathcal{X}} \Call{Acquisition}{x, g_\theta, \mathcal{I}, \beta_t}$
    \State $D \gets D \cup \{(x_t, f(x_t))\}$
  \EndFor
  \Statex

  \State \textbf{return} $\argmin_{(x, y) \in D} y$
\EndProcedure
\Statex

\Procedure{IHVP}{$g_\theta$, $D$, $P$}
\BlockComment{Compute $H_\theta^{-1} \nabla_\theta L(z, \theta)$ for $z \in D$.}

  \State $\pi_P \gets \Call{FullyConnectedNetwork}{P, P^T}$
  \State $S_H \gets \Call{Sample}{D, n_H}$
  \State $L_H \gets \{\nabla_\theta L(z, \theta): z \in S_H\}$
  \State $J_\theta \gets (1 / n_H) \sum\limits_{z \in S_H} L(z, \theta)$
  \State $\nu_J \gets \nabla_\theta J_\theta$
  \State $D_H \gets \{(v, \nabla_\theta v^T \nu_J): v \in L_H\}$
  \Statex

  \State $\Call{TrainNetwork}{\pi_P, D_H}$
  \State $U, \Sigma, V \gets \Call{SVD}{P}$
  \State $W \gets U \Sigma^{\dagger^2}$
  \Statex

  \State $\mathcal{I} \gets \{W U^T v: v \in \Call{Sample}{L_H, n_I}\}$
  \State \textbf{return} $P$, $\mathcal{I}$
\EndProcedure
\Statex

\Procedure{Acquisition}{$x$, $g_\theta$, $\mathcal{I}$, $\beta$}
\BlockComment{compute the acquisition function at $x$}
  \State $\mu \gets g_\theta(x)$
  \State $\nu_\mu \gets \nabla_\theta \mu$
  \State $\sigma \gets \sqrt{\frac{1}{n_I} \sum\limits_{\iota \in \mathcal{I}} \left(\nu_\mu^T \iota\right)^2}$
  \State \textbf{return} $\mu - \beta^{1/2}\sigma$
\EndProcedure
\Statex

\end{algorithmic}

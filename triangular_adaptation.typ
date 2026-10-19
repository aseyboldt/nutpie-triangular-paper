#set document(title: "Triangular transport adaptation in nutpie")

#set page(
  paper: "a4",
  numbering: "1",
)
#set par(justify: true)
#set text(
  font: "Libertinus Serif",
  size: 11pt,
)

#set math.equation(numbering: "(1)")

#import "split_figure.typ": split-figure
#import "transformer_figure.typ": slice-figure, conditioner-diagram
#import "fill_figure.typ": fill-figure

#let inner(x, y) = $lr(angle.l #x, #y angle.r)$

#title("Triangular transport adaptation in nutpie")

In previous work @seyboldt2026preconditioninghamiltonianmontecarlo we showed how
minimizing a Fisher divergence can lead to linear preconditioners for MCMC
sampling. We generalize this to non-linear triangular maps.

Instead of sampling the posterior $p_cal(T)$ on the target space $cal(T) =
bb(R)^d$ directly, we pick a diffeomorphism $F: cal(A) -> cal(T)$ and run HMC on
the adapted space $cal(A) = RR^d$, where the density is
$
  p_cal(A)(x; F) = p_cal(T)(F(x)) abs(det J_F (x)).
$

Running HMC with the identity metric on $p_cal(A)$ is the same thing as running
HMC on $cal(T)$ with the metric pushed forward by $F$ (Appendix B of the paper),
so choosing $F$ is choosing a position-dependent mass matrix -- a flat
Riemannian metric.

If the adapted distribution $p_cal(A)$ is close to a standard normal $q =
cal(N)(0, I)$, it makes sense to modify the leapfrog step to take advantage of
this. Instead of splitting the Hamiltonian $H = K + V$, we split it into the
Hamiltonian $H = H_q + H_(q p)$, where $H_q$ is the Hamiltonian of the standard
normal distribution $q$, and $H_(q p)$ is a correction to recover $H$:
$
  H_q(x, rho) = 1/2 norm(x)^2 + 1/2 norm(rho)^2, quad
  H_(q p)(x, rho) = -log ((p_cal(A)(x)) / q(x)).
$

A leapfrog step with this split makes a half-step in the exact flow of $H_q$,
which is just a rotation, then a momentum kick along $nabla log (p_cal(A) / q)$, and then another half-step in the exact
flow of $H_q$. The closer the adapted distribution is to a standard normal, the
smaller the gradient step becomes, and the more accurate the integrator is. The
fisher divergence
$
  D_F (p_cal(A) \| q) = integral norm(partial_x log((p_cal(A)(x)) / q(x)))^2 p_cal(A)(x) dif x  = integral norm(partial_x log p_cal(A)(x) + x)^2 p_cal(A)(x) dif x
$
is exactly the expected squared norm of that gradient, so it is a natural
measure of how close the adapted distribution is to a standard normal. As the
distance approaches zero, the integration error vanishes and the leapfrog step
becomes exact, irrespective of the step size or the dimension of the space. At
that point a leapfrog with step size $pi / 2$ gives us an independent posterior draw.

#figure(
  split-figure,
  caption: [
    Four leapfrog steps ($epsilon = 0.9$) on a slightly non-normal 2d density
    $p_cal(A)$ (solid contours; dashed: $q = cal(N)(0, I)$), from the same start
    ($bullet$) and initial momentum. Coloured lines are the position updates,
    arrows the momentum kicks (drawn at half scale), and the dotted line the
    exact trajectory, ending at $times$. Left: the usual split, with straight
    drifts and kicks along the full score. Right: exact rotations under $q$ and
    kicks along $partial_x log (p_cal(A) \/ q)$ only, which are much smaller and
    stay close to the exact trajectory.
  ],
  placement: auto,
) <fig:split>

Our previous paper developed three different classes of affine preconditioners
around this idea: diagonal, dense and low-rank plus diagonal. Generalizations
that use nonlinear transformations instead seem straightforward at first: We
simply replace the linear transformation by a normalizing flow like for instance
a coupling flow. While this does work, we ran into practical issues quickly. As
the flow becomes more flexible, we need more and more warmup draws to fit it
reasonably. To make these ideas more scalable, we use model specific
information. The structural sparsity of the Hessian encodes which parameters are
conditionally independent, which constrains the possible dependencies in a flow
dramatically.

We will develop this framework first in the linear context and then generalize to
non-linear transformations.

= Triangular linear maps

Let's start by assuming that our posterior distribution is Gaussian, with mean
$mu$ and precision $Q$. Let's also assume that $Q$ is sparse, with a known
sparsity pattern (we will discuss later, how to get this from the model). We
want to find a linear map $F$ that transforms the posterior into a standard
normal distribution.

Any factorization $L^top L = Q$ gives a linear map $F(x) = L^(-1) x + mu$ that
transforms the posterior into a standard normal. By restricting $L$ to be lower
triangular with a positive diagonal, the choice becomes unique, and $L^top L =
Q$ is the (reversed) Cholesky factorization of $Q$.

To make use of the sparsity of $Q$ we use the well developed theory of sparse
Cholesky factorizations. We write $L = D (I - B)$ with $D$ positive diagonal and
$B$ strictly lower triangular. Then we can read off $F^(-1)(y) = D (I - B) (y -
mu)$ row by row. With $"Affine"(sigma, nu)(x) = sigma x + nu$,
$
  F^(-1)_i (y) = "Affine"(sigma_i, nu_i)^(-1)(y_i), quad
  sigma_i = d_i^(-1), quad
  nu_i = mu_i + sum_(j < i) b_(i j) (y_j - mu_j).
$

In a dense $Q$ we have $d (d-1) \/ 2$ regression coefficients, which is too many
for large models. But in most models the majority of them are exactly zero.
We call the parameters that do appear in row $i$ its _parents_,
$
  P(i) = {j < i : b_(i j) != 0}.
$

So when is $b_(i j)$ zero? Exactly when $y_i$ and $y_j$ are conditionally
independent given the other earlier parameters $y_(< i)$. But this is a
statement about the _marginal_ distribution of $y_1, dots, y_i$, whereas the
model directly tells us about conditional independence in the _joint_
distribution: $Q_(i j) = 0$ means $y_i perp y_j | y_"rest"$. Let's write $G$ for
the graph with an edge wherever $Q_(i j) != 0$.

To get from the joint to the marginals, we can integrate out the last parameter
$y_d$. For a Gaussian this replaces the precision by its Schur complement,
$
  Q' = Q_(< d, < d) - Q_(< d, d) Q_(d, < d) \/ Q_(d d),
$
and the correction term can only be nonzero for pairs of parameters that were
both neighbours of $y_d$. So integrating out a parameter connects all of its
neighbours with each other. If we keep going and integrate out $y_d, y_(d-1),
dots, y_(i+1)$, we get the graph of the marginal of $y_1, dots, y_i$, and the
edges from $i$ to earlier parameters in that graph are the parents:
$
  P(i) = {j < i : (i, j) in G union "fill"}.
$
The new edges are called _fill_, and this is exactly the symbolic phase of a
sparse Cholesky factorisation: $C$ is the Cholesky factor of $Q$ with the
parameters eliminated from last to first, and its extra nonzeros are the usual
fill-in.

= Choosing the order

How much fill we get depends a lot on the order of the parameters. A star is the
simplest example: a hub $h$ that is connected to leaves $ell_1, dots, ell_k$,
which are not connected to each other. If $h$ comes first, nothing is integrated
out before the leaves are done, there is no fill, and every leaf just has $h$ as
its parent. If $h$ comes last, it is integrated out first, which connects all
the leaves with each other. Then leaf $ell_j$ has all earlier leaves as parents,
and we have $k (k-1) \/ 2$ extra coefficients.

#figure(
  fill-figure,
  caption: [
    Sparsity pattern of the triangular factor for a star with $k = 6$ leaves. Row $i$ shows the
    parents of parameter $i$. Black: diagonal and edges of $G$, gray: fill. With the hub first
    there is no fill. With the hub last, integrating it out first connects all leaves, and every
    leaf gets all earlier leaves as parents.
  ],
  placement: auto,
) <fig:fill>

Parameters that couple many others should come early. Hierarchical scale
parameters are the typical example. We use standard fill-reducing orderings in
nutpie, METIS or AMD, applied to the reversed order. For instance, in centred
eight schools, each $theta_j$ is connected to $mu$ and $log tau$, and $mu$ to
$log tau$. With the order $log tau, mu, theta_1, dots, theta_8$ there is no fill
at all: $P(mu) = {log tau}$ and $P(theta_j) = {log tau, mu}$.

The order also decides how sequential the map is. The inverse $F^(-1)$ is always
fully parallelizable for each variable, but $F$ itself only variables whose
parents are already known can be computed together. The _level_ of a parameter
is the length of its longest chain of parents, and the number of levels is the
number of steps we can't parallelise. Eight schools has three.

= Nonlinear triangular maps

So far everything was Gaussian, and the map was linear. Triangular transport
maps @ramgraber2026friendlyintroductiontriangulartransport generalize this
construction to arbitrary (within reason) posteriors and nonlinear maps.

Instead of the sparsity pattern of the precision matrix, we now use the
structural sparsity of the Hessian. Two nodes in the graph $G$ are connected, if
the Hessian of $log p_cal(T)$ has a nonzero entry anywhere in parameter space.
This is exactly the Markov network of the model. The concept of fill also
transfers directly @3291125.3309628.

For any absolutely continuous posterior with a given variable order there is
exactly one transformation
$
  F^(-1)(y) = (S_1 (y_1, y_P(1)), dots, S_d (y_d, y_P(d)))
$
that transforms it into a standard normal distribution. (todo ref) $S_i$ is
monotone increasing in its first argument. We split $S_i$ into two parts: a
scalar monotone function $T$, which we call the _transformer_, and a conditioner
$C$ that computes the coefficients of $T$ from the parents:
$
  S_i (y_i, y_P(i)) = T(y_i; C (i, y_P(i))).
$

#figure(
  slice-figure,
  caption: [
    One row of a triangular map, with parent $y_1$ and child $y_2$. Each slice $y_1 = c$ of the
    target (left; ridges are the conditional densities $p(y_2 | y_1 = c)$) gets its own monotone
    transformer (middle, same colour), with coefficients $pi(c)$ chosen by the conditioner. Here the location
    is quadratic in $y_1$, the log-scale linear (a scale skip) and the skewness constant. Every
    slice is mapped to the same standard normal (right).
  ],
  placement: auto,
) <fig:slices>

In the linear case, the transformer is an inverse affine map, and the
conditioner is a linear function of the parents:
$
  T(y; m, ell) = "Affine"(e^ell, m)^(-1)(y) = e^(-ell) (y - m), quad
  C(i, y_P(i)) = (m_i, ell_i) = (mu_i + sum_(j in P(i)) b_(i j) (y_j - mu_j), -log d_i).
$

In the following, we assume several conditioners $C_k$ whose outputs add
together to give the transformer coefficients,
$pi_i = sum_k C_k (i, y_P(i))$.

#figure(
  conditioner-diagram,
  caption: [
    The conditioner computes the transformer coefficients $pi_i$ of row $i$ as a sum of
    constants, a location skip (only $m_i$), a scale skip (only $ell_i$) and an MLP (all
    coefficients). The transformer then maps $y_i$ to $x_i$.
  ],
  placement: auto,
) <fig:conditioner>

= Choosing the transformer and conditioner

We support four transformers: The simplest is to just stick with affine. With a
conditioner that is expressive enough, this can already capture a lot of
structure, but long or asymmetric tails are often a problem. We extend it with
the sinh-arcsinh family (`sas`, `asymmetric_sas`, `tangent_sas`), which adds
skewness and tail-weight coefficients. Each transformer keeps a location and
a scale parameter.

The conditioner computes those coefficients from the parents. We have three
different ways of doing that: a constant, a linear function of the parents, and
a small MLP. The linear function can be split into a location skip and a scale
skip, which are linear functions of the parents that add to the location and
log-scale coefficients.

All the pieces have their own ridge penalties, and the MLPs start with outputs
close to zero. The first fits of a new flow only use the sparse linear part, and
later fits add the rest, starting from the identity in those directions.

= Optimisation

Finding a stable way to optimize these maps was the main challenge of this
method.

During a warmup window we have a set of draws $y^((k))$ and corresponding scores
$beta^((k))$, and we want to fit the map $F_theta$ to minimize the estimate of
the Fisher divergence
$
  hat(D)_F(p_cal(A) \| q) =
  1 / N sum_k norm(alpha^((k)) + x^((k)))^2
  = 1 / N sum_k norm(r^((k)))^2,
  quad r^((k)) = alpha^((k)) + x^((k)),
$
where
$
  alpha^((k)) = partial_x log p_cal(A)(x^((k)))
  = J_F (x)^top (beta^((k)) - g^((k))),
  quad
  g = partial_y log abs(det J_(F^(-1))(y^((k))))
$


The parameters of the conditioners in $F$ can be optimized with first order
methods like Adam or L-BFGS, but they tend to often stall long before we reach a
good minimum. But we can make more use of the problem structure: On the adapted
space we have the metric of the standard normal, which we also use to define the
Hamiltonian. The pullback of that to the parameters of $F$ gives us a natural
metric on the parameter space. Since our objective is a non-linear least squares
problem, the Gauss–Newton approximation of the Hessian is exactly that metric.
This motivates the use of Levenberg–Marquardt to make use of second order
information. Even for moderate problem sizes, the full Gauss–Newton matrix is
too large to store, so we use conjugate gradients to solve the linear system in
each step. The Gauss-Newton matrix has a natural block structure, where all
parameters that belong to the conditioners of one posterior variable form a
block. The blocks are small enough to store, and we use them as a preconditioner
for the conjugate gradient solve.

= Implementation

The loss function itself already contains the jacobian of $F$, so that gradients
of the loss with respect to the parameters of $F$ use second order information.
As the loss function also makes heavy use of ragged arrays, it is not well-suited
to current autodiff frameworks like Jax.

We used a combination of hand-written jax code as reference and partially-LLM
generated code for the derivatives in Rust. This allowed us reasonably quick
iteration with verifiable results, but also decent run-time performance.

To speed up the optimization we also made use of several ideas. The cost and
parallelizability of $F$ depends heavily on the direction of evaluation:
$F^(-1)$ (i.e., the map from the target space to the adapted space) can be
evaluated fully in parallel. $F$ is at least partially sequential in nature. The
evaluation of the residuals involves both directions: We need to map the
posterior draws from the target space to the adapted space, and apply the
transposed jacobian of $F$ to the scores. To avoid the full cost of the
evaluation of $F$, we compute the sparse jacobian $J_(F^(-1))$ during the pass
through $F^(-1)$. Since $J_F = J_(F^(-1))^(-1)$, the backward pass then only
needs to solve a sparse triangular linear system, which is much faster than a
full $F$ evaluation.

We then need vector jacobian products and jacobian vector products of this
residual evaluation. In addition, we need the diagonal blocks of the full
Gauss-Newton matrix.

todo: mention Takahashi's equation etc

= The sparsity structure

todo Explain how we get the sparsity. structural program analysis for jax,
hessian vector autodiff probes with bloom filter for stan.

== Typical sparsity patterns

todo: what if the posterior is not sparse? We are allowed to lie about
the sparsity. This is a common method in spatial models, but here it doesn't
affect the posterior draws, only the efficiency of the sampler.

We need model specific information about the model, for instance
the spatial distance between points that correspond to parameters.

= Evaluation

todo examples:

centered eight school is perfectly fine, and gives better ess/grad than
noncentered eight school. Optimization takes in total ~4s for 4 chains.

2000 dim stochastic volatility model samples with very good ess/grad in ~4min.

Dense posteriors with >200 dim run out of memory quickly, and get quite
inefficient.

#bibliography("references.bib")

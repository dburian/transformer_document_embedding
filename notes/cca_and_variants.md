[cca_paper]: https://link.springer.com/chapter/10.1007/978-1-4612-4380-9_14
[dcca_paper]: https://proceedings.mlr.press/v28/andrew13.html
[gdcca_paper]: https://arxiv.org/pdf/1702.02519.pdf
[soft_cca_paper]: https://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Scalable_and_Effective_CVPR_2018_paper.pdf
[stochastic_cca_paper]: https://arxiv.org/pdf/1510.02054.pdf

# Canonical Correlation Analysis (CCA)

[CCA][cca_paper]'s goal is to find uncorrelated (orthogonal) projections of two
views such that the projections are maximally correlated.

## Variants

There are many variants:

- Kernel CCA -- linear projections are replaced by kernels
- [Deep CCA][dcca_paper] -- linear projections are replaced by deep networks
- [Generalized CCA][gdcca_paper] -- CCA generalized to multiple views
- ...

Since CCA is popular choice for learning multiview representations it has been
adjusted to suit the use in neural networks. More specifically [to use CCA in
stochastic updates from mini-batches][stochastic_cca_paper] and to decompose
CCA into two losses -- [Soft CCA][soft_cca_paper].

## Soft CCA

Soft CCA decomposes the traditional target into two losses:
- L2 norm to push projections closer together
- Stochastic Decorrelation Loss (SDL) to punish correlation of different
  (off-diagonal) features in mini-batches

The motivation for this is the following form of CCA:

$$
argmin_{\theta_1, \theta_2} = ||P_\theta_1(X_1) - P_\theta_2(X_2)||_F^2
$$

such that $P_\theta_1(X_1)^TP_\theta_1(X_1) = I$ and $P_\theta_2(X_2)^TP_\theta_2(X_2) = I$.

Where $P_\theta(X)$ is a projection of a mini-batch $X \in
\mathbb{R}^{batch_size \times dim}$ parametrized by $\theta$

The total loss is then:

$
L = L_2(P_\theta_1(X_1) - P_\theta_2(X_2)) + \lambda (
    L_{SDL}(P_\theta_1(X_1)) +
    L_{SDL}(P_\theta_2(X_2))
)
$

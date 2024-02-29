# Max-marginals loss with embeddings

Max-marginals loss is very popular for learning embeddings. We can try similar
approach but applied to similarity to teacher embeddings. E.g. let's say we are
training a student model from teacher embeddings and we have a batch $x_1,
\ldots x_n$ of student embeddings and $y_1, \ldots, y_n$ of teacher embeddings.

There are pairs:
- positive: $(x_i, y_i)$
- negative: $(x_i, y_j)$, where $i \ne j$

If $\delta$ is a similarity function, the total loss would be:

$$
L(x_i, y_i) = \sum_{i} \Big[
    \delta(x_i, y_i) - \lambda \frac{1}{n} \sum_{j \ne i} \delta(x_i, y_j)
\Big]
$$

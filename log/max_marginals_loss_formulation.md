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


## Experiments

I've implemented this idea and experimented with it. Here are some info:

- with cosine the in-batch negatives are always around 1 (orthogonal) and
  although it seems it doesn't provide any additional signal during training, in
  evaluation it proves to be superior to just cosine
- with mse it simply doesn't work at all. The loss has hard time closing the
  distance to positive while increasing the distance to negative. Instead it
  moves the embedding so that positive and negative distances are equal and
  leaves it there. Looking at it in more detail I've found that while average
  MSE between two random sbert embeddings (all taken from Wikipedia dataset) are
  around 11, the average distance to predicted embedding in-batch is around
  0.22. Additionally the in-batch distances to positives are always very close
  to the distance to the negatives.

### Deep dive

I've done more experiments, mainly focusing on what is actually happening with
the mse.

Relevant files: `../notebooks/max_marginals_losses.nb.py`

#### What is actually happening

MSE & MM MSE results:
- the problem is not that negatives are 'hard' or whatever, the model can learn
  to differentiate between them and makes the task easier with training
- the problem seems to be that the model is favorizing increasing the distances
  between the positives and the negatives instead of minimizing the positives
  (which brings up the question if the same is happening for cos)
    - distances between positives and negatives were much higher after training
      with mm mse than with just mse, but also the distances to positives were
      much higher

COS & MM COS results:
- the same thing happens as with MSE

#### How to remedy the situation

Hypothesis:
> Problem with `mse` is that grows qaudratically. So the model lowers the loss
> much more by acting in the higher regions. So while the positive side in in the
> lower regions and increasing doesn't hurt a lot, the negative side is in the
> higher regions and increasing it brings the loss down by a lot. In other words a
> slight increase in positive side is worth it if we simultaneously bring the
> negative further and therefore lower the loss.

However replacing MSE with Huber or L1 loss didn't help. I've also tried
clipping the negatives to a maximum value and playing with weighting the
negatives and the positives. The results are:

It seems that
- the clipping does pretty much nothing
    - when used such that the negative is clipped to the positive it diverges
    - when it is clipped to constant it behaves the same as with no negatives
      until the negatives are smaller then the constant, when the net reaches a
      plateau -- only minuscule improvement to the metric itself, positive and
      negatives fight
- the lambda only decreases the effect of having negatives -- with really small
  lambda the net doesn't diverge but also doesn't converge as quickly as with
  only the positive.  With 0.1 and `max_marginals_mse` == `mse`.
- huber and l1 just don't converge as quickly as mse

It seems that having any kind of negative only hurts the net in the convergence.
I think the positive effect of `max_marginals_cos_dist` is that the net can
differentiate between the embeddings in a better way, though they are a bit
further from SBERT.

#### Contrastive

At the and I also implemented contrastive loss which just minimizes the
cross-entropy between real similarities (1 for the same document, 0s for others)
and the similarities computed by cosine. The results is that it behaves like a
max marginals with cosine.

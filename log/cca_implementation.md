[i/cca_lowers_with_dim_and_sample_count]: imgs/cca_lowers_with_dim_and_sample_count.png

# Logging CCA

Initially I computed it with sklearn to assure that the results will be valid.
However upon testing this turned out to be quite time-consuming as average time
per iteration came from 0.2s to 30s and more. Even though results of my
implementation were relatively close to results of sklearn's implementation, on
real data there was quite a big difference.

In my implementation both regularization constant and epsilon have a large
influence on the end result. There was big deliberation on what to do, because
my implementation seems to produce wrong results, while sklearn takes too long
to compute despite using smaller `max_iter` or larger `tol`. The solution was to
use `cca-zoo`, which provides quick implementation which is closer to the
iterative computation that sklearn does.

Now it has become evident that even with randomly generated input vectors the
CCA correlation is quite high. However, the larger the number of samples
considered, the smaller the correlation is.

![CCA diminishes with number of
samples][i/cca_lowers_with_dim_and_sample_count].

Ergo it might make sense to compute just the correlation (no extra projection).

### solution

Possible solutions:
- dedicated trainer
- callbacks
- clever metrics with update frequency


I tried computing sklearn every time the buffer was renewed. The times were:
- 128 dims, 640 window: ~ 1min
- 256 dims, 1280 window: ~ 3min
- 512 dims, 2560 window: ~ 8min
- 768 dims, 3840 window: ~ 13min

did not: 5
out of: 24

But the time is not the worst thing. The computations sometimes do not converge,
to be exact in 5 out of 24 cases (20.2%).

`max_iter` limits the maximum number of iterations. When maximum number of
iterations is reached, the result is computed with the not-yet converged values.
The convergent warning is raised when the algorithm diverges **within the
maximum number of iterations.**

Questions:
- [x] can epsilon be set such that the computed value is nearly identical across
  cca dimensions and window size? - No, changing either the cca dimension or
  window size leads to different ideal epsilon. As rule of thumb the smaller
  either window size or dimension the larger is ideal epsilon.
- [] For given dimension and window size, is one epsilon ideal for an arbitrary
  input?
- [x] Can we approximate the result by decreasing the window? - No, the result
  can differ even up to 50%.
- [x] Does decreasing `max_iter` lead to faster and relatively precise
  computation? -- It does not seem so. The time reduction is small.
- [x] How much can we fasten sklearn computation by increasing tol? -- No time
  saved

The following questions are difficult to answer:

- [x] Does cca with lower dimension correspond to cca with larger dimension? --
  From my testing, during training it does not seem so
- [x] Does cca for lower window sizes correspond to cca for larger window sizes?
  -- Again it does not seem so, more tests would be necessary.

Because:
1. To answer them, I would need data with varying CCA. Creating data
   artificially is doable, but it is questionable whether they will be
   representative of the real data.

To resolve this, the only solution I see is to compute CCA with sklearn when
training, assuming the model is going to change dramatically so that cca will
change as well.

### implementation

At the end I decided to use implementation from `cca-zoo`. Though its `CCA` is
quicker and produces much more stable results than mine implementation, it is a
weaker model than sklearns iterative algorithm. Meaning it scores less than
sklearn's model. But from my testing it seems the relative difference
corresponds.

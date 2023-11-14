# Windowed vs normal metrics

## Disadvantages of windowed metrics

For windowed metrics we assume that we do not call `reset()` after logging i.e.
`compute()`.

- its not evident how large window does particular metric have -- it is not
  logged anywhere and code can change
- train and validation metrics are not comparable -- for windows larger than
  validation size
- they do not capture the whole validation data -- for windows smaller than
  validation size some computation is worthless
- small selection of metrics, more custom code

## Disadvantages of non-windowed metrics

For non-windowed metrics we assume we call `reset()` after every log.

- CCA does not work -- it needs to have some minimum samples to produce a value


## Solution

I've decided that:

### CCA metrics won't ever reset

The only disadvantage is that for validation metrics it might happen that
new validation round considers some data from an old round. So it might be
advisable to generate some warning. *And do not validate too far apart with too
little data.* But this seems unlikely.

Let us try to determine from where we should warn the user. It should be from a
piece of code which knows:
1. what kind of metric it is working with
2. that the metric will be processing validation data
3. the size of validation data

So it is not the training algorithm (because of 1.). Not the metric itself
because (2. and 3.). The only suitable place is when instantiating the metric in
preparation for training because we know all three things. Yet this place is not
suitable as the situation is really custom and I am bound to forget when
creating new model. Nevertheless, I use CCA only in a single model, when I
implement others I might realize that there is some code duplication, which also
includes these warnings.

### CCA metrics will have fixed buffer size depending on the number of components

Ergo: they will produce nans before reaching the required buffer size. The
advantage of this step is to make the CCA metrics comparable among experiments
and among different splits. If the CCA will create a value it will have some
informational value.

The buffer size should be included in the metric name to make it evident.

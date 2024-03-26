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

### Windowed is clearly better

We need windows for CCA, correlations, and cross-correlations.

### Windowed metrics will output nans before filling up the window

Metrics with different windows are not comparable. Ergo its pointless to output
any values with the window buffer half empty or bigger than it should be.

### All metrics with the same name should be comparable

This means that we need to include the number of components (in the case of CCA)
and the window size in the name of the metric.

### CCA needs at least as many inputs as components

CCA windowed metric must have more inputs than components. Assertion is needed.

### Sliding windows instead of fixed ones

In order to use all validation data (if the validation split is larger than the
window) we use sliding windows. The metric computes the value every time the
buffer fills up and stores it in a `Mean` metric. After it deletes the oldest
$n$ inputs and waits for the buffer to fill up again. When logging the metric
logs the average and resets it automatically.

Resetting the metric would reset the average as well as clear the buffer. This
means that windowed metrics shouldn't reset as they govern the resetting
themselves. The important exception is between validation runs where we want to
make sure, we start with an empty buffer.

I chose to use sliding window instead of using larger window (perhaps as big as
validation split) because CCA can sometimes be NaN. In such cases we wouldn't
have any feedback even though we did an intensive calculation.

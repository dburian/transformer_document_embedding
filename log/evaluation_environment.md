# Evaluation environment

We will regularly add new *tasks* and new *models*. We want easy, reliable,
streamlined way how to test the model on a given task. Something lean, flexible,
yet powerful.

## Solution

Intuitively we want an interface for both *task* and *model* to allow mixing one
with another.

The emphasis is on simplicity. These should be *helpers* functions. The more
strict these functions will be the more complex they'll be. I'd rather focus on
the model's themselves.

Lot of this hangs on the fact that I do not yet know how the baselines will look
like. I should probably research that, before I'm sketching out an interface
that will work with them.

---

### Task

Task definition should contain everything related to the data. So:

- loading the dataset
- preparing the data
- understanding the result -- evaluation of the model (metrics)

Sketch of mandatory interface:

- `get_data(split)` - returns given split of the dataset as pairs.
- `metrics` - property, returning metrics implemented as `tf.metrics.Metric`

#### Evaluation in the task or not?

Doing evaluation in the task would make it so we can easily make sure the model
will never see testing data in training and the separation of concernes would be
quite clear. On the other hand `tf.keras.Model`'s interface is not really
friendly towards the idea that the model should be evaluated outside of
`tf.keras.Model` and I expect pytorch's interface to be similar. So in the end I
decided for the model to take care of the evaluation.

However, there is an advantage doing evaluation in one place. When we get to it
(comparing models) I should think of something.

### Model

Model covers everything from learning to prediction. The models I'll be testing
are expected to differ in implementation, so we will probably want some
interface we can rely on in our evaluation environment plus some wrappers for
tensorflow and pytorch models.

<!-- Sketch of mandatory interface: -->

<!-- - `train((x, y) | tf.data.Dataset)` - trains the model with the given data -->
<!-- - `predict((x,) | tf.data.Dataset)` - predicts labels for given features -->

<!-- We should also think about our embedding scenario. It seems it will be a common -->
<!-- problem the model predicts embeddings, which do not yet solve the task (i.e. -->
<!-- they are not the predicted labels). It seems like we could getaway with solving -->
<!-- the problem with base classes which should take care of the overhead. -->

### Evaluation environment

Once we are comparing models, it would be quite handy to have one function into
which we can plug in our model and get results.

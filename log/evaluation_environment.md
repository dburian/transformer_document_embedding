# Evaluation environment

We will regularly add new *tasks* and new *models*. We want easy, reliable,
streamlined way how to test given model on given task. Something lean, flexible,
yet powerful.

### Dataset and model problems

In order to design the interfaces for model and task we need to solve two
problems: dataset and model problem.

- **Dataset problem**: Each dataset will come from another source, meaning it
  will naturally be in different format. Models on the other hand need unified
  interface so we can use different tasks for the same model.
- **Model problem**: Each model will most probably have to be adapted for each
  task. From hyperparameters to different heads.

The model problem seems unsolvable. We can get around code duplication with
sub-classing. If we accept this, suddenly we solve the dataset problem as well.
Each task will have the natural format and the models tailored for the
particular task will accept the given data format.

Obviously there are more ways how to design the interfaces but this to me seems
intuitive. I should be on the lookout for sub-classing just for the purpose of
data transformation. Then I need to think of another solution. Perhaps creating
transformation pipelines.

### Result of evaluation

There is the question what should I remember from the given experiment.

- tensorboard logs: this is a must have for debugging. I do not know any other
  tool, which is that easy to use and helpful.
- dict of final scores for each metric: this is obvious. I need some numbers in
  my latex tables.

Maybe there will be other things I need (like graphs train size to evaluation
metrics).

## Pseudocode of basic experiment

```python
model = import(args.model_package).Model()
task = import(args.task_package).Task()

model.train(task.train)

model.save(args.model_path)

test_predictions = model.predict(task.test)
results = task.evaluate(test_predictions)

utils.save_results(results)
```

### The because-s

- importing models and tasks by package path

Can be done differently (e.g. w/ dictionary), but this requires the least amount
of extra code. Just define a model in separate file and assign it to `Model`
variable.

- evaluation in model vs evaluation in task

I decided against evaluation in model, mainly because I want each model to be
evaluated equally. This also minimizes the code which should evaluate
predictions, which would have to be in every model.

## Model interface

- `train(train_data)` -- trains the model given the training data
- `save(dir_path)` -- saves the model into given directory
- `load(dir_path)` -- loads the saved model from the given directory,
- `predict(inputs)` -- returns predictions for given inputs, format specific to
  given task

## Task interface

- `train` -- property returning training data
- `test` -- property returning testing inputs
- `evaluate(test_predictions)` -- evaluates test predictions, format of
  test_predictions specific to given task


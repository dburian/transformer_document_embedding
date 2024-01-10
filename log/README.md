# Logs

These are random pieces of information collected when writing the thesis.


## Structure

- results of experiments belong to the logs about the experiments, not to the
  logs about the model which was evaluated
- don't be shy about creating too many logs -- few lines may deserve a special
  file

## Workflow

1. Create a TODO/Problem/Tmp. note
2. Write everything that may be important
3. When finished, copy the important bits to a separate log
4. Commit, push, delete the TODO/Problem/Tmp. note

### Advice

- differentiate between temporary scribbles and logs that carry information that
  will be used in the future
    - temporary scribbles/TODOs are about tasks
    - logs contain information that should be preserved
- logs should be structured, temporary scribbles needn't to
- temporary scribbles should be deleted -- don't hoard them, otherwise there is
  no compulsion which would make you structure the information in them and write
  it down as a log
- don't log what is logged in code -- the same advice as when commenting code.
  There is no use saying that `add(x, y)` returns sum of two numbers. If longer
  explanation is necessary or if important ideas are behind some decisions
  create a log, otherwise just code it.

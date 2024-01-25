# Student depth and breadth experiments

Combining the best of [depth experiments](./student_depth_experiments.md) and
[breadth experiments](./student_breadth_experiments.md).

## Old stuff

Questions:
- Will static loss help to lower contextualization loss?
- Which combination of static and contextual loss yield the best results?
- How to achieve good

Answers:

- Static loss has no effect on contextualization loss -- no magic here.
- Large grad. accum. step (64, batch size 6) lead to overfitting of sdl1 loss

### Start fine-tuning

Down the road, planned questions...

Questions:
- What is the most efficient projection to optimize for given loss?
- Are the projections same for different losses?
- Does some combination of static and contextual loss perform better in terms of metrics than another?
- How does the distribution of lengths influence performances on contextual and
  on static loss?

Projection net:
- projection layers:
    - as minimal as possible -- small output
    - bottle neck -- medium output
    - tube -- medium output
    - beefy -- large output
- static loss type:
    - mse
    - cos

Losses:
- static loss type:
    - mse
    - cos
- contextual loss type:
    - mse
    - cos

Length distribution:
- sampling of dataset
    - original distribution
    - more towards shorter sentences
    - more towards longer sentences

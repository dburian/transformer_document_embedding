# Timeline

1. **Download the datasets** *1%: 6.10. 2022*
    - paper: description of datasets
2. **Properly define tasks, metrics** *4%: 20.10. 2022*
    - paper: description of tasks
3. **Pre-process data for the given tasks** *4%: 3.11. 2022*
4. **Look for related work -- define baseline models** *6%: 24.11. 2022*
    - paper: related work chapter
5. **Carry out experiments on baseline models** *5%: 11.12. 2022*
6. **My model:** *50%: 12.6. 2023*
    1. **Sketch out my solution (model)**
    2. **Carry out experiments**
    3. **Analyze results and adjust my model**
    - paper: describe solution, what worked, what did not (the roadmap to my
    final solution)

8. **Write evaluation** *4%*
9. **Write conclusion** *4%*
10. **Rewrite - related work, tasks, my model** *4%*
11. **Write introduction** *4%*
12. **Write abstract** *4%*

13. **Final edit - images, run through checker** *10%*
---

- 100% ~ 3 semesters
- 1 semester ~ 13 weeks (+ 5 weeks)
- 1% ~ 0.36 week (0.54 week) ~ 2.5 days (3.5 days)


## Download the datasets

The only problem may be RELISH.
- describe datasets

## Properly define tasks

- in classification the task is quite clear
- for [RELISH] look into metrics
- [MS MARCO] use qrels from TREC Deeplearning tasks

## Prepare datasets

Just to have `tf.data.Dataset` with (x, y) pairs.

## Look for related work

Here is the main research phase. After this you should have an idea how our
model will look like.

In paper describe:
  - SBERT,
  - Longformer,
  - BOW models,
  - any relevant and similar paper you find

## Experiments on baseline models

Try to download the code.

## My experiments

Try to keep the length of the iteration cycle short. Once one iteration takes
too long the work prolongs like crazy.


# Paper TOC

1. Introduction

- what we are aiming to do
- why is it useful

2. Related work

- what has been done
- approaches to our problem
- mention models we will use

3. Benchmarks
    1. Tasks

        - define tasks
        - why I've chosen this task?

    2. Models

        - describe models

4. My model

- explain the architecture and reasons behind it
- learning and pitfalls

5. Evaluation

- overview of the experiments and results
- why my model scored low or high?

6. Conclusion

- what is the result of my work



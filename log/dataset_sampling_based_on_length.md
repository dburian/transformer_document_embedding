[teacher_student_training]: teacher_student_training.md

# Dataset sampling based on length

Because the [teacher-student training][teacher_student_training] depends on the
distribution of input lengths the distribution itself becomes a hyperparameter
which we'd like to set.

Observations:
- re-sampling a dataset on the fly would mean that:
    - we would be unsure about the dataset size, which would vary wildly between
      different re-sampling parameters
    - we would create a new dataset every time we run an experiment
- re-sampling script (jupyter notebook) is a better solution

In addition there is a sampler that keeps the length distributions similar
between effective batches (grad. acc. steps * batch size). I think the code is
without errors but it could be better as towards the end of the training, mean
length rises. I think the problem is that for some buckets, probabilities are
so low that in practice they are never selected. This hoards their count, which
is, however, insignificant compared to other buckets. It only appears towards
the end, when the probabilities of all buckets drop. What we could do is:

- have at least one example of each bucket in an effective batch
- use the default sampling with one pass with length resampler done beforehand
    - this is what I have done and apparently it keeps the length distribution
      very steady (maybe not that steady as resampling online and being in the
      middle of the training)
- use larger buckets with `[512, 1024, 3076]` on should be fine

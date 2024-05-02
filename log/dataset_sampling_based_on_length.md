[teacher_student_training]: teacher_student_training.md

# Dataset sampling based on length

We can either resample to get a target distribution or resample the dataset to
get more even distribution throughout the dataset.

## Sampling for target distribution

Because the [teacher-student training][teacher_student_training] depends on the
distribution of input lengths the distribution itself becomes a hyperparameter
which we may like to set.

Observations:
- re-sampling a dataset on the fly would mean that:
    - we would be unsure about the dataset size, which would vary wildly between
      different re-sampling parameters
    - we would create a new dataset every time we run an experiment
- re-sampling script (jupyter notebook) is a better solution

## Sampling for more even distribution

After shuffeling the dataset the distribution is already quite even. Resampling
creates the evenness also in smaller batches (like 64). However, the current
resampling code has an issue: the effective batch gets topped according to the
distribution of the bucket sizes. Since the longest bucket has relatively small
percentage (the length distribution is exponential) it gets rarely picked and
therefore the ending batches (last 10k inputs) have large amount of long inputs.
We can mitigate this by slightly favoring buckets with small probabilities: we
floor all the picks based on probabilities and top-up the effective batch with
buckets with the smallest probability. This had opposite effect: at the end of
training, the mean length dropped (last 5k inputs).

This could or could not be an issue. For large datasets it certainly won't be an
issue. For smaller ones, it could. Other solutions:

- have at least one example of each bucket in an effective batch
- use the default sampling with one pass with length resampler done beforehand
    - this is what I have done and apparently it keeps the length distribution
      very steady (maybe not that steady as resampling online and being in the
      middle of the training)
- use larger buckets with `[512, 1024, 3076]` on should be fine
- top-up the effective batch with the bucket whose sample distribution is
  significantly less than the target distribution


#### Effect of buffer size

I compared the effects of resampling with 10k and 1k buffer. The differences
are:
    - very minor actually
    - with 1k there are a bit higher spikes in mean length
    - at the end of the training the distribution spikes more with 1k but later
      in the dataset than 10k

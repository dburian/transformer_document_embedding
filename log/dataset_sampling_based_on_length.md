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

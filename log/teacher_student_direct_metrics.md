# Metrics for direct assessment of teacher-student training

To asses how well teacher-student training is in fact doing, without running a
complete benchmark, I decided to use metrics. These should quantitatively
determine how well is the student model mimicking the teacher model.

## Contextual metrics

Typically in comparison to SBERT embeddings.

- MSE -- how exactly is the student mimicking the contextual embedding
- COS -- how close the student is to the teacher (Cosine is probably **the**
  criterion for search)

## Static metrics

Typically with DBOW embeddings.

- CCA @ {16, 32, 64, 128, 256, ...} components -- how correlated the student can
  be to the teacher

[gensim_tut]: https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html#sphx-glr-auto-examples-howtos-run-doc2vec-imdb-py
[mikolov_2013]: https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html

# IMDB Classification task experiment results

## Longformer

The smaller model consumed somewhere in the neighbourhood of 7600 MB of vmem.
The larger model ran out of memory (with bs=1 and gradient_checkpointing) almost
at the very beginning of training.

### Test set evaluations

Using the default CLS head, which is:

- FC, 768 dim
- dropout, 0.1
- tanh,
- dropout, 0.1
- FC

and the parameters:

```yaml
epochs: 10
label_smoothing: 0.1
warmup_steps: 0.1 * train_size
batch_size: 1
classifier_dropout: 0.1
```
**we achieved 0.9506 accuracy on test set** after 5-7 epochs (125k-175k steps
with grad_accumulation step set to 16).

Overfitting seemed to happen after 7 epochs, but the validation score was
fairly consistent throughout the training (above 0.9433).


### HP search

One search results (4 epochs):

- the differences of accuracies were really minor
- `relu` helps
- reducing hidden dimension of classification head is the only really bad
  decision
- smaller/larger dropout on the classifier caused peaks in achieved accuracy in
  2./3. epoch, but 0.1 was overall the best.

| label_smoothing  | classifier_dropout | classifier_activation | classifier_dim | val. accuracy | val. loss |
| ---------------- | ------------------ | --------------------- | -------------- | ------------- | --------- |
|                  |                    | relu                  |                | 0.95120       | 0.20479   |
| 0.1              | 0.1                | tanh                  | 768.0          | 0.95099       | 0.19709   |
|                  | 0.0                |                       |                | 0.94800       | 0.24426   |
|                  | 0.5                |                       |                | 0.94679       | 0.30623   |
|                  |                    | 250.0                 |                | 0.94459       | 0.25216   |
| .0               |                    |                       |                | 0.94440       | 0.27785   |
|                  |                    | 50.0                  |                | 0.94340       | 0.24950   |

## Paragraph Vector

Originally followed the setup in the paper, but results were bad (60% binary
accuracy). Then I found a tutorial that tries to recreate the setup as well with
much better results - [gensim tutorial][gensim_tut]. I decided to first finetune
dbow then focus on dm.

### DBOW

- frequent word subsampling (discussed in detail in [article following
  word2vec][mikolov_2013]) does hurt the performance -- even as small as 1e-6

    - dbow_kwargs.sample = 0

- decreasing vector size to 100 does not hurt the performance (performed
  slightly better than original 400)

    - dbow_kwargs.vector_size = 100

- using min_count = 2 (throwing away any words appearing only in a single
  document) saves memory and does not hurt performance

    - dbow_kwargs.min_count = 2

- from 20 to 30 epochs only a slight improvement (1% in accuracy) suggesting
  that we've reached the maximal potential of DBOW

    - dbow_kwargs.epochs = 30

- 10 epochs for classification head is plenty, 15 worsen the score a bit

    - cls_head_epochs = 10

- the model overfits on its own -- classification activation, number of hidden
  units, dropout and label smoothing needs to be balanced

    | cls_head_activation |cls_head_dropout |cls_head_label_smoothing |cls_head_hidden_units | test_binary_accuracy |
    | ------------------- | --------------- | ---------------------   | -------------------- | -------------------- |
    | 'relu'              |  0.5            | 0.15                    | 25                   | 0.87355              |
    | 'relu'              |  0.2            | 0.2                     | 10                   | 0.87339              |
    | 'relu'              |  0.2            | 0.15                    | 40                   | 0.87203              |
    | 'relu'              |  0.5            | 0.20                    | 25                   | 0.87071              |


- finally adding also the `test` split for DBOW training achieved 0.8959,
  which is even better than gensim's recreation of the experiment.

### DM

- only tested with the best arguments for dbow, for which it was reasonable to
  assume same values would be beneficial as well:

    - dm_kwargs.sample = 0
    - dm_kwargs.min_count = 2
    - dm_kwargs.negative = 5.0
    - cls_head_kwargs.epochs = 10
    - cls_head_kwargs.label_smoothing = 0.15
    - cls_head_kwargs.hidden_dim = 25
    - cls_head_kwargs.hidden_activation = 'relu'
    - cls_head_kwargs.hidden_dropout = 0.5

- benefits from much longer training, which may not be as long if we tried also
  100 vector size. Initially I decided larger embed. dim is beneficial, but
  looking at the results, I now realize it is not entirely convincing.

    | dm_kwargs.vector_size | dm_kwargs.epochs | test_binary_accuracy |
    | --------------------- | ---------------- | -------------------- |
    | 400                   | 60               | 0.66391              |
    | 100                   | 80               | 0.66163              |
    | 400                   | 600              | 0.85831              |
    | 400                   | 800              | 0.86507              |
    | 400                   | 1000             | 0.86187              |

- This was before I introduced test set to training:

    | dm_kwargs.vector_size | dm_kwargs.epochs | test_binary_accuracy |
    | --------------------- | ---------------- | -------------------- |
    | 400                   | 800              | 0.8606               |
    | 100                   | 800              | 0.85895              |

So clearly benefits of larger vector size are marginal. Disappointedly test
split did not do much.

## SBERT

The results of various configurations of the classification head were really
close.

| val accuracy | hidden dropout | hidden features | label smoothing | model |
| ------------ | -------------- | --------------- | --------------- | ----- |
| 0.9448       | 0.1            | 25              | 0.15            | all-mpnet-base-v2 |
| 0.9348       | 0.1            | 25              | 0.15            | all-distilroberta-v1 |
| 0.9344       | 0.5            | 50              | 0.15            | all-distilroberta-v1 |
| 0.9334       | 0.5            | 25              | 0.2             | all-distilroberta-v1 |
| 0.9324       | 0.5            | 25              | 0.15            | all-distilroberta-v1 |
| 0.932        | 0.5            | 0               | 0.15            | all-distilroberta-v1 |
| 0.93         | 0.5            | 25              | 0.1             | all-distilroberta-v1 |
| 0.929        | 0              | 25              | 0.15            | all-distilroberta-v1 |
| 0.9282       | 0.5            | 150             | 0.15            | all-distilroberta-v1 |
| 0.8763       | 0.1            | 25              | 0.15            | all-MiniLM-L12-v2 |

Observations:
- too many hidden features hurts the performance
- best results seem to have around 25 - 50 hidden features with .15 label
  smoothing and at least 0.1 dropout (though 0.5 does not hurt that much)
- after 10 epochs validation accuracy seem to have stabilized

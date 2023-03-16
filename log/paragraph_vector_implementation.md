[gensim_tut]: https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html#sphx-glr-auto-examples-howtos-run-doc2vec-imdb-py
[mikolov_2013]: https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html

# Paragraph Vector implementation

I chose to implement PV using the `gensim` package. It seemed like
polished-enough code which copied the initial implementation.

## Convergence

Out-of-the-box gensim's PV does not return loss. It seems abnormal and
there have been lots of discussions around this. But the status is that there is
a PR, no ones working on it for some time now. This means that there is no
built-in way of telling how many epochs should one do.

Initially I though I resolved the issue by watching the change in document
embeddings of randomly chosen documents. If change to their embeddings was only
marginal between epochs I concluded the training converged.

Later I realised gensim uses linear decay on alpha, which meant that even
though embeddings should change (i.e. they caused the loss to be higher), they
did not get to change because alpha was small and therefore dampened the
gradient. Normalizing the change by currently employed alpha did not help
either. The expectation is that the change should be linearly decreasing as the
training progresses. Instead the changes suddenly diminishes very quickly at the
end of the training, no matter if it trained for 5 or 30 epochs.

The above is to say I probably need another convergence metric.

## Hyperparameters

Although initially I though I would implement the model same as was described in
the paper, I've read
[here](https://groups.google.com/g/gensim/c/Ab4dcRaF9n8/m/XXl08mRiDgAJ) that it
is not advisable. Especially the following parameters:

- `dm_concat` - concatenation of context words rather then mean creates larger
  model that may not be worth it,
- `negative` - I remember that It was said that negative sampling is better
  choice then hierarchical. Though the original model used hierarchical softmax
  it may be beneficial in terms of speed to switch to negative sampling.
- `min_count` - maybe beneficial?

## Finetuning

### IMDB

Originally followed the setup in the paper, but results were bad (60% binary
accuracy). Then I found a tutorial that tries to recreate the setup as well with
much better results - [gensim tutorial][gensim_tut]. I decided to first finetune
dbow then focus on dm.

#### DBOW

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

#### DM

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
    | 100                   | 800              | 0.85895              |
    | 400                   | 800              | 0.8606               |

So clearly benefits of larger vector size are marginal. Disappointedly test
split did not do much.

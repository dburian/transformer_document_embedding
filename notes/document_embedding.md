[datasets]: datasets.md
[goal]: goal.md
# Document embedding

The central point of my thesis is developing *good* document embeddings.
Therefore let's define the characteristics of a *good* document embedding. These
should agree with the [thesis's goal][goal].

## Formal definition of embedding
#### Function from continuous piece of text to real vector of given dimension

- To reason about a document we need a *set of numbers* describing it.
- The numbers should have implicit meaning attached -- they need to be in a
  defined order. Therefore we need a *vector*.
- The numbers should describe the text to infinite precision -- we need *real
  numbers*.
- Our focus is on *continuous pieces of text* rather than unconnected sentences.

## Definition of good document embedding

### The 'how'

How the embedding is computed. We look into how the model should work and
compare it against other models.


#### The embedding should capture the whole document

- Every little piece of a document could be potentially define it. Because we do
  not know where or if such part of document exists, we should capture the whole
  document, not just it's part.
- This requires the model to be unlimited in the input size.
- Not: SBERT
- Better: Longformer
- Yes: PV

#### Embeddings should capture structure

- Every piece of information contained in the text should be viewed in *context
  of the whole/neighbouring text*, not in separation.
- This would allow the embedding to capture structure of the text (e.g. text
  divided into parts, each part has its task and follows the previous one)
- Not: PV
- Yes: SBERT, Longformer

### The 'what'

What information the embedding contains. We observe performance of the
embeddings on various benchmarks. The way of thinking about this is if the model
would capture this and that it would have good scores in this benchmark.

Benchmarks were picked from [dataset index][datasets].

#### Similarity

This is the category of tasks SBERT excels at. This is where we'll see if our
model is any good.

- Documents similar (in terms of words, structure, meaning) to each other should
  be the same.
- Benchmarks:
    - wine,
    - game,
    - (similarity Wikipedia triplets)
    - PAN plagiarism detection

#### Topic

- Texts have usually something to say. They talk about a single or a group of
  topics. The embeddings should capture the topic.
- Tested by benchmarks on relatedness:
    - topic classification: Long document dataset
    - sharing topic: AAN citation recommendation,
    - related topic: Wiki65K, (AAN104K)

#### The author

- Every text tells us something about its author. The embeddings should include
  this information.
- Tested by benchmarks:
    - author's sentiment: IMDb
    - author's polarity: Hyperpartisian News
    - all author's characteristics: Blog Authorship corpus

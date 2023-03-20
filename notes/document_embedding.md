[goal]: goal.md
# Document embedding

The central point of my thesis is developing *good* document embeddings.
Therefore let's define the characteristics of a *good* document embedding. These
should agree with the [thesis's goal][goal].

Focus more on **how** the embedding is computed rather than **what** is the
result. The 'how' we can test -- just look at the model and we know what it
cannot do, the 'what' is harder -- we have no ways of assuring this is exactly
what the model does (we have some benchmarks but that is not conclusive).

## Definition of good document embedding

#### Formal definition: Function from continuous piece of text to real vector of given dimension

- To reason about a document we need a *set of numbers* describing it.
- The numbers should have implicit meaning attached -- they need to be in a
  defined order. Therefore we need a *vector*.
- The numbers should describe the text to infinite precision -- we need *real
  numbers*.
- Our focus is on *continuous pieces of text* rather than unconnected sentences.

### The 'how'

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

#### The embeddings of related documents should be close

- The meaning of a document is hard to define (it is subjective). Therefore we
  focus on *relatedness* as a looser term, without further specifying what
  characteristics related documents share.
- Tested by benchmarks on relatedness:
    - similarity:  wine, game, PAN plagiarism detection, Wiki65, (similarity
      wikipedia triplets)
    - sharing topic: AAN citation recommendation

#### The embeddings should capture content

- Texts have usually something to say. They talk about a single of a group of
  topics. The embeddings should capture the topic.
- Tested by benchmarks: IMDb, Hyperpartisian News, Long document dataset

#### The embeddings should capture the author's characteristics

- Every text tells us something about its author. The embeddings should include
  this information and *distinguish texts written by different authors*.
- Tested by a benchmark: Blog Authorship corpus

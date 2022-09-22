# Diploma thesis

## Ideas

 - music classification - classification of short (few seconds) music clips
   based on longer clips (few minutes)
 - target audience of text - classification of texts (articles, statuses) based
   on intended target audiences
 - fact checking
 - how the author thinks about the subject - the way how the author structures
   his ideas, chooses words can tell us how he/she thinks about the subject


---

# Music classification

MIR - music information retrieval. Generate labels based on short music clips.
The focus is on emotion retrieval.

## Datasets

### MediaEval DEAM - Database for Emotional Analysis in Music, 2013

- valorence and arousal axis
- 1802 songs

### FMA: Free Music Archive, 2017

[Article](https://arxiv.org/pdf/1612.01840v3.pdf)

- 100 000 clips
- mean duration ~ 190s
- #tags only for 22%
- hierarchical genre tags

### The MTG-Jamendo dataset, 2019

[Article](https://repositori.upf.edu/bitstream/handle/10230/42015/bogdanov_ICML2019__Jamendo.pdf?sequence=1)

- 55 000 clips
- median duration 224s
- 95 genre tags
- 41 instrumental tags
- 59 mood tags


### MTAT
### MSD

### Music4All, 2020

[Article](https://sites.google.com/view/contact4music4all)

---

# Predicting target audiences from text

The goal is to classify texts based on intended target audience. Alternatively
based on author. The classes should give a clear idea of either the 


---

# Document embeddings

> Proposed title: Text analysis using embeddings

The idea is to embed documents in such a way that similar documents would have
similar vectors. And then show how an analysis of such text could be done using
the embeddings and perhaps a labelled corpora.

> If thought corrupts language, language can corrupt thought." George Orwell
> You shall know a word by a company it keeps.
> -> You shall know a person by the words he/she uses.


## Goal and desired properties

Some embeddings are linear: e.g. **word2vec** "king" - "men" = "queen". This is
a property, which I would like the embedding to have. Then similar documents
would not only be close to each other in the feature space, but also the
differences between them would be characterized by the direction between the two
vectors. This property then allows us to navigate the feature space, even though
it is unlabelled, purely by the relation of two document embeddings.

> End goal: determine someone's bubble based on the texts he/she likes. For
> example: web site, anonymous account, twitter posts, like/dislike -> profile


## Problems

There are several problems: (add them as they arise)

1. Two documents: "I am truly happy." "I am not happy." have very similar
   syntax, yet largely different meaning. Using purely word embeddings with
   simple architecture to arrive to document embeddings may prove to be
   troublesome in this instance since the vectors should be dissimilar (or
   nearly opposite) yet there is only one word and therefore only one embedding
   that can distinguish the two document embeddings.

2. How can I be sure the given embedding posses the truly the properties I
   desire? I can only test for them.

> I should learn why there is a linear relationship between **word2vec's** word
> embeddings. - There is no nonlinearity.

## Sources

- [Shay Palachy - Document Embedding Techniques](https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d)

Medium article summing up techniques from up to 2019.

- [Le & Mikolov - Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

Language model like Continuous Bag of Words (CBOW) but with extra paragraph
embedding acting as a memory holding the context of the current paragraph.

> Maybe adding not just the paragraph embedding, but also document embedding and
> the author embedding, would lead to even better results and also would give us
> the embedding of the author.


> Exactly what I first had in mind. Leaves me thinking what else is there to be
> done? Should I focus on comparisons between the documents? Maybe in
> combination with labelled data? Or should I focus on the generative part --
> training a generative model so that it creates texts like the given document?

The downside of this approach is that to gain an embedding vector of a new
paragraph, its embedding needs to be learned to convergence (i.e. it takes
time).

> Getting rid of this downside could be another aim.




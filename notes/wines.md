[paper]: https://arxiv.org/pdf/2106.01186.pdf

# Wikipedia wines dataset

Introduced by [Ginzburg et al. (SDR paper)][paper].

## Description

Articles from Wikipedia parsed and split to sections. Each article has a title,
each section has a title and parsed text. The articles are accompanied by
expertly assessed similarities. There are about 1660 articles, 89 source
articles each with ~10 similar articles.

Upon closer look, the articles are sometimes **related only through external
knowledge**. E.g. Moët & Chandon winery is 'similar' to Chardonnay. When
looking at the text of Moët & Chandon on Wikipedia term 'Chardonnay' appears
only once.

I also found that in more than half of the considered source-target pairs the
target text does not contain any words from the source's title. It is true also
the other way around. **We should really try TF-IDF on this one.**

[cda_paper]: https://aclanthology.org/2020.emnlp-main.407/
# Document-pair classification tasks

When I say document-pair classification tasks I mean those referenced [on Zhou's
github io page](https://xuhuizhou.github.io/Multilevel-Text-Alignment/) from
[Cross-document attention paper][cda_paper].

## Document's ids

There is a funny table (Table 1, page 5) in the paper hinting that documents are
not unique. In other words the same document can appear in several pairs. There
is no identification in the original dataset but using python's `hash` on
the document text yields the correct number of unique documents.

### Ids an Paragraph Vector

I decided to use the ids in the training of PV since training the same vector
several times from scratch is not very efficient. For PV to train on each
document exactly once per epoch I implemented a check inside
`PairedGensimCorpus`. This ensures that PV is not overfitted on documents which
appear more often.

The ids in validation and test sets are not used. So even if some
validation/test document appeared in training set, it's id would be inferred.

## Statistics

dataset | train pairs | unique train docs | val pairs | unique train docs
---     | ---         | ---               | ---       | ---
OC      | 240 000     | 458 219           | 30 000    | ?
S2ORC   | 152 000     | 233 243           | 19 000    | ?
AAN     | 106 592     |  11 895           | 13 324    | ?
PAN     |  17 968     |  26 254           |  2 908    | ?


# Results for WikipediaSimilarities

## Wines

| model      | MRR    | MPR    | HR@10  | HR@100 | notes |
| ---------- | ------ | ------ |------- | ------ | -- |
| BigBird    | .33085 | .19879 | .65169 | .94382 |       |
| Longformer | .27915 | .21015 | .60674 | .94382 | pooler_type= 'mean' |
| SBERT      | .33524 | .20369 | .65169 | .97753 | |
| PV_DBOW    | .35964 | .20086 | .66292 | .94382 | |
| TFIDF      | .36529 | .16773 | .71191 | .94382 | smartirs = 'lfn' |

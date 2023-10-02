
# Results for WikipediaSimilarities

Explanation of metrics is in the [description of the task](../notes/wikipedia_similarities.md).

## Wines

### With correct metrics

| model      | MRR        | MPR        | HR@10      | HR@100     | notes |
| ---------- | ---------- | ---------- | ---------- | ---------- | ----- |
| SBERT      |   .43672   | **.16117** |   .15607   | **.56489** | transformer_model='all-mpnet-base-v2' |
| SBERT      |   .41420   |   .20348   |   .13222   |   .49683   | transformer_model='all-distilroberta-v1' |
| TFIDF      |   .45667   |   .16755   | **.17940** |   .54329   | smartirs='lfn' |
| PV DBOW    | **.47576** |   .18726   |   .14905   |   .49109   | |
| Longformer |   .37290   |   .20994   |   .11382   |   .44063   | pooler_type='mean' |
| BigBird    |   .42922   |   .20287   |   .11940   |   .48361   | pooler_type='mean' |


## Games


| model      | MRR        | MPR        | HR@10      | HR@100     | notes |
| ---------- | ---------- | ---------- | ---------- | ---------- | ----- |
| SBERT      |   .51713   |   .18915   |   .16493   |   .39141   | transformer_model='all-mpnet-base-v2' |
| SBERT      |   .53365   |   .19913   |   .16221   |   .37689   | transformer_model='all-distilroberta-v1' |
| TFIDF      |   .51965   | **.13868** | **.19995** | **.53637** | smartirs='lfn' |
| PV DBOW    | **.62673** |   .15368   |   .19234   |   .48516   | |
| Longformer |   .48848   |   .19274   |   .13796   |   .31651   | pooler_type='mean' |
| BigBird    |   .49156   |   .19953   |   .13191   |   .32453   | pooler_type='mean' |

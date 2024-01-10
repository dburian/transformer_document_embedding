
# Results for WikipediaSimilarities

Explanation of metrics is in the [description of the task](../notes/wikipedia_similarities.md).

## Wines

### Baselines

| model      | MRR        | MPR        | HR@10      | HR@100     | notes |
| ---------- | ---------- | ---------- | ---------- | ---------- | ----- |
| SBERT      |   .43672   | **.16117** |   .15607   | **.56489** | transformer_model='all-mpnet-base-v2' |
| SBERT      |   .41420   |   .20348   |   .13222   |   .49683   | transformer_model='all-distilroberta-v1' |
| TFIDF      |   .45667   |   .16755   | **.17940** |   .54329   | smartirs='lfn' |
| PV DBOW    | **.47576** |   .18726   |   .14905   |   .49109   | |
| Longformer |   .37290   |   .20994   |   .11382   |   .44063   | pooler_type='mean' |
| BigBird    |   .42922   |   .20287   |   .11940   |   .48361   | pooler_type='mean' |

### My finetuning

model                                              |  MRR       |  MPR       |  HR@10     |  HR@100    | notes
---                                                |  ---       |  ---       |  ---       |  ---       | ---
Longformer                                         |   .37290   |   .20994   |   .11382   |   .44063   | pooler_type='mean'
Longformer, only depth, contrastive cos            | **.44409** | **.14452** | **.16291** | **.57506** | contrastive_test/m.k.c_l_k.c_l_t=contrastive_cos_dist
Longformer, only depth, good mse                   |   .16195   |   .36438   |   .04214   |   .20154   | transformer_student_only_contextual/m.k.c_l_k.c_l_t=mse-m.t_k.g_a_s=8
Longformer, only depth, bad mse                    |   .45988   |   .17211   |   .13713   |   .51907   | transformer_student_only_contextual/m.k.c_l_k.c_l_t=cos_dist-m.t_k.g_a_s=8
Longformer, only breadth, soft_cca, best train CCA |   .26631   |   .27620   |   .05836   |   .30217   | sdl_alpha/m.k.s_l_k.s_c_s_a=0.99
Longformer, only breadth, mse, better val CCA      |   .29023   |   .24866   |   .07761   |   .36818   | transformer_student_only_static/m.k.s_l_k.s_l_t=mse-m.t._k.g_a_s=4
Longformer, both, best cca                         |   .39307   |   .18775   |   .12829   |   .47163   | transformer_student/m.k.c_l_k.l=0.05
Longformer, both, best cos                         |   .42753   |   .17012   |   .12803   |   .51984   | transformer_student/m.k.c_l_k.l=0.2
Longformer, both, in between                       |   .43318   |   .18027   |   .12812   |   .48742   | transformer_student/m.k.c_l_k.l=0.1


## Games

### Baselines

| model      | MRR        | MPR        | HR@10      | HR@100     | notes |
| ---------- | ---------- | ---------- | ---------- | ---------- | ----- |
| SBERT      |   .51713   |   .18915   |   .16493   |   .39141   | transformer_model='all-mpnet-base-v2' |
| SBERT      |   .53365   |   .19913   |   .16221   |   .37689   | transformer_model='all-distilroberta-v1' |
| TFIDF      |   .51965   | **.13868** | **.19995** | **.53637** | smartirs='lfn' |
| PV DBOW    | **.62673** |   .15368   |   .19234   |   .48516   | |
| Longformer |   .48848   |   .19274   |   .13796   |   .31651   | pooler_type='mean' |
| BigBird    |   .49156   |   .19953   |   .13191   |   .32453   | pooler_type='mean' |

### My finetuning

model                                              |  MRR       |  MPR       |  HR@10     |  HR@100    | notes
---                                                |  ---       |  ---       |  ---       |  ---       | ---
Longformer                                         |   .48848   |   .19274   |   .13796   |   .31651   | pooler_type='mean'
Longformer, only depth, contrastive cos            | **.50645** | **.19088** |   .15385   | **.39792** | contrastive_test/m.k.c_l_k.c_l_t=contrastive_cos_dist
Longformer, only depth, good mse                   |   .19198   |   .28123   |   .03486   |   .11401   | transformer_student_only_contextual/m.k.c_l_k.c_l_t=mse-m.t_k.g_a_s=8
Longformer, only depth, bad mse                    |   .53289   |   .19642   | **.15953** |   .38270   | transformer_student_only_contextual/m.k.c_l_k.c_l_t=cos_dist-m.t_k.g_a_s=8
Longformer, only breadth, soft_cca, best train CCA |   .31566   |   .27722   |   .07482   |   .17926   | sdl_alpha/m.k.s_l_k.s_c_s_a=0.99
Longformer, only breadth, mse, better val CCA      |   .23304   |   .29027   |   .05182   |   .16896   | transformer_student_only_static/m.k.s_l_k.s_l_t=mse-m.t._k.g_a_s=4
Longformer, both, best cca                         |   .45968   |   .21735   |   .13600   |   .33941   | transformer_student/m.k.c_l_k.l=0.05
Longformer, both, best cos                         |   .46074   |   .19418   |   .13823   |   .36091   | transformer_student/m.k.c_l_k.l=0.2
Longformer, both, in between                       |   .47714   |   .19376   |   .13635   |   .34864   | transformer_student/m.k.c_l_k.l=0.1

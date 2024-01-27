
# Results for WikipediaSimilarities

Explanation of metrics is in the [description of the task](../notes/wikipedia_similarities.md).

## Baselines

### Wines


| model      | MRR        | MPR        | HR@10      | HR@100     | notes |
| ---------- | ---------- | ---------- | ---------- | ---------- | ----- |
| SBERT      |   .43672   | **.16117** |   .15607   | **.56489** | transformer_model='all-mpnet-base-v2' |
| SBERT      |   .41420   |   .20348   |   .13222   |   .49683   | transformer_model='all-distilroberta-v1' |
| TFIDF      |   .45667   |   .16755   | **.17940** |   .54329   | smartirs='lfn' |
| PV DBOW    | **.47576** |   .18726   |   .14905   |   .49109   | |
| Longformer |   .37290   |   .20994   |   .11382   |   .44063   | pooler_type='mean' |
| BigBird    |   .42922   |   .20287   |   .11940   |   .48361   | pooler_type='mean' |

### Games

| model      | MRR        | MPR        | HR@10      | HR@100     | notes |
| ---------- | ---------- | ---------- | ---------- | ---------- | ----- |
| SBERT      |   .51713   |   .18915   |   .16493   |   .39141   | transformer_model='all-mpnet-base-v2' |
| SBERT      |   .53365   |   .19913   |   .16221   |   .37689   | transformer_model='all-distilroberta-v1' |
| TFIDF      |   .51965   | **.13868** | **.19995** | **.53637** | smartirs='lfn' |
| PV DBOW    | **.62673** |   .15368   |   .19234   |   .48516   | |
| Longformer |   .48848   |   .19274   |   .13796   |   .31651   | pooler_type='mean' |
| BigBird    |   .49156   |   .19953   |   .13191   |   .32453   | pooler_type='mean' |


## First evaluation round

- caveat: the below results used wrong cos and mse metrics
- CCA is measured on the final projeciton layers

### Wines

model                                              |  MRR       |  MPR       |  HR@10     |  HR@100    | notes
---                                                |  ---       |  ---       |  ---       |  ---       | ---
Longformer                                         |   .37290   |   .20994   |   .11382   |   .44063   | pooler_type='mean'
Longformer, only depth, contrastive cos            |   .44409   | **.14452** | **.16291** | **.57506** | contrastive_test/m.k.c_l_k.c_l_t=contrastive_cos_dist
Longformer, only depth, good mse                   |   .16195   |   .36438   |   .04214   |   .20154   | transformer_student_only_contextual/m.k.c_l_k.c_l_t=mse-m.t_k.g_a_s=8
Longformer, only depth, bad mse                    | **.45988** |   .17211   |   .13713   |   .51907   | transformer_student_only_contextual/m.k.c_l_k.c_l_t=cos_dist-m.t_k.g_a_s=8
Longformer, only breadth, soft_cca, best train CCA |   .26631   |   .27620   |   .05836   |   .30217   | sdl_alpha/m.k.s_l_k.s_c_s_a=0.99
Longformer, only breadth, mse, better val CCA      |   .29023   |   .24866   |   .07761   |   .36818   | transformer_student_only_static/m.k.s_l_k.s_l_t=mse-m.t._k.g_a_s=4
Longformer, both, best cca                         |   .39307   |   .18775   |   .12829   |   .47163   | transformer_student/m.k.c_l_k.l=0.05
Longformer, both, best cos                         |   .42753   |   .17012   |   .12803   |   .51984   | transformer_student/m.k.c_l_k.l=0.2
Longformer, both, in between                       |   .43318   |   .18027   |   .12812   |   .48742   | transformer_student/m.k.c_l_k.l=0.1

### Games

model                                              |  MRR       |  MPR       |  HR@10     |  HR@100    | notes
---                                                |  ---       |  ---       |  ---       |  ---       | ---
Longformer                                         |   .48848   |   .19274   |   .13796   |   .31651   | pooler_type='mean'
Longformer, only depth, contrastive cos            |   .50645   | **.19088** |   .15385   | **.39792** | contrastive_test/m.k.c_l_k.c_l_t=contrastive_cos_dist
Longformer, only depth, good mse                   |   .19198   |   .28123   |   .03486   |   .11401   | transformer_student_only_contextual/m.k.c_l_k.c_l_t=mse-m.t_k.g_a_s=8
Longformer, only depth, bad mse                    | **.53289** |   .19642   | **.15953** |   .38270   | transformer_student_only_contextual/m.k.c_l_k.c_l_t=cos_dist-m.t_k.g_a_s=8
Longformer, only breadth, soft_cca, best train CCA |   .31566   |   .27722   |   .07482   |   .17926   | sdl_alpha/m.k.s_l_k.s_c_s_a=0.99
Longformer, only breadth, mse, better val CCA      |   .23304   |   .29027   |   .05182   |   .16896   | transformer_student_only_static/m.k.s_l_k.s_l_t=mse-m.t._k.g_a_s=4
Longformer, both, best cca                         |   .45968   |   .21735   |   .13600   |   .33941   | transformer_student/m.k.c_l_k.l=0.05
Longformer, both, best cos                         |   .46074   |   .19418   |   .13823   |   .36091   | transformer_student/m.k.c_l_k.l=0.2
Longformer, both, in between                       |   .47714   |   .19376   |   .13635   |   .34864   | transformer_student/m.k.c_l_k.l=0.1

## Second evaluation round

- correct SBERT cos & mse metrics
- CCA is measured on outputs

### Wines

model                                                 |   MRR      |   MPR      |   HR@10    |   HR@100   | notes
---                                                   |   ---      |   ---      |   ---      |   ---      | ---
Longformer                                            |   .37290   |   .20994   |   .11382   |   .44063   | pooler_type='mean'
_                                                     | _          | _          | _          | _          | _
Longformer, only depth, short, `contrastive_cos_dist` |   .43270   |   .15193   |   .14902   |   .58815   | depth_loss_short/m.k.d_l_k.l_t=contrastive_cos_dist
Longformer, only depth, short, best cos               |   .43462   |   .14920   |   .15859   |   .58512   | depth_loss_short/m.k.d_l_k.l_t=cos_dist
Longformer, only depth, short, best mse & worst cos   |   .44536   |   .17396   |   .12536   |   .52749   | depth_loss_short/m.k.d_l_k.l_t=mse
Longformer, only depth, short, worst mse              |   .39471   |   .16423   |   .14104   |   .55277   | depth_loss_short/m.k.d_l_k.l_t=contrastive_mse
Longformer, only depth, `contrastive_cos_dist`        | **.47337** | **.14185** |   .14924   | **.59756** | depth_loss/m.k.d_l_k.l_t=contrastive_cos_dist
Longformer, only depth, best cos                      |   .46829   |   .14257   | **.16303** |   .58729   | depth_loss/m.k.d_l_k.l_t=cos_dist
Longformer, only depth, worst mse                     |   .46252   |   .14472   |   .15642   |   .57418   | depth_loss/m.k.d_l_k.l_t=contrastive_mse
Longformer, only depth, best mse & worst cos          |   .45882   |   .16423   |   .13158   |   .54830   | depth_loss/m.k.d_l_k.l_t=mse
_                                                     | _          | _          | _          | _          | _
Longformer, only breadth, 100 DBOW, best cca          |   .42714   | **.20533** | **.13278** | **.48654** | soft_cca_projections_dbow_100/m.k.b_l_k.t_p=[f=256-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[f=128-a=None-n=None,f=768-a=None-n=None]
Longformer, only breadth, 100 DBOW, second best cca   |   .42942   |   .22727   |   .11737   |   .43047   | soft_cca_projections_dbow_100/m.k.b_l_k.t_p=[f=256-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[f=32-a=None-n=None,f=768-a=None-n=None], had almost the same CCA as best CCA
Longformer, only breadth, 100 DBOW, worst cca         |   .29150   |   .22446   |   .09671   |   .40812   | soft_cca_projections_dbow_100/m.k.b_l_k.t_p=[f=768-a=relu-n=None,f=1024-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[f=32-a=None-n=None,f=768-a=None-n=None]
Longformer, only breadth, 768 DBOW, best cca          |   .39630   |   .21241   |   .11947   |   .44719   | soft_cca_projections_dbow_768/m.k.b_l_k.t_p=[f=768-a=relu-n=None,f=1024-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[]
Longformer, only breadth, 768 DBOW, second best cca   |   .32548   |   .20697   |   .10413   |   .40429   | soft_cca_projections_dbow_768/m.k.b_l_k.t_p=[f=256-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[]
Longformer, only breadth, 768 DBOW, worst cca         | **.44600** |   .23143   |   .11935   |   .27825   | soft_cca_projections_dbow_768/m.k.b_l_k.t_p=[f=256-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[f=128-a=None-n=None,f=768-a=None-n=None]


### Games

model                                                 |   MRR      |   MPR      |   HR@10    |   HR@100   | notes
---                                                   |   ---      |   ---      |   ---      |   ---      | ---
Longformer                                            |   .48848   |   .19274   |   .13796   |   .31651   | pooler_type='mean'
_                                                     | _          | _          | _          | _          | _
Longformer, only depth, short, `contrastive_cos_dist` |   .48844   | **.18570** |   .15122   |   .36987   | depth_loss_short/m.k.d_l_k.l_t=contrastive_cos_dist
Longformer, only depth, short, best cos               |   .50092   |   .20209   |   .15054   |   .36688   | depth_loss_short/m.k.d_l_k.l_t=cos_dist
Longformer, only depth, short, best mse & worst cos   |   .48175   |   .19242   |   .14513   |   .36145   | depth_loss_short/m.k.d_l_k.l_t=mse
Longformer, only depth, short, worst mse              |   .39400   |   .21290   |   .12815   |   .33849   | depth_loss_short/m.k.d_l_k.l_t=contrastive_mse
Longformer, only depth, `contrastive_cos_dist`        |   .51317   |   .19551   |   .15588   | **.38260** | depth_loss/m.k.d_l_k.l_t=contrastive_cos_dist
Longformer, only depth, best cos                      | **.52884** |   .19853   | **.15768** |   .37479   | depth_loss/m.k.d_l_k.l_t=cos_dist
Longformer, only depth, best mse & worst cos          |   .51018   |   .19676   |   .15679   |   .37211   | depth_loss/m.k.d_l_k.l_t=mse
Longformer, only depth, worst mse                     |   .45011   |   .21369   |   .13731   |   .34732   | depth_loss/m.k.d_l_k.l_t=contrastive_mse
_                                                     | _          | _          | _          | _          | _
Longformer, only breadth, 100 DBOW, best cca          |   .45131   | **.18882** |   .12843   |   .34058   | soft_cca_projections_dbow_100/m.k.b_l_k.t_p=[f=256-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[f=128-a=None-n=None,f=768-a=None-n=None]
Longformer, only breadth, 100 DBOW, second best cca   |   .42199   |   .21193   |   .12354   |   .33443   | soft_cca_projections_dbow_100/m.k.b_l_k.t_p=[f=256-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[f=32-a=None-n=None,f=768-a=None-n=None], had almost the same CCA as best CCA
Longformer, only breadth, 100 DBOW, worst cca         |   .34224   |   .21146   |   .09748   |   .27116   | soft_cca_projections_dbow_100/m.k.b_l_k.t_p=[f=768-a=relu-n=None,f=1024-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[f=32-a=None-n=None,f=768-a=None-n=None]
Longformer, only breadth, 768 DBOW, best cca          | **.46365** |   .19759   | **.13410** | **.34741** | soft_cca_projections_dbow_768/m.k.b_l_k.t_p=[f=768-a=relu-n=None,f=1024-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[]
Longformer, only breadth, 768 DBOW, second best cca   |   .44600   |   .23143   |   .11935   |   .27825   | soft_cca_projections_dbow_768/m.k.b_l_k.t_p=[f=256-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[]
Longformer, only breadth, 768 DBOW, worst cca         |   .18577   |   .26955   |   .04503   |   .11923   | soft_cca_projections_dbow_768/m.k.b_l_k.t_p=[f=256-a=relu-n=None,f=768-a=None-n=None]-m.k.b_l_k.b_p=[f=128-a=None-n=None,f=768-a=None-n=None]

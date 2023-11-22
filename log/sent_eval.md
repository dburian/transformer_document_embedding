[sent_eval_toolkit]: https://github.com/facebookresearch/SentEval
[sent_eval_paper]: https://arxiv.org/pdf/1803.05449.pdf
[sent_eval_toolkit_my_fork]: https://github.com/dburian/SentEval
# SentEval

To compare my models to [SBERT](../notes/sbert.md), I've used [SentEval
toolkit][sent_eval_toolkit] introduced in [corresponding
paper][sent_eval_paper].

## Changes

The toolkit almost seems unmaintained, to get it running I've had to make some
changes to my [fork][sent_eval_toolkit_my_fork].

## Format

### Semantic Textual Similarity yearly tasks

Each STS task has several subtasks. For each subtask pearson and spearman
correlations are reported. Averaging these values gives overall mean and
weighted (by number of samples in each subtask) mean which are reported per each
task.

Example:

```python
 'STS16': {
    'all': {
        'pearson': {
            'mean': 0.823480291148185,
            'wmean': 0.8229901666745529
        },
        'spearman': {
            'mean': 0.8315505319234319,
            'wmean': 0.8311468141271767
        }
    },
    'answer-answer': {
        'nsamples': 254,
        'pearson': PearsonRResult(statistic=0.7484412314698039, pvalue=7.584409286175542e-47),
        'spearman': SignificanceResult(statistic=0.7403850270021851, pvalue=2.274115629651662e-45)
    },
    'headlines': {
        'nsamples': 249,
        'pearson': PearsonRResult(statistic=0.8435117412106601, pvalue=1.270934917080509e-68),
        'spearman': SignificanceResult(statistic=0.8624622261942524, pvalue=5.244838528358086e-75)
    },
    'plagiarism': {
        'nsamples': 230,
        'pearson': PearsonRResult(statistic=0.8239378844427931, pvalue=3.6878455448223e-58),
        'spearman': SignificanceResult(statistic=0.8356067805834284, pvalue=3.029195741288471e-61)
    },
    'postediting': {
        'nsamples': 244,
        'pearson': PearsonRResult(statistic=0.8801832225228615, pvalue=2.7755518784293122e-80),
        'spearman': SignificanceResult(statistic=0.8973216582778556, pvalue=6.312524040590851e-88)
    },
    'question-question': {
        'nsamples': 209,
        'pearson': PearsonRResult(statistic=0.8213273760948067, pvalue=2.3231727098241284e-52),
        'spearman': SignificanceResult(statistic=0.8219769675594379, pvalue=1.6521643790109859e-52)
    }
}
```

#### Comparison values

- `all.pearson.wmean`
- `all.spearman.wmean`

### Semantic Textual Similarity Benchmark & SICK Relatedness

As for any STS task the significant values are pearson and spearman
correlations.

Example:

```python
'STSBenchmark': {'devpearson': 0.8599761743046369,
                'mse': 0.8488129652983787,
                'ndev': 1500,
                'ntest': 1379,
                'pearson': 0.8227907201082828,
                'spearman': 0.8286803761890804,
                'yhat': array([1.42099433, 1.81586456, 1.16911951, ..., 4.30818082, 4.6622326 ,
    3.76806937])}}
'SICKRelatedness': {'devpearson': 0.8840489603683709,
                    'mse': 0.22278254403937933,
                    'ndev': 500,
                    'ntest': 4927,
                    'pearson': 0.8838385553373241,
                    'spearman': 0.8318364337568132,
                    'yhat': array([3.72951189, 4.02144841, 1.16334642, ..., 2.74487759, 4.72951188,
    4.79750425])},
```

#### Comparison values

- `pearson`
- `spearman`

### Classification tasks

These include:
- MR
- CR
- SUBJ
- MPQA
- SST2 (binary SST)
- TREC
- MRPC

For all of these accuracy is **the** reported metric (for MRPC there is also F1).

Example:
```python
'CR': {'acc': 87.34, 'devacc': 88.05, 'ndev': 3775, 'ntest': 3775},
'MPQA': {'acc': 89.44, 'devacc': 89.65, 'ndev': 10606, 'ntest': 10606},
'MR': {'acc': 84.84, 'devacc': 85.15, 'ndev': 10662, 'ntest': 10662},
'MRPC': {'acc': 75.01, 'devacc': 76.25, 'f1': 83.35, 'ndev': 4076, 'ntest': 1725},
'SST2': {'acc': 88.8, 'devacc': 89.11, 'ndev': 872, 'ntest': 1821},
'SUBJ': {'acc': 94.16, 'devacc': 94.58, 'ndev': 10000, 'ntest': 10000},
'TREC': {'acc': 95.2, 'devacc': 88.74, 'ndev': 5452, 'ntest': 500}
```

#### Comparison values

- `acc`

# NER

**记录相关paper的复现**


实验基于数据集[CLUE-NER](https://github.com/CLUEbenchmark/CLUENER2020)，结果如下：
    
* 验证集结果
    experiment | p | r | f1
    -- | -- | -- | --
    BILSTM-CRF | 69.04% | 68.19% | 68.62%
    TENER |  66.10% | 64.57% | 65.33%
    BERT+CRF | 79.16% | 81.61% | 80.37%
    BERT-pipeline | 82.17% | 80.86% | **81.51%**
    BERT-MRC | - | - | -
    Biaffine-NER | 84.85% | 77.31% | 80.90%
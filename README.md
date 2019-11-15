# A Modular Architecture for Unsupervised Sarcasm Generation

If you happen to use the code/data/resources shared here, fully or partially, do cite our paper. 

```
@inproceedings{mishra-etal-2019-modular,
    title = "A Modular Architecture for Unsupervised Sarcasm Generation",
    author = "Mishra, Abhijit  and
      Tater, Tarun  and
      Sankaranarayanan, Karthik",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1636",
    doi = "10.18653/v1/D19-1636",
    pages = "6146--6155"
}
```

Generation of sarcasm from literal negative opinion happens in four stages (1) Sentiment Neutralization (2) Positive Sentiment Induction (3) Negative Situation Retrieval (4) Sarcasm Synthesis. Each of the modules are independent of each other and form a pipeline during testing. They have to be trained and tuned separately. Please look inside the respective folders' `READMEs` for more details. 

## Training Data ##

Training and evaluation of individual systems require three unlabeled and non-aligned corpora (a) Sarcasm Corpus (S), (b) Positive Sentiment (P) (c) Negative Situation Corpus (N). These can be found in the dataset folder

## Testing Data ##

The 203 test examples containing <literal_sentence, sarcastic_sentence> are given under `benchmark_dataset` folder inside `dataset` folder. 

## Output of various systems

This folder contains inputs given and output obtained from various systems. Except our model variants, all the other systems receive the original input sentences given in `original_input.txt` (same are `input.txt` in benchmark dataset folder. 

## Comparision Systems ##

Pointers to various systems used for comparision are given inside the comparision system folder. We also provide a script for heuristic based sentiment flipping.  


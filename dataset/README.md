## Sentiment Dataset ##

The Sentiment Bearing corpora (P,N) have been created by taking strong sentiment sentences from different data sources like :
1. Stanford Treebank Dataset
2. IMDB Reviews Dataset
3. Amazon Product Reviews
4. Sentiment 140

## Sarcasm Dataset ##

This is a corpus containing sarcastic sentences (S). For this, sentences with sarcasm-positve gold labels are extracted from Ghosh et. al, 2016, and Riloff et. al, 2013


## Extracted Negative Situations based on Riloff et al., 2013##

The negative situations are present in `negSituations.txt`. These have been extracted using corpora P, N and S. 

## Other files ##

Files such as `sentiment_data.tsv` and `dataSamplesTest.tsv` are transformations of P,N needed to train the neutralization module. 

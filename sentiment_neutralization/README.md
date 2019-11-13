### Running this module is divided into 2 steps.

## Sentiment Classification ##
We use a LSTM model for sentiment classification which involves self attention.

- Step 1 - Run the *sentiment_classification.py* file. This takes parameters from *neutralisation_params.py* file. This builds a classifier which classifies sentences into Positive and Negative. The input here is a csv file such as *sentiment_data.csv* which requires 2 columns, text and label.

```sh
$ python sentiment_classification.py
```

## Sentiment Neutralization ##
The words which receive most attention in the sentiment classification task are removed for Sentiment Neutralization

- Step 2 - Run the *sentiment_neutralisation.py* file. This loads the model and pickle saved by previously run classification code and removes the words which had high attention depending on the label. It reads data from *dataSampleTest.csv* file which has 2 columns namely : phrase & Actual Negative.

```sh
$ python sentiment_neutralisation.py
```

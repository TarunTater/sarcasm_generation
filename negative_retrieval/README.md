The file *negative_situation_retrieval.py* is used for retrieving negative situations with for a particular text.
```sh
$ python negative_situation_retrieval.py
```
It uses Lucene based information retrieval. The python library used for this is pylucene. In case you are not able to install lucene on your system, we have also provided a Dockerfile (Docker version 18.06.1-ce / 18.09.2) which installs lucene.

The file *negSituations.txt* contains all the negative situations, the number and quality of situations can be enhanced depending on the usecase.

For each sentence in *negative_situation_to_retrieve.txt* , negative situations are retrieved. This file containes all the actual sentences, this can be changed to all neutral sentences also depending on the usecase.

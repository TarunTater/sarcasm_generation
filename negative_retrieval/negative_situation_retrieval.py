import re
import time
import json
from types import NoneType
import requests
import numpy as np
import pandas as pd
import pickle
#Lucene
import os
import lucene

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField, StoredField, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene import document, store, util
from java.nio.file import Paths

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import pandas as pd

allSentDf = pd.read_csv("../data/sentiment_data.csv")

with open("negSituations.txt", "r") as f:
    allNegSituations = f.readlines()

neg = allSentDf[allSentDf["label"] == 1]

allNegPhrases = list(neg["phrase"])

with open("../data/negSituations.txt", "r") as f:
    allNegSituations = f.readlines()

allNegSituations = map(lambda s: s.strip(), allNegSituations)
allNegPhrases = map(lambda s: s.strip(), allNegPhrases)

lucene.initVM()
    # ANALYZER
analyzer = StandardAnalyzer()

#Directory
path = Paths.get('negSituationIndex')
directory = SimpleFSDirectory(path)

writerConfig = IndexWriterConfig(analyzer)
writer = IndexWriter(directory, writerConfig)

print writer.numDocs()
# INDEXING ALL DOCUMENTS/ARTICLES IN THE CORPUS
for each in allNegSituations:
#     print(each)
    document = Document()
    document.add(Field("negativeSituations", each, TextField.TYPE_STORED))
    writer.addDocument(document)

print writer.numDocs()
writer.close()


analyzer = StandardAnalyzer()

#Directory
path = Paths.get('negSituationIndex')
directory = SimpleFSDirectory(path)
reader = DirectoryReader.open(directory)
searcher = IndexSearcher(reader)

# QUERYING FOR A QUESTION


with open("../data/test.txt", "r") as f:
    allTestSent = f.readlines()
allTestSent = map(lambda s: s.strip(), allTestSent)


queryParser = QueryParser("negativeSituations", analyzer)
num= 0
numNo = 0
totalNum = 0
ans = []
allAns = []
for each in allTestSent:
    totalNum = totalNum + 1
    if(totalNum % 1000 ==0):
        print(totalNum, time.time() - tic )
#     print("***", each, "******")
    query = queryParser.parse(queryParser.escape(each))
    hits = searcher.search(query, 3)

    docsScores = [hit.score for hit in hits.scoreDocs]
#     print docsScores
    currentAns = []
    currentAns.append(each)
    if(docsScores != []):
        num = num + 1
        ans.append(1)
        for hit in hits.scoreDocs:
            doc_id = hit.doc
            #print doc_id, hit.toString()
            docT = searcher.doc(hit.doc)
            docText = docT.get("negativeSituations")
#             currentAns.append(docText)

#             print docText
    else:
        numNo = numNo + 1
        ans.append(0)
    allAns.append(currentAns)

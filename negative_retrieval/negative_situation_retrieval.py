"""Negative Situation Retrieval Code"""

import time
import pandas as pd
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


def main():
    """Function to index negative situations and retrive based on input sentence"""

    all_sent_df = pd.read_csv("../data/sentiment_data.csv")
    neg = all_sent_df[all_sent_df["label"] == 1]
    all_neg_phrases = list(neg["phrase"])
    with open("../data/negSituations.txt", "r") as fpointer:
        all_neg_situations = fpointer.readlines()

    all_neg_situations = map(lambda s: s.strip(), all_neg_situations)
    all_neg_phrases = map(lambda s: s.strip(), all_neg_phrases)

    lucene.initVM()
    analyzer = StandardAnalyzer()
    path = Paths.get('negSituationIndex')
    directory = SimpleFSDirectory(path)
    writer_config = IndexWriterConfig(analyzer)
    writer = IndexWriter(directory, writer_config)

    print(writer.numDocs())
    # INDEXING ALL DOCUMENTS/ARTICLES IN THE CORPUS
    for each in all_neg_situations:
        document = Document()
        document.add(Field("negativeSituations", each, TextField.TYPE_STORED))
        writer.addDocument(document)

    print(writer.numDocs())
    writer.close()

    analyzer = StandardAnalyzer()
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)

    # QUERYING FOR A QUESTION
    with open("../data/negative_situation_to_retrieve.txt", "r") as fpointer:
        all_test_sent = fpointer.readlines()
    all_test_sent = map(lambda s: s.strip(), all_test_sent)

    query_parser = QueryParser("negativeSituations", analyzer)

    total_num = 0
    tic = time.time()
    all_ans = []
    for each in all_test_sent:
        total_num = total_num + 1
        if total_num % 1000 == 0:
            print(total_num, time.time() - tic)

        query = query_parser.parse(query_parser.escape(each))
        hits = searcher.search(query, 3)
        docs_scores = [hit.score for hit in hits.scoreDocs]
        current_ans = []
        if docs_scores != []:
            for hit in hits.scoreDocs:
                doc_t = searcher.doc(hit.doc)
                doc_text = doc_t.get("negativeSituations")
                current_ans.append(doc_text)
        else:
            continue

        current_ans = list(set(current_ans))
        all_ans.append(current_ans)

    print(all_ans)

if __name__ == '__main__':
	main()

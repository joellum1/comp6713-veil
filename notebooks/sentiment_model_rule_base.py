import os
import re
import sys
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# Loughran-McDonald Master Dictionary
"""
    Code taken from source website
    Load the L&M Master Dictionary from a CSV file downloaded locally in ./data directory
"""
class MasterDictionary:
    def __init__(self, cols, _stopwords):
        for ptr, col in enumerate(cols):
            if col == '':
                cols[ptr] = '0'
        try:
            self.word = cols[0].upper()
            self.sequence_number = int(cols[1])    
            self.word_count = int(cols[2])
            self.word_proportion = float(cols[3])
            self.average_proportion = float(cols[4])
            self.std_dev_prop = float(cols[5])
            self.doc_count = int(cols[6])
            self.negative = int(cols[7])
            self.positive = int(cols[8])
            self.uncertainty = int(cols[9])
            self.litigious = int(cols[10])
            self.strong_modal = int(cols[11])
            self.weak_modal = int(cols[12])
            self.constraining = int(cols[13])
            self.complexity = int(cols[14])
            self.syllables = int(cols[15])
            self.source = cols[16]
            if self.word in _stopwords:
                self.stopword = True
            else:
                self.stopword = False
        except:
            print('ERROR in class MasterDictionary')
            print(f'word = {cols[0]} : seqnum = {cols[1]}')
            quit()
        return

def load_masterdictionary(file_path, print_flag=False, f_log=None, get_other=False):
    start_local = dt.datetime.now()
    # Setup dictionaries
    _master_dictionary = {}
    _sentiment_categories = ['negative', 'positive', 'uncertainty', 'litigious', 
                             'strong_modal', 'weak_modal', 'constraining', 'complexity']
    _sentiment_dictionaries = dict()
    for sentiment in _sentiment_categories:
        _sentiment_dictionaries[sentiment] = dict()
   
    # Load slightly modified common stopwords. 
    # Dropped from traditional: A, I, S, T, DON, WILL, AGAINST
    # Added: AMONG
    _stopwords = ['ME', 'MY', 'MYSELF', 'WE', 'OUR', 'OURS', 'OURSELVES', 'YOU', 'YOUR', 'YOURS',
                  'YOURSELF', 'YOURSELVES', 'HE', 'HIM', 'HIS', 'HIMSELF', 'SHE', 'HER', 'HERS', 'HERSELF',
                  'IT', 'ITS', 'ITSELF', 'THEY', 'THEM', 'THEIR', 'THEIRS', 'THEMSELVES', 'WHAT', 'WHICH',
                  'WHO', 'WHOM', 'THIS', 'THAT', 'THESE', 'THOSE', 'AM', 'IS', 'ARE', 'WAS', 'WERE', 'BE',
                  'BEEN', 'BEING', 'HAVE', 'HAS', 'HAD', 'HAVING', 'DO', 'DOES', 'DID', 'DOING', 'AN',
                  'THE', 'AND', 'BUT', 'IF', 'OR', 'BECAUSE', 'AS', 'UNTIL', 'WHILE', 'OF', 'AT', 'BY',
                  'FOR', 'WITH', 'ABOUT', 'BETWEEN', 'INTO', 'THROUGH', 'DURING', 'BEFORE',
                  'AFTER', 'ABOVE', 'BELOW', 'TO', 'FROM', 'UP', 'DOWN', 'IN', 'OUT', 'ON', 'OFF', 'OVER',
                  'UNDER', 'AGAIN', 'FURTHER', 'THEN', 'ONCE', 'HERE', 'THERE', 'WHEN', 'WHERE', 'WHY',
                  'HOW', 'ALL', 'ANY', 'BOTH', 'EACH', 'FEW', 'MORE', 'MOST', 'OTHER', 'SOME', 'SUCH',
                  'NO', 'NOR', 'NOT', 'ONLY', 'OWN', 'SAME', 'SO', 'THAN', 'TOO', 'VERY', 'CAN',
                  'JUST', 'SHOULD', 'NOW', 'AMONG']

    # Loop thru words and load dictionaries
    with open(file_path) as f:
        _total_documents = 0
        _md_header = f.readline()  # Consume header line

        for line in f:
            cols = line.rstrip('\n').split(',')
            word = cols[0]
            _master_dictionary[word] = MasterDictionary(cols, _stopwords)
            for sentiment in _sentiment_categories:
                if getattr(_master_dictionary[word], sentiment):
                    _sentiment_dictionaries[sentiment][word] = 0
            _total_documents += _master_dictionary[cols[0]].doc_count
            if len(_master_dictionary) % 5000 == 0 and print_flag:
                print(f'\r ...Loading Master Dictionary {len(_master_dictionary):,}', end='', flush=True)

    if print_flag:
        print('\r', end='')  # clear line
        print(f'\nMaster Dictionary loaded from file:\n  {file_path}\n')
        print(f'  master_dictionary has {len(_master_dictionary):,} words.\n')

    if f_log:
        try:
            f_log.write('\n\n  FUNCTION: load_masterdictionary' +
                        '(file_path, print_flag, f_log, get_other)\n')
            f_log.write(f'\n    file_path  = {file_path}')
            f_log.write(f'\n    print_flag = {print_flag}')
            f_log.write(f'\n    f_log      = {f_log.name}')
            f_log.write(f'\n    get_other  = {get_other}')
            f_log.write(f'\n\n    {len(_master_dictionary):,} words loaded in master_dictionary.\n')
            f_log.write(f'\n    Sentiment:')
            for sentiment in _sentiment_categories:
                f_log.write(f'\n      {sentiment:13}: {len(_sentiment_dictionaries[sentiment]):8,}')
            f_log.write(f'\n\n  END FUNCTION: load_masterdictionary: {(dt.datetime.now()-start_local)}')
        except Exception as e:
            print('Log file in load_masterdictionary is not available for writing')
            print(f'Error = {e}')

    if get_other:
        return _master_dictionary, _md_header, _sentiment_categories, _sentiment_dictionaries, _stopwords, _total_documents
    else:
        return _master_dictionary



# map_function : (string, dictionary) -> [(positive 
#                   | negative 
#                   | uncertainty 
#                   | litigious 
#                   | strong_modal 
#                   | weak_modal 
#                   | constraining 
#                   | cns (Can Not Say) )]
def map_function(text, master_dictionary):
    tokens = re.split(r"[,:'\s.]", text)
    tokens = [t for t in tokens if len(t) > 1]

    text_sentiment = []
    for token in tokens:
        token_upper = token.upper()
        try:
            entry = master_dictionary[token_upper]
            if entry.stopword:
                continue
            if entry.positive > 0: text_sentiment.append("positive")
            elif entry.negative > 0: text_sentiment.append("negative")
            elif entry.uncertainty > 0: text_sentiment.append("uncertainty")
            elif entry.litigious > 0: text_sentiment.append("litigious")
            elif entry.strong_modal > 0: text_sentiment.append("strong_modal")
            elif entry.weak_modal > 0: text_sentiment.append("weak_modal")
            elif entry.constraining > 0: text_sentiment.append("constraining")
            else: text_sentiment.append("cns")
        except KeyError:
            text_sentiment.append("cns")

    return text_sentiment

# map_sentiment: [sentiment_tag] -> sentiment (positive | negative | neutral)
def map_sentiment(sentiment_list):
    counts = {}
    for s in sentiment_list:
        if s == "cns": continue
        counts[s] = counts.get(s, 0) + 1

    if not counts: return "cns"
    return max(counts, key=counts.__getitem__)

# sentiment_normalisation: sentiment -> sentiment (positive | negative | neutral)
SENTIMENT_NORMALISATION = {
    "neutral":      "neutral",
    "cns":          "neutral",
    "uncertainty":  "neutral",
    "strong_modal": "neutral",
    "weak_modal":   "neutral",
    "positive":     "positive",
    "negative":     "negative",
    "litiguous":    "negative",
    "constraining": "negative",
}

# main pipeline
def main():
    print(os.getcwd())
    DATA_PATH = "data/processed/sentiment/stitched_sentiment.csv"
    MD_PATH = "data/Loughran-McDonald_MasterDictionary_1993-2025.csv"
    LOG_PATH = "./notebooks/Load_MD_Logfile.txt"

    # Pre-processing
    df = pd.read_csv(DATA_PATH)
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df.dropna(subset=["sentiment"])

    # Load Loughran-McDonald Master Dictionary (code from source website)
    start = dt.datetime.now()
    print(f"\n\n{start.strftime('%c')}\nPROGRAM NAME: {sys.argv[0]}\n")
    with open(LOG_PATH, "w") as f_log:
        master_dictionary, md_header, sentiment_categories, \
            sentiment_dictionaries, stopwords, total_documents = \
            load_masterdictionary(MD_PATH, print_flag=True, f_log=f_log, get_other=True)
    print(f"\n\nRuntime: {dt.datetime.now() - start}")
    print(f"\nNormal termination.\n{dt.datetime.now().strftime('%c')}\n")

    # Tag tokens & predict sentiment
    df["sentiment_list"] = df["data"].apply(lambda text: map_function(text, master_dictionary))
    df["predicted_sentiment"] = df["sentiment_list"].apply(map_sentiment)
    # Normalise to positive / negative / neutral
    df["predicted_sentiment"] = df["predicted_sentiment"].map(SENTIMENT_NORMALISATION)

    # Evaluation
    eval_df = df.dropna(subset=["predicted_sentiment", "sentiment"])
    y_true = eval_df["sentiment"]
    y_pred = eval_df["predicted_sentiment"]
    labels = ["positive", "negative", "neutral"]
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics_df = pd.DataFrame(
        {"Precision": precision, "Recall": recall, "F1 Score": f1},
        index=labels,
    )
    print("Per-Class Metrics:")
    print(metrics_df.round(4))
    print("\nAggregated Metrics:")
    for avg in ["macro", "weighted"]:
        p = precision_score(y_true, y_pred, average=avg, zero_division=0)
        r = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f = f1_score(y_true, y_pred, average=avg, zero_division=0)
        print(f"{avg.capitalize():10} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")
    print("\nFull Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
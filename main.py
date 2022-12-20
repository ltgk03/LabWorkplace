import datasets
import pandas as pd
import numpy as np

import gensim
import rouge

## Load the full dataset of 300k articles
dataset = datasets.load_dataset("cnn_dailymail", "3.0.0")
lst_dics = [dic for dic in dataset["train"]]

dtf = pd.DataFrame(lst_dics).rename(columns={"article": "text", "highlights": "y"})[['text', 'y']].head(20000)
dtf.head()

i = 1
print("--- Full text ---")
print(dtf["text"][i])
print("--- Summary ---")
print(dtf["y"][i])

dtf_train = dtf.iloc[i + 1:]
dtf_test = dtf.iloc[:i + 1]

def textrank(corpus, ratio=0.2):
    if type(corpus) is str:
        corpus = [corpus]
    lst_summaries = [gensim.summarization.summarize(text, ratio=ratio) for text in corpus]
    return lst_summaries

predicted = textrank(corpus = dtf_test["text"], ratio=0.2)
print("--- Predicted ---")
print(predicted[i])

def evaluate_summary(y_test, predicted):
    rouge_score = rouge.Rouge()
    scores = rouge_score.get_scores(y_test, predicted, avg = True)
    score_1 = round(scores["rouge-1"]["f"], 2)
    score_2 = round(scores["rouge-2"]["f"], 2)
    score_L = round(scores["rouge-l"]["f"], 2)
    print("rouge1: ", score_1, "| rouge2: ", score_2, "| rougeL: ", score_L, "--> avg rouge: ", round(np.mean([score_1,score_2,score_L]), 2))
i = 1
evaluate_summary(dtf_test["y"][i], predicted[i])





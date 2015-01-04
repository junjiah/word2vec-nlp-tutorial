import re
import pandas as pd
from bs4 import BeautifulSoup
import nltk.data
from nltk.corpus import stopwords

letters_only_pattern = re.compile('[^a-zA-Z]')
stops = set(stopwords.words('english'))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def submit(model, id_list, test_features, name='bow', prob=False):
    if prob:
        p = model.predict_proba(test_features)[:, 1]
    else:
        p = model.predict(test_features)
    output = pd.DataFrame(data={'id': id_list, 'sentiment': p})
    output.to_csv("%s.csv" % name, index=False, quoting=3)


def review_to_sentences(review, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [review_to_wordlist(s, remove_stopwords)
                 for s in raw_sentences
                 if s]
    return sentences


def review_to_wordlist(raw_review, remove_stopwords=False):
    global letters_only_pattern, stops
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub(letters_only_pattern, ' ', review_text)
    words = letters_only.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in stops]
    return words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

count_vectorizer = CountVectorizer(analyzer='word',
                                   tokenizer=None,
                                   preprocessor=None,
                                   stop_words=None,
                                   max_features=5000)

tfidf_vectorizer = TfidfVectorizer(min_df=3,
                                   max_features=None,
                                   strip_accents='unicode',
                                   analyzer='word',
                                   token_pattern=r'\w{1,}',
                                   use_idf=1,
                                   smooth_idf=1,
                                   sublinear_tf=1,
                                   stop_words='english')


def bow_fit(clean_reviews):
    global count_vectorizer
    count_vectorizer.fit(clean_reviews)


def bow_transform(clean_reviews):
    global count_vectorizer
    return count_vectorizer.transform(clean_reviews).toarray()


def tfidf_fit(clean_review):
    global tfidf_vectorizer
    tfidf_vectorizer.fit(clean_review)


def tfidf_transform(clean_reviews):
    global tfidf_vectorizer
    return tfidf_vectorizer.transform(clean_reviews).toarray()
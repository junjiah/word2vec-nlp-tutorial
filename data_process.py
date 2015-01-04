from itertools import chain, imap
import pandas as pd
from gensim.models import word2vec

from bow_feature import *
from word2vec_feature import *
from utility import *


def give_me_data_and_bow(tfidf=False):
    """
    Helper function to process data and feed to BOW feature extractor.
    :param tfidf: true if want to use tf-idf as features, otherwise only tf
    :return: extracted features and original data
    """
    train = pd.read_csv('data/labeledTrainData.tsv', header=0,
                        delimiter='\t', quoting=3)
    clean_train_reviews = [" ".join(review_to_wordlist(r)) for r in train['review']]

    # testing data
    test = pd.read_csv('data/testData.tsv', header=0, delimiter='\t',
                       quoting=3)
    clean_test_reviews = [" ".join(review_to_wordlist(r)) for r in test['review']]

    print 'Feeding into %s vectorizer...' % ('tf-idf' if tfidf else 'tf')
    if tfidf:
        tfidf_fit(clean_train_reviews + clean_test_reviews)
        transform = tfidf_transform
    else:
        bow_fit(clean_train_reviews + clean_test_reviews)
        transform = bow_transform
    return map(transform, (clean_train_reviews, clean_test_reviews)) + [train, test]


def give_me_data_and_word2vec(cluster=False, model_name='model/300features_40minwords_10context'):
    """
    Helper function to process data and feed to word2vec features transformer.
    :param cluster: true if want to use k-means centroids to represent features, otherwise
    use averaged vectors
    :param model_name: model file path
    :return: extracted features and original data
    """
    # specify current model
    word2vec_model = word2vec.Word2Vec.load(model_name)

    # training data
    review_map_f = lambda r: review_to_wordlist(r, remove_stopwords=True)
    train = pd.read_csv('data/labeledTrainData.tsv', header=0, delimiter='\t',
                        quoting=3)
    test = pd.read_csv('data/testData.tsv', header=0, delimiter='\t',
                       quoting=3)
    clean_train_reviews = map(review_map_f, train['review'])
    clean_test_reviews = map(review_map_f, test['review'])

    if cluster:
        # feed into k-means, SLOW!
        word_centroid_map = word2vec_cluster(word2vec_model)
        centroid_map_f = lambda r: word2vec_transform_centroid(r, word_centroid_map)
        train_features = np.asarray(map(centroid_map_f, clean_train_reviews))
        test_features = np.asarray(map(centroid_map_f, clean_test_reviews))
    else:
        num_features = 300
        train_features = word2vec_transform_avg(clean_train_reviews, word2vec_model, num_features)
        test_features = word2vec_transform_avg(clean_test_reviews, word2vec_model, num_features)

    return train_features, test_features, train, test


def train_my_word2vec(model_name='model/300features_40minwords_10context'):
    """
    Train a word2vec model using labeled/unlabeled sentences and save to disk
    """
    # training data
    train = pd.read_csv('data/labeledTrainData.tsv', header=0,
                        delimiter='\t', quoting=3)
    unlabeled_train = pd.read_csv("data/unlabeledTrainData.tsv", header=0,
                                  delimiter="\t", quoting=3)
    # map to get sentences
    sentences = imap(lambda r: review_to_sentences(r),
                     chain(train['review'], unlabeled_train['review']))
    # flatten
    sentences = list(chain.from_iterable(sentences))

    # training word2vec
    word2vec_fit(sentences, model_name)

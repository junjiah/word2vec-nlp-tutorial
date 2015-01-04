import numpy as np
import logging
from gensim.models import word2vec
from sklearn.cluster import KMeans


def word2vec_fit(sentences, model_name):
    # set up
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    down_sampling = 1e-3

    # model training
    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=down_sampling)
    model.init_sims(replace=True)
    model.save(model_name)


def word2vec_transform_avg(reviews, model, num_features):
    def feature_vec(words, model):
        vec = sum(model[word] for word in words if word in model)
        return vec / vec.size

    cnt = 0
    review_features = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
        if cnt % 1000 == 0:
            print 'word2vec feature: processing review %d of %d' % (cnt, len(reviews))
        review_features[cnt] = feature_vec(review, model)
        cnt += 1
    return review_features


def word2vec_cluster(model):
    word_vectors = model.syn0
    k = word_vectors.shape[0] / 5

    kmeans = KMeans(n_clusters=k, verbose=2)
    centroids = kmeans.fit_predict(word_vectors)

    word_centroid_map = dict(zip(model.index2word, centroids))
    return word_centroid_map


def word2vec_transform_centroid(wordlist, word_centroid_map):
    k = max(word_centroid_map.values()) + 1
    centroid_vector = np.zeros(k, dtype='float32')
    for word in wordlist:
        if word in word_centroid_map:
            centroid_vector[word_centroid_map[word]] += 1
    return centroid_vector
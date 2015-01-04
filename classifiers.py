from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_random_forest(instances, labels):
    forest = RandomForestClassifier(n_estimators=100, verbose=2)
    return forest.fit(instances, labels)


def train_logistic_regression(instances, labels):
    lr = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                            C=1, fit_intercept=True, intercept_scaling=1.0,
                            class_weight=None, random_state=None)
    return lr.fit(instances, labels)

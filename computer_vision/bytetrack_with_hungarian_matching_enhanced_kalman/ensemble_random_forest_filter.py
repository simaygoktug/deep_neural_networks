import numpy as np
from scipy.stats import entropy 

class RandomForestEnsembleKalmanFilter(object):
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2, min_samples_leaf=1, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap

        self.estimators = []
        for _ in range(self.n_estimators):
            estimator = RandomForestRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, bootstrap=self.bootstrap)
            self.estimators.append(estimator)

    def fit(self, X, y):
        for estimator in self.estimators:
            estimator.fit(X, y)

    def predict(self, X):
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(X))
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)

class RandomForestRegressor(object):
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1, bootstrap=True):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap

        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
        self.tree_.fit(X, y)

    def predict(self, X):
        return self.tree_.predict(X)

class DecisionTreeRegressor(object):
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root_ = None

    def fit(self, X, y):
        self.root_ = Node(X, y, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)

    def predict(self, X):
        return self.root_.predict(X)

class Node(object):
    def __init__(self, X, y, max_depth, min_samples_split, min_samples_leaf):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.X = X
        self.y = y

        self.is_leaf_ = False
        self.feature_ = None
        self.threshold_ = None
        self.left_ = None
        self.right_ = None

    def fit(self):
        if self.y.shape[0] < self.min_samples_split:
            self.is_leaf_ = True
            return

        feature, threshold = self.find_best_split()
        if feature is None:
            self.is_leaf_ = True
            return

        self.feature_ = feature
        self.threshold_ = threshold

        left_X, left_y = self.X[self.X[:, feature] <= threshold], self.y[self.X[:, feature] <= threshold]
        right_X, right_y = self.X[self.X[:, feature] > threshold], self.y[self.X[:, feature] > threshold]

        self.left_ = Node(left_X, left_y, max_depth=self.max_depth - 1, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
        self.right_ = Node(right_X, right_y, max_depth=self.max_depth - 1, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)

    def find_best_split(self):
        best_gain = float("-inf")
        best_feature = None
        best_threshold = None

        for feature in range(self.X.shape[1]):
            thresholds = np.unique(self.X[:, feature])
            for threshold in thresholds:
                gain = self.information_gain(threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def information_gain(self, threshold):
        left_X, left_y = self.X[self.X[:, self.feature_] <= threshold], self.y[self.X[:, self.feature_] <= threshold]
        right_X, right_y = self.X[self.X[:, self.feature_] > threshold], self.y[self.X[:, self.feature_] > threshold]

        left_entropy = entropy(left_y)
        right_entropy = entropy(right_y)

        split_entropy = (left_X.shape[0] / self.X.shape[0]) * left_entropy + (right_X.shape[0] / self.X.shape[0]) * right_entropy

        info_gain = entropy(self.y) - split_entropy
        return info_gain

    def entropy(self, y):
        p = np.sum(y == 1) / y.shape[0]
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
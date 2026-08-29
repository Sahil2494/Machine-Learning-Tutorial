from DecisionTree import DecisionTree
from collections import Counter
import numpy as np

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, no_of_features=self.n_features)
            X_sample, y_sample = self.bootstrap_samples(X,y)
            tree.fit(X=X_sample, y=y_sample)
            self.trees.append(tree)

    #select random (data-points) datasets for each decision tree
    def bootstrap_samples(self, X, y):
        n_samples = X.shape[0]

        #replace = true ensures duplicate record may be selected
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]


    def _most_common_label(self, y):
            counter = Counter(y)
            value = counter.most_common(1)[0][0]
            #   most_common(n: int | None = None) -> list[tuple[Any, int]]
            # List the n most common elements and their counts from the most
            # common to the least. If n is None, then list all element counts.
            
            # >>> Counter('ababaa').most_common(2)
            # [('a', 4), ('b', 2)]
    
            #we selected (1)[0][0] -> 'a'
            return value

    def predict(self, X):
            predictions = np.array([tree.predict(X) for tree in self.trees])
            # return this -> [{1,0,1,1},{0,0,1,1},{1,1,0,1}], inner list includes prediction for each sample
    
            #but we need it in a different structure
            #like row 1 -> predictions of sample 1 from each tree
            #row 2 -> predictions of sample 2 from each tree
            # and so on
    
            tree_preds = np.swapaxes(predictions, 0, 1)
            predictions = np.array([self._most_common_label(tree_pred) for tree_pred in tree_preds]) 
            return predictions
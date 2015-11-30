"""
randomforestclassifier.py

Builds up a random forest classifier
"""

import math
from classifier import Classifier
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from sklearn.ensemble import RandomForestClassifier as RandForestClassifier

class RandomForestClassifier(Classifier):

    # n_estimators is the number of estimators to use in the random forest
    def __init__(self, n_estimators):
        self._forest = RandForestClassifier(n_estimators = n_estimators)
        self._n_train_recipes = 0   # count the number of training recipes seen
        self._n_ingredients = 0     # count the number of ingredients seen
        self._ingredients = {}      # map ingredient -> its number representation
        self._n_learned = 0         # number of recipes we've learned so far
        # self._matrix = None         # n_train_recipes x n_ingredients matrix, 0's and 1's (sparse)
        self._classes = []          # n_train_recipes-length list, strings indicating cuisine type
        self._has_fit = False       # have we fit the data yet?
        self._rows = []
        self._cols = []

    # We have to call this on both the train and test data sets to proper initialize our matricies
    # Run this on every recipe BEFORE calling learn()!
    def init_ingredients(self, recipe, is_train_set):
        if is_train_set:
            self._n_train_recipes += 1
        for ingredient in recipe:
                if not ingredient in self._ingredients:
                    self._ingredients[ingredient] = self._n_ingredients
                    self._n_ingredients += 1

    # Use this to build up our matricies which the random forest will use
    def learn(self, ingredients, cuisine):
        # if self._matrix == None:
            # self._matrix = csr_matrix((self._n_train_recipes, self._n_ingredients), dtype=np.dtype(bool))
            # self._matrix = lil_matrix((self._n_train_recipes, self._n_ingredients), dtype=np.dtype(bool))
            # print "Created " + str(self._n_train_recipes) + " by " + str(self._n_ingredients) + " sparse matrix"
        for ingredient in ingredients:
            # self._matrix[self._n_learned, self._ingredients[ingredient]] = True
            self._rows.append(self._n_learned)
            self._cols.append(self._ingredients[ingredient])
        self._classes.append(cuisine)
        # print "[" + str(self._n_learned) + "]" + " " + cuisine
        self._n_learned += 1

    # Run the random forest on an unclassified recipe
    def classify(self, ingredients):
        # Fit the training data to create the decision trees
        if not self._has_fit:
            print "Fitting the random forest..."
            rows = np.array(self._rows)
            cols = np.array(self._cols)
            d = np.ones((len(rows),))
            coom = coo_matrix((d, (rows, cols)), shape=(self._n_train_recipes, self._n_ingredients))
            self._forest = self._forest.fit(coom, self._classes)
            # self._forest = self._forest.fit(self._matrix, self._classes)
            print "Done!"
            self._has_fit = True
        # Run the random forest decision trees
        # We're going a little nonconventional here by running the classifier one at a time
        r = []
        c = []
        for ingredient in ingredients:
            r.append(0)
            c.append(self._ingredients[ingredient])
        rows = np.array(r)
        cols = np.array(c)
        d = np.ones((len(rows),))
        coom = coo_matrix((d, (rows, cols)), shape=(1, self._n_ingredients))
        output = self._forest.predict(coom)
        # print "Prediction: " + output[0]
        return output[0]

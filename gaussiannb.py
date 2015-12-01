"""
gaussiannb.py

Builds a Gaussian NB classifier
"""

from classifier import Classifier
from matrixdatabase import MatrixDatabase

from sklearn.naive_bayes import MultinomialNB
import numpy as np

class GaussianNBClassifier(Classifier):
	
	def __init__(self, matrixdatabase):
		self._matrix_database = matrixdatabase
		self._has_fit = False
		self._mnnb = MultinomialNB()

	def learn(self, ingredients, cuisine):
		return

	def classify(self, ingredients):
		if not self._has_fit:
			matrix, classes = self._matrix_database.make_train_matrix()
			matrix = matrix.toarray()
			self._mnnb = self._mnnb.fit(matrix, classes)
			print 'Fitting complete...'
			self._has_fit = True
		output = self._mnnb.predict(self._matrix_database.make_row_from_recipe(ingredients).toarray())
		return output[0]
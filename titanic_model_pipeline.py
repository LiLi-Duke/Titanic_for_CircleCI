import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplot.pyplot as plt


def load_data(file):
	"""
	Load data
	"""
	whole = h2o.import_file(path=file)
	


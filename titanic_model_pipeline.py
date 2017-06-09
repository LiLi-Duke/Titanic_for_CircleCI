import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config


def load_data(train_file, test_file):
	"""
	Load data
	"""
	train = h2o.import_file(path=train_file)
	test = h2o.import_file(path=test_file)

	print "The training data has %s records" % (train.shape[0])
	print "The testing data has %s records" % (test.shape[0])
	return train, test

def train_model(train, test, feature_col, target_col, model_type, outdir):
	"""
	Train a LR model or GBM model
	"""
	train[target_col] = train[target_col].asfactor()
	test[target_col] = test[target_col].asfactor()

	if model_type == "lr":
		model = H2OGeneralizedLinearEstimator(
			model_id='titanic_model',
			family='binomial',
			seed=1234)
	elif model_type == "gbm":
		model = H2OGradientBoostingEstimator(
			model_id='titanic_model')
	else:
		raise Exception('specify model type: lr or gbm')

	model.train(x=feature_col, y=target_col, training_frame=train, validation_frame=test, model_id='titanic_model')

	# save pojo
	h2o.download_pojo(model, outdir)

	print model
	return model, train


def output_auc(model, test, outdir):
	"""
	output AUC on train and test in a txt file
	"""
	train_auc = model.auc(train=True)
	test_perf = model.model_performance(test)
	test_auc = test_perf.auc()
	outfile = outdir+'/auc.txt'
	with open(outfile, 'wb') as doc:
		doc.write("train_auc: " + str(train_auc) + "\n")
		doc.write("test_auc: " + str(test_auc) + "\n")



def main():
	"""
	main
	"""
	train, test = load_data(config.train_file, config.test_file)
	model, train = train_model(train, test, config.feature_col, config.target_col, config.model_type, config.outdir)
	output_auc(model, test, config.outdir)


if __name__ == "__main__":
	h2o.init()
	main()

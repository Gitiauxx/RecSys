import os
import pandas as pd
import numpy as np
from methods.recommend.recommend.pmf import PMF
from methods.recommend.recommend.bpmf import BPMF
from methods.svd import matrix_factorization_numba as MF
#from fastFM.mcmc import FMClassification, FMRegression
from sklearn.preprocessing import OneHotEncoder
from bisect import bisect_left
from methods.boosting_rec import PMFBoosting
START_TEST = 14
DATA_DIRECTORY = './data'

PARAMETERS = {
#'pmf':{'niter': 1, 'dim': 15}, 
#'bpmf':{'niter': 60, 'dim':15},
'pmf_bias':{'niter': 20, 'dim': 15},
'pmf_boosted': {'niter': 100, 'dim': 40}
#'fastfm': {'niter': 60, 'dim': 15},
#'fastfm_features': {'niter': 100, 'dim': 30, 'features':['iid', 'cdisc', 'cohort', 'major']}
			 #'svd_bias':{'niter': 10, 'dim':15, 'reg': 0.01, 'learning_rate': 0.02}
}

DEMOG = {'race': {'WHITE', 'HISPANIC', 'BLACK', 'ASIAN'}, 'SEX':{'M', 'F'}}

GRADES = [0, 1, 4/3, 5/3, 2, 7/3, 8/3, 3, 10/3, 11/3, 4, 13/3]
 
class baseline(object):
 
	def __init__(self, dataname):
		self.data = pd.read_csv(os.path.join(DATA_DIRECTORY, dataname))
		self.n_items = len(self.data.drop_duplicates('cid'))
		self.n_users = len(self.data.drop_duplicates('sid'))
		
 
	def split_train_test(self):
		data = self.data
		data['iid'] = data['iid'] - data['iid'].min()
		
		# put in the test set the last term for each student
		data['last_term'] = data.groupby('sid').termnum.transform("max")
		data['nterm'] = data.groupby('sid').termnum.transform("size")
		#data.loc[data.grdpts < 2, 'grdpts'] = 0
		#data[data.grdpts != 1.33]
		#data = data[data.grdpts != 1.67]
		data = data[data.grdpts != 4.33]
		#data = data[data.grdpts > 0]
		
		#data = data[data.grdpts]
		
		#data1 = data[data.grdpts <=1.6]
		#data2 = data[data.grdpts > 1.6]
		#data2 = data2.loc[np.random.choice(data2.index, int(0.2 * len(data2)), replace=False), :]
		#data = pd.concat([data1, data2])
		
		# test set 
		test = data[data.termnum >= data.last_term]
		train = data[data.termnum < data.last_term]
		
		# increase weight grades
		#train2 = train[train.grdpts == 0]
		#train2 = train2.loc[np.random.choice(train2.index, 5 * len(train2), replace=True), :]
		#train3 = train[train.grdpts == 2.0]
		#train3 = train3.loc[np.random.choice(train3.index, 5 * len(train3), replace=True), :]
		#train = pd.concat([train, train2, train3])
		
		#remove cold start
		test = test[test.nterm > 1]	
		
		self.test = test
		self.train = train
		
	def svd_bias(self):
		
		dim = PARAMETERS['svd_bias']['dim']
		niter = PARAMETERS['svd_bias']['niter']
		learning_rate = PARAMETERS['svd_bias']['learning_rate']
		reg = PARAMETERS['svd_bias']['reg']
		
		train = np.array(self.train[['sid', 'cid', 'grdpts']])
		test = np.array(self.test[['sid', 'cid', 'grdpts']])
	
		mf = MF(train, 
				self.n_users, 
				self.n_items,   
				dim, 
				learning_rate,
                reg)
		
		mf_fit = mf.fit(niter)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((train.shape[0], train.shape[1] + 1))
		train_results[:, :3] = train
		train_results[:, 3] = mf.predict(train)
		
		return train_results, mf.predict(test)

	def pmf(self):
		
		dim = PARAMETERS['pmf']['dim']
		niter = PARAMETERS['pmf']['niter']
		max_grade = self.data.grdpts.max()
		
		train = np.array(self.train[['sid', 'cid', 'grdpts']])
		test = np.array(self.test[['sid', 'cid', 'grdpts']])
		
		pmf = PMF(n_user=self.n_users, 
				n_item=self.n_items, 
				n_feature=dim, 
				epsilon=1,
                max_rating=max_grade, min_rating=0, seed=0)
		
		pfm_fit = pmf.fit(train, niter)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((train.shape[0], train.shape[1] + 1))
		train_results[:, :3] = train
		train_results[:, 3] = pmf.predict(train)
		
		return train_results, pmf.predict(test)
		
	def pmf_bias(self):
		
		dim = PARAMETERS['pmf_bias']['dim']
		niter = PARAMETERS['pmf_bias']['niter']
		max_grade = self.data.grdpts.max()
		
		train = np.array(self.train[['sid', 'cid', 'grdpts']])
		test = np.array(self.test[['sid', 'cid', 'grdpts']])
		
		pmf = PMF(n_user=self.n_users, 
				n_item=self.n_items, 
				n_feature=dim, 
				epsilon=0.25,
                max_rating=max_grade, min_rating=0, seed=0)

		pfm_fit = pmf.fit_bias(train, niter, weight=False)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((train.shape[0], train.shape[1] + 1))
		train_results[:, :3] = train
		train_results[:, 3] = pmf.predict_bias(train)
		
		return train_results, pmf.predict_bias(test)
		
		
	def pmf_boosted(self):
		
		dim = PARAMETERS['pmf_boosted']['dim']
		niter = PARAMETERS['pmf_boosted']['niter']
		max_grade = self.data.grdpts.max()
		
		train = np.zeros((self.train.shape[0], self.train[['sid', 'cid', 'grdpts']].shape[1] + 1))
		train[:, :3] = np.array(self.train[['sid', 'cid', 'grdpts']])
		train[:, 3] = 1 
		test = np.array(self.test[['sid', 'cid', 'grdpts']])
		
		pmf = PMFBoosting(n_user=self.n_users, 
				n_item=self.n_items, 
				n_feature=dim, 
				epsilon=0.5,
                max_rating=max_grade, min_rating=0, seed=0, 
				delta=0.5, nboost=20)

		pfm_fit = pmf.iter_boosting(train, niter)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((train.shape[0], train.shape[1]))
		train_results[:, :3] = train[:, :3]
		train_results[:, 3] = pmf.predict_boosted(train)
		
		return train_results, pmf.predict_boosted(test)
		
	def bpmf(self):
		
		dim = PARAMETERS['bpmf']['dim']
		niter = PARAMETERS['bpmf']['niter']
		max_grade = self.data.grdpts.max()
		
		train = np.array(self.train[['sid', 'cid', 'grdpts']])
		test = np.array(self.test[['sid', 'cid', 'grdpts']])
		
		bpmf = BPMF(n_user=self.n_users, 
				n_item=self.n_items, 
				n_feature=dim, 
                max_rating=max_grade, min_rating=0, seed=0).fit(train, niter)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((train.shape[0], train.shape[1] + 1))
		train_results[:, :3] = train
		train_results[:, 3] = bpmf.predict(train)
		
		return train_results, bpmf.predict(test)
		
	def fastfm(self):
		
		n_iter = PARAMETERS['fastfm']['niter']
		rank = PARAMETERS['fastfm']['dim']
		
		trainX0 = np.array(self.train[['sid', 'cid']])
		trainY = np.array(self.train[['grdpts']])
		
		testX = np.array(self.test[['sid', 'cid']])
		testY = np.array(self.test[['grdpts']])
		
		encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX0)
		trainX = encoder.transform(trainX0)
		testX = encoder.transform(testX)
    
		clf = FMRegression(rank=rank, n_iter=n_iter)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((trainX0.shape[0], trainX0.shape[1] + 2))
		train_results[:, :2] = trainX0
		train_results[:, 2] = trainY.T
		train_results[:, 3] = clf.fit_predict(trainX, trainY.flatten(), trainX)
		
		return train_results, clf.fit_predict(trainX, trainY.flatten(), testX)
		
	def fastfm_features(self):
		
		n_iter = PARAMETERS['fastfm_features']['niter']
		rank = PARAMETERS['fastfm_features']['dim']
		features = PARAMETERS['fastfm_features']['features']
	
		
		trainX0 = np.array(self.train[['sid', 'cid'] + features])
		trainY = np.array(self.train[['grdpts']])
		
		testX = np.array(self.test[['sid', 'cid'] + features])
		testY = np.array(self.test[['grdpts']])
		
		encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX0)
		trainX = encoder.transform(trainX0)
		testX = encoder.transform(testX)
    
		clf = FMRegression(rank=rank, n_iter=n_iter)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((trainX0.shape[0], 4))
		train_results[:, :2] = trainX0[:, :2]
		train_results[:, 2] = trainY.T
		train_results[:, 3] = clf.fit_predict(trainX, trainY.flatten(), trainX)
		
		return train_results, clf.fit_predict(trainX, trainY.flatten(), testX)
    	
	def rmse(self, predicts, observed):
		return np.sqrt(np.mean((predicts - observed) ** 2))
		
	def std(self, predicts, observed):
		return np.sum((predicts - observed) ** 2) / predict.shape[0]
		
	def run_array(self, rnum, minority=False):
	
		self.split_train_test()
		self.results = pd.DataFrame()
	
		for algo, algo_par in PARAMETERS.items():
			recsys = getattr(self, algo)
			
			train_predict, test_predict = recsys()
			trainp = pd.DataFrame(train_predict)
			trainp.columns = ['sid', 'cid', 'grdpts', 'predicted']
			train = pd.merge(trainp[['sid', 'cid', 'predicted']], self.train, on=['sid', 'cid'])
			
			self.results.ix[algo, 'rmse_train'] = self.rmse(train.predicted, train.grdpts)
			
			testp = self.test[['sid', 'cid']]
			testp['predicted'] = test_predict
			test = pd.merge(testp, self.test, on=['sid', 'cid'])
			self.results.ix[algo, 'rmse_test'] = self.rmse(test.predicted, test.grdpts)
			
			if minority:
				for varname in DEMOG:
					for var in DEMOG[varname]:
						self.results.ix[algo, 'rmse_train_%s' % (var.lower())] = self.rmse(train[train[varname] == var].predicted, 
																			train[train[varname] == var].grdpts)
			
						testp = self.test[['sid', 'cid']]
						testp['predicted'] = test_predict
						test = pd.merge(testp, self.test, on=['sid', 'cid'])
						self.results.ix[algo, 'rmse_test_%s' % (var.lower())] = self.rmse(test[test[varname] == var].predicted, 
																	test[test[varname] == var].grdpts)
			
			
		self.results['runID'] = int(rnum)
		return self.results
		
	def mse(self, table):
		table['diff'] = (table.predicted - table.grdpts) ** 2
		series = table.groupby('sid').diff.mean()
		return np.mean(series)
	
	def mse_per_grade(self, table):
		table['diff'] = (table.predicted - table.grdpts) ** 2
		series = table.groupby(['sid', 'grdpts'])[['diff']].mean().reset_index()
		return series.groupby('grdpts').diff.mean()
		
	def var_adj(self, table):
		table['diff'] = (table.predicted - table.grdpts) ** 2
		series = table.groupby('sid').diff.mean()
		return np.var(series)  / len(series)
		
		
	def run(self, minority=False):
	
		self.split_train_test()
		self.results = pd.DataFrame()
	
		for algo, algo_par in PARAMETERS.items():
			recsys = getattr(self, algo)
			
			train_predict, test_predict = recsys()
			trainp = pd.DataFrame(train_predict)
			trainp.columns = ['sid', 'cid', 'grdpts', 'predicted']
			train = pd.merge(trainp[['sid', 'cid', 'predicted']], self.train, on=['sid', 'cid'])
			self.results.ix[algo, 'mse_train'] = self.mse(train)
			
			testp = self.test[['sid', 'cid']]
			testp['predicted'] = test_predict
			test = pd.merge(testp, self.test, on=['sid', 'cid'])
			self.results.ix[algo, 'mse_test'] = self.mse(test)
			
			if minority:
				for varname in DEMOG:
					for var in DEMOG[varname]:
			
						self.results.ix[algo, 'mse_test_%s' % (var.lower())] = self.mse(test[test[varname] == var])
						self.results.ix[algo, 'mse_train_%s' % (var.lower())] = self.mse(train[train[varname] == var])
						self.results.ix[algo, 'mse_test_stem_%s' % (var.lower())] = self.mse(test[(test[varname] == var) & (test['in_stem'] == 1)])
						self.results.ix[algo, 'mse_train_stem_%s' % (var.lower())] = self.mse(train[(train[varname] == var) & (train['in_stem'] == 1)])
			
						self.results.ix[algo, 'var_test_%s' % (var.lower())] = self.var_adj(test[test[varname] == var])
						self.results.ix[algo, 'var_train_%s' % (var.lower())] = self.var_adj(train[train[varname] == var])
						self.results.ix[algo, 'var_test_stem_%s' % (var.lower())] = self.var_adj(test[(test[varname] == var) & (test['in_stem'] == 1)])
						self.results.ix[algo, 'var_train_stem_%s' % (var.lower())] = self.var_adj(train[(train[varname] == var) & (train['in_stem'] == 1)])

		return self.results, test
		
	def run_mse(self):
		self.split_train_test()
		results = pd.DataFrame(index=self.data.drop_duplicates('grdpts').grdpts)
		
		train_predict, test_predict = self.pmf_boosted()
			
		testp = self.test[['sid', 'cid']]
		testp['predicted'] = test_predict
		test = pd.merge(testp, self.test, on=['sid', 'cid'])
		
		trainp = pd.DataFrame(train_predict)
		trainp.columns = ['sid', 'cid', 'grdpts', 'predicted']
		train = pd.merge(trainp[['sid', 'cid', 'predicted']], self.train, on=['sid', 'cid'])
		
		print(train[train.grdpts == 1][['predicted']].describe())
		results['All_TR'] = self.mse_per_grade(train)
		
		
		results['All'] = self.mse_per_grade(test)
		
		#for varname in DEMOG:
			#for var in DEMOG[varname]:
				#results[var.lower()] = self.mse_per_grade(test[test[varname] == var]) 
		
		return results


class experiment_size(baseline):

	def __init__(self, dataname):
		super().__init__(dataname)

	def run(self, dim_min, dim_max, minority=False):
	
		self.split_train_test()
		self.results = pd.DataFrame()
		
		for dim  in np.arange(dim_min, dim_max + 1, 5):
			
			train_predict, test_predict = self.fastfm_features(dim)
			trainp = pd.DataFrame(train_predict)
			trainp.columns = ['sid', 'cid', 'grdpts', 'predicted']
			train = pd.merge(trainp[['sid', 'cid', 'predicted']], self.train, on=['sid', 'cid'])
			self.results.loc[dim, 'mse_train'] = self.mse(train)
			
			testp = self.test[['sid', 'cid']]
			testp['predicted'] = test_predict
			test = pd.merge(testp, self.test, on=['sid', 'cid'])
			self.results.ix[dim, 'mse_test'] = self.mse(test)
		
			if minority:
				for varname in DEMOG:
					for var in DEMOG[varname]:
			
						testp = self.test[['sid', 'cid']]
						testp['predicted'] = test_predict
						test = pd.merge(testp, self.test, on=['sid', 'cid'])
						self.results.loc[dim, 'mse_test_%s' % (var.lower())] = self.mse(test[test[varname] == var])
						self.results.loc[dim, 'mse_train_%s' % (var.lower())] = self.mse(train[train[varname] == var])
			
						testp = self.test[['sid', 'cid']]
						testp['predicted'] = test_predict
						test = pd.merge(testp, self.test, on=['sid', 'cid'])
						self.results.loc[dim, 'var_test_%s' % (var.lower())] = self.var_adj(test[test[varname] == var])
						self.results.loc[dim, 'var_train_%s' % (var.lower())] = self.var_adj(train[train[varname] == var])
			
		return self.results
			
	def fastfm_features(self, dim):
		
		n_iter = PARAMETERS['fastfm_features']['niter']
		rank = dim
		features = PARAMETERS['fastfm_features']['features']
	
		
		trainX0 = np.array(self.train[['sid', 'cid'] + features])
		trainY = np.array(self.train[['grdpts']])
		
		testX = np.array(self.test[['sid', 'cid'] + features])
		testY = np.array(self.test[['grdpts']])
		
		encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX0)
		trainX = encoder.transform(trainX0)
		testX = encoder.transform(testX)
    
		clf = FMRegression(rank=rank, n_iter=n_iter)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((trainX0.shape[0], 4))
		train_results[:, :2] = trainX0[:, :2]
		train_results[:, 2] = trainY.T
		train_results[:, 3] = clf.fit_predict(trainX, trainY.flatten(), trainX)
		
		return train_results, clf.fit_predict(trainX, trainY.flatten(), testX)
		
class experiment_var(baseline):

	def __init__(self, dataname):
		super().__init__(dataname)
		
	def split_train_test(self, reduction, majority):

		data = self.data
		data['iid'] = data['iid'] - data['iid'].min()
		
		# put in the test set the last term for each student
		data['last_term'] = data.groupby('sid').termnum.transform("max")
		data['nterm'] = data.groupby('sid').termnum.transform("size")
		
		# reduce the size of the majority
		varname = majority[0]
		var = majority[1]
		
		data2 = data[data[varname] == var]
		data = data.loc[data[varname] != var]
		data2 = data2.loc[np.random.choice(data2.index, int(reduction * len(data2)), replace=False)]
		
		data = pd.concat([data, data2])
		
		# test set 
		test = data[data.termnum >= data.last_term]
		test = test.loc[np.random.choice(test.index, int(0.5 * len(test)), replace=False)]
		train = data.loc[~data.index.isin(test.index)]
		
		#remove cold start
		test = test[test.nterm > 1]	
		
		self.test = test
		self.train = train
		
	def run(self, reduction_min, reduction_max, majority):
	
		self.results = pd.DataFrame()
		
		for r in np.arange(reduction_min, reduction_max, 5):
			
			size = 0.2 + r/100
			self.split_train_test(size, majority)
			
			train_predict, test_predict = self.fastfm_features()
			trainp = pd.DataFrame(train_predict)
			trainp.columns = ['sid', 'cid', 'grdpts', 'predicted']
			train = pd.merge(trainp[['sid', 'cid', 'predicted']], self.train, on=['sid', 'cid'])
			self.results.loc[r, 'mse_train'] = self.mse(train)
			
			testp = self.test[['sid', 'cid']]
			testp['predicted'] = test_predict
			test = pd.merge(testp, self.test, on=['sid', 'cid'])
			self.results.ix[r, 'mse_test'] = self.mse(test)
		

			for varname in DEMOG:
				for var in DEMOG[varname]:
			
				
					self.results.loc[r, 'mse_test_%s' % (var.lower())] = self.mse(test[test[varname] == var])
					self.results.loc[r, 'mse_train_%s' % (var.lower())] = self.mse(train[train[varname] == var])
			
					
					self.results.loc[r, 'var_test_%s' % (var.lower())] = self.var_adj(test[test[varname] == var])
					self.results.loc[r, 'var_train_%s' % (var.lower())] = self.var_adj(train[train[varname] == var])
			
		return self.results

class experiment_noise(baseline):
	
	def __init__(self, dataname):
		super().__init__(dataname)

	def fastfm_features(self, demog=None):
		
		n_iter = PARAMETERS['fastfm_features']['niter']
		rank = PARAMETERS['fastfm_features']['dim']
		if demog is None:
			features = PARAMETERS['fastfm_features']['features']
		else:
			features = PARAMETERS['fastfm_features']['features'] + demog
	
		trainX0 = np.array(self.train[['sid', 'cid'] + features])
		trainY = np.array(self.train[['grdpts']])
		
		testX = np.array(self.test[['sid', 'cid'] + features])
		testY = np.array(self.test[['grdpts']])
		
		encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX0)
		trainX = encoder.transform(trainX0)
		testX = encoder.transform(testX)
    
		clf = FMRegression(rank=rank, n_iter=n_iter)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((trainX0.shape[0], 4))
		train_results[:, :2] = trainX0[:, :2]
		train_results[:, 2] = trainY.T
		train_results[:, 3] = clf.fit_predict(trainX, trainY.flatten(), trainX)
		
		return train_results, clf.fit_predict(trainX, trainY.flatten(), testX)
		
	def run(self, demog=None):
	
		self.results = pd.DataFrame()
		self.split_train_test()
		
		if demog is None:
			train_predict, test_predict = self.fastfm_features()
			r = 'no_features'
		else:
			train_predict, test_predict = self.fastfm_features(demog)
			r = ''.join('%s_' %feature for feature in demog)
					
		trainp = pd.DataFrame(train_predict)
		trainp.columns = ['sid', 'cid', 'grdpts', 'predicted']
		train = pd.merge(trainp[['sid', 'cid', 'predicted']], self.train, on=['sid', 'cid'])
		self.results.loc[r, 'mse_train'] = self.mse(train)
			
		testp = self.test[['sid', 'cid']]
		testp['predicted'] = test_predict
		test = pd.merge(testp, self.test, on=['sid', 'cid'])
		self.results.ix[r, 'mse_test'] = self.mse(test)
		

		for varname in DEMOG:
			for var in DEMOG[varname]:
			
				testp = self.test[['sid', 'cid']]
				testp['predicted'] = test_predict
				test = pd.merge(testp, self.test, on=['sid', 'cid'])
				self.results.loc[r, 'mse_test_%s' % (var.lower())] = self.mse(test[test[varname] == var])
				self.results.loc[r, 'mse_train_%s' % (var.lower())] = self.mse(train[train[varname] == var])
			
				testp = self.test[['sid', 'cid']]
				testp['predicted'] = test_predict
				test = pd.merge(testp, self.test, on=['sid', 'cid'])
				self.results.loc[r, 'var_test_%s' % (var.lower())] = self.var_adj(test[test[varname] == var])
				self.results.loc[r, 'var_train_%s' % (var.lower())] = self.var_adj(train[train[varname] == var])
			
		return self.results
	
class experiment_bin(baseline):
	
	def __init__(self, dataname):
		super().__init__(dataname)
		
	def takeClosest(self, myList, myNumber):
		"""
		Assumes myList is sorted. Returns closest value to myNumber.

		If two numbers are equally close, return the smallest number.
		"""
		pos = bisect_left(myList, myNumber)
		if pos == 0:
			return myList[0]
		if pos == len(myList):
			return myList[-1]
		before = myList[pos - 1]
		after = myList[pos]
		if after - myNumber < myNumber - before:
			return after
		else:
			return before

	def fastfm_features(self, demog=None):
		
		n_iter = PARAMETERS['fastfm_features']['niter']
		rank = PARAMETERS['fastfm_features']['dim']
		if demog is None:
			features = PARAMETERS['fastfm_features']['features']
		else:
			features = PARAMETERS['fastfm_features']['features'] + demog
	
		trainX0 = np.array(self.train[['sid', 'cid'] + features])
		trainY = np.array(self.train[['grdpts']])
		
		testX = np.array(self.test[['sid', 'cid'] + features])
		testY = np.array(self.test[['grdpts']])
		
		encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX0)
		trainX = encoder.transform(trainX0)
		testX = encoder.transform(testX)
    
		clf = FMRegression(rank=rank, n_iter=n_iter)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((trainX0.shape[0], 4))
		train_results[:, :2] = trainX0[:, :2]
		train_results[:, 2] = trainY.T
		train_results[:, 3] = clf.fit_predict(trainX, trainY.flatten(), trainX)
		
		test_results = clf.fit_predict(trainX, trainY.flatten(), testX)
		test_results_grade = np.zeros(test_results.shape[0])
		for i in range(test_results.shape[0]):
			test_results_grade[i] = self.takeClosest(GRADES, test_results[i])
		
		return train_results, test_results_grade	
		
if __name__ =="__main__":
	import sys
	"""
	expNoise = experiment_noise(sys.argv[1])
	demog_list = [['sex'], ['srace'],  ['sex', 'srace'] , ['is_male'], ['is_black'], ['is_male', 'is_black']]
	
	
	res1 = expNoise.run()
	res_list = [res1]
	for features in demog_list:
		res2 = expNoise.run(features)
		res_list.append(res2)
	res = pd.concat(res_list)
	
	"""
	base = baseline(sys.argv[1])
	res = base.run_mse()
	
	
	for col in list(res.columns):
			sys.stdout.write("%s, " %col)
	sys.stdout.write('\n')
	for i in res.index:
		sys.stdout.write("%s, "%i)
		for col in list(res.columns):
			wres = res.loc[i, col]
			if type(wres) is float:
				sys.stdout.write("%5f, " %wres)
			else:
				sys.stdout.write("%s, " %wres)
				
		sys.stdout.write('\n')
	#base.results.to_csv("./results/baseline_rmse_%s" %sys.argv[0])
	
	
	
	
	
		
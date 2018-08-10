import pandas as pd
from methods.mfc import MFC
import os
import numpy as np


START_TEST = 14
DATA_DIRECTORY = './data'

PARAMETERS = {
#'pmf':{'niter': 60, 'dim': 15}, 
#'bpmf':{'niter': 60, 'dim':15},
#'pmf_bias':{'niter': 60, 'dim': 15},
#'fastfm': {'niter': 60, 'dim': 15},
'mfc': {'niter': 30, 'dim': 10}
			 #'svd_bias':{'niter': 10, 'dim':15, 'reg': 0.01, 'learning_rate': 0.02}
}

DEMOG = {'race': {'WHITE', 'HISPANIC', 'BLACK', 'ASIAN'}, 'SEX':{'M', 'F'}}
 
class CMethod(object):
 
	def __init__(self, dataname):
		self.data = pd.read_csv(os.path.join(DATA_DIRECTORY, dataname))
		self.n_items = len(self.data.drop_duplicates('cid'))
		self.n_users = len(self.data.drop_duplicates('sid'))
		
	
	def split_train_test(self, size=None):
		data = self.data
		data['iid'] = data['iid'] - data['iid'].min()
		
		# observed failing rates
		data['failing'] = (data.grdpts <= 2).astype('int32')
		
		
		# random draw of a subset of the data
		if size is not None:
			data = data.loc[np.random.choice(data.index, int(size * len(data)), replace=False), :]
		
		# put in the test set the last term for each student
		data['last_term'] = data.groupby('sid').termnum.transform("max")
		data['nterm'] = data.groupby('sid').termnum.transform("size")
		
		# test set 
		test = data[data.termnum == data.last_term]
		train = data[data.termnum < data.last_term]
		
		# increase weight failing grades
		train2 = train.loc[np.random.choice(train[train.grdpts <= 2].index, 6 * len(train[train.grdpts <= 2]), replace=True), :]
		#train = pd.concat([train, train2])
		
		#remove cold start
		test = test[test.nterm > 1]	
		
		self.test = test
		self.train = train
		
	def mfc(self):
			
		dim = PARAMETERS['mfc']['dim']
		niter = PARAMETERS['mfc']['niter']
		
		train = np.array(self.train[['sid', 'cid', 'failing']])
		test = np.array(self.test[['sid', 'cid', 'failing']])
		
		mfc = MFC(n_users=self.n_users, 
				n_items=self.n_items, 
				dim=dim, 
				learning_rate=0.01,  
				n_iter=niter)

		mfc.fit(train)
		
		# collect all train data to keep track of the order (otherwise reshuffled)
		train_results = np.zeros((train.shape[0], train.shape[1] + 1))
		train_results[:, :3] = train
		train_results[:, 3] = mfc.predict(train)
		
		return train_results, mfc.predict(test)
		
	def fnr(self, predicted, observed):
		return np.mean(predicted[observed == 0])
		
	def fpr(self, predicted, observed):
		return np.mean( 1 - predicted[observed == 1])
		
	def one_loss(self, predicted, observed):
		diff_obs_pred = predicted - observed
		res = (diff_obs_pred == 0).astype('int32')
		return np.sum(res) / len(res)
		
		
	def run_size(self, size_min, size_max):
	
		self.results = pd.DataFrame()
		
		for i in range(size_min, size_max + 1):
			size = i / 10 
			self.split_train_test(size)
			
			train_predict, test_predict = self.mfc()
			trainp = pd.DataFrame(train_predict)
			trainp.columns = ['sid', 'cid', 'failing', 'predicted']
			train = pd.merge(trainp[['sid', 'cid', 'predicted']], self.train, on=['sid', 'cid'])
			
			self.results.ix[size, 'fpr_train'] = self.fpr(train.predicted, train.failing)
			self.results.ix[size, 'fnr_train'] = self.fnr(train.predicted, train.failing)
			
			testp = self.test[['sid', 'cid']]
			testp['predicted'] = test_predict
			test = pd.merge(testp, self.test, on=['sid', 'cid'])
			
			self.results.ix[size, 'fpr_test'] = self.fpr(test.predicted, test.failing)
			self.results.ix[size, 'fnr_test'] = self.fnr(test.predicted, test.failing)
			self.results.ix[size, 'accuracy_test'] = self.one_loss(test.predicted, test.failing)
			
			for varname in DEMOG:
				for var in DEMOG[varname]:
			
					t = test[test[varname] == var]
					self.results.loc[size, 'fpr_test_%s' % (var.lower())] = self.fpr(t.predicted, t.failing)
					self.results.loc[size, 'fnr_test_%s' % (var.lower())] = self.fnr(t.predicted, t.failing)
					self.results.loc[size, 'accuracy_test_%s' % (var.lower())] = self.one_loss(t.predicted, t.failing)
					
					tr = train[train[varname] == var]
					self.results.loc[size, 'fpr_train_%s' % (var.lower())] = self.fpr(tr.predicted, tr.failing)
					self.results.loc[size, 'fpr_train_%s' % (var.lower())] = self.fnr(tr.predicted, tr.failing)	
					self.results.loc[size, 'accuracy_train_%s' % (var.lower())] = self.one_loss(tr.predicted, tr.failing)	
			
		return self.results
		
		
if __name__ =="__main__":
	import sys
	
	cmethod = CMethod(sys.argv[1])
	res = cmethod.run_size(10, 10)
	
	
	
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
		
	#res.to_csv('.\\results\\classifier_failures_rebalanced.csv')
			
		
		
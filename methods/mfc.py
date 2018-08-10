import numpy as np

class MFC(object):

	def __init__(self, n_users, n_items, dim, learning_rate=0.01, reg=0.05, batch_size=100, n_iter=10):

		self._user_features = np.random.normal(scale=1/dim, size =(n_users, dim))
		self._item_features = np.random.normal(scale=1/dim, size =(n_items, dim))
		
		# batch size
		self.batch_size = batch_size
		
		# learning rate
		self.learning_rate = learning_rate
		
		# regularizer
		self.reg = reg
		
		# iteration
		self.n_iter = n_iter
	
	def gradient_descent(self, rating, batch):
	
		start_batch = int(batch * self.batch_size)
		end_batch = min(rating.shape[0], int((batch + 1) * self.batch_size))
		data = rating[start_batch:end_batch, :]
		
		user = self._user_features.take(data[:, 0].astype('int32'), axis=0)
		item = self._item_features.take(data[:, 1].astype('int32'), axis=0)
		
		pred = np.sum(np.multiply(user, item), axis=1)
		pred = np.exp(pred)
		pred = pred / (1 + pred)
		
		err = pred - data[:, 2] 
		
		update_u = np.multiply(item, err[:, np.newaxis]) + self.reg * user
		update_v = np.multiply(user, err[:, np.newaxis]) + self.reg * item
		
		self._u_feature_mom[data[:,0].astype('int32'), :] = 0.8 * self._u_feature_mom[data[:,0].astype('int32'), :] \
															+ self.learning_rate / data.shape[0] * update_u
		self._i_feature_mom[data[:,1].astype('int32'), :] = 0.8 * self._i_feature_mom[data[:,1].astype('int32'), :] \
															+ self.learning_rate / data.shape[0] * update_v	
															
		# update latent variables
		self._user_features -= self._u_feature_mom
		self._item_features -= self._i_feature_mom

	def fit(self, rating):
	
		batch_num = int(np.ceil(rating.shape[0] / self.batch_size))
		
		# momentum
		self._u_feature_mom = np.zeros(self._user_features.shape)
		self._i_feature_mom = np.zeros(self._item_features.shape)
		
		for iter in np.arange(self.n_iter):
			
			np.random.shuffle(rating)
			for batch in range(batch_num):
				self.gradient_descent(rating, batch)
	
	def predict(self, test):
		
		user = self._user_features.take(test[:, 0].astype('int32'), axis=0)
		item = self._item_features.take(test[:, 1].astype('int32'), axis=0)
		
		pred = np.sum(np.multiply(user, item), axis=1)
		pred = np.exp(pred)
		pred = pred / (1 + pred)
		
		return (pred > 0.5 ).astype('int32')
		
						
if __name__ == "__main__":

	import pandas as pd
	data = pd.read_csv(".\data\preprocessed_students.csv")
	data['failing'] = (data.grdpts == 0.0).astype('int32')
	data['last_term'] = data.groupby('sid').termnum.transform("max")
	data['nterm'] = data.groupby('sid').termnum.transform("size")
	
	test = data[data.termnum == data.last_term]
	test = test[test.nterm > 1]	
	train = data[data.termnum < data.last_term]
	
	n_user = len(data.drop_duplicates('sid'))
	n_item = len(data.drop_duplicates('cid'))
	rating = np.array(train[['sid', 'cid', 'failing']])
	
	mfc = MFC(n_user, n_item, 40)
	mfc.fit(rating, np.array(test[['sid', 'cid', 'failing']]))
	pred, err = mfc.predict(np.array(test[['sid', 'cid', 'failing']]))
	test['predicted'] = pred
	test['pred'] = (test.predicted > 0.5).astype('int32')
	print(test[test.failing == 0].groupby('SEX').pred.mean())
	print(test[test.failing == 1].groupby('SEX').pred.mean())
	print(err)
	
			
		
		
		
		
		
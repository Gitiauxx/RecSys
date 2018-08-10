import numpy as np
from methods.recommend.recommend.pmf import PMF
import pandas as pd
from numpy.random import RandomState

class PMFBoosting(PMF):

	def __init__(self, n_user, n_item, n_feature, batch_size=1000, epsilon=50.0,
                 momentum=0.8, seed=None, reg=1e-2, converge=1e-10,
                 max_rating=None, min_rating=None, delta=0.5, nboost=10):
		
		super(PMF, self).__init__()
		
		self.n_user = n_user
		self.n_item = n_item
		self.n_feature = n_feature

		self.random_state = RandomState(seed)

		# batch size
		self.batch_size = batch_size

		# learning rate
		self.epsilon = float(epsilon)
		self.momentum = float(momentum)
		
		# regularization parameter
		self.reg = reg
		self.converge = converge
		self.max_rating = float(max_rating) \
			if max_rating is not None else max_rating
		self.min_rating = float(min_rating) \
			if min_rating is not None else min_rating

		# data state
		self.mean_rating_ = None
        
		# user/item features
		self.user_features_ = 0.1 * self.random_state.rand(n_user, n_feature)
		self.item_features_ = 0.1 * self.random_state.rand(n_item, n_feature)
		self.a = 0.05 * self.random_state.rand(n_user)
		self.b = 0.05 * self.random_state.rand(n_item)
		
		self.delta = delta
		self.nboost = nboost
		self.alpha = np.zeros(nboost)
		self.models = {}
		
		
	def disagreement(self, data, niter):
	
		
		predicted = self.predict_bias(data)
		#predicted[predicted < 1] = 0
		d = pd.DataFrame(predicted)
		d.columns = ['predicted']
		d['grade'] = data[:, 2]
		print(d[d.grade ==1].predicted.describe())
		difference = (predicted - data[:, 2]) * 2
		disag = 0.5 *(difference > self.delta).astype('int32') + 0.5 *(difference > 2 * self.delta).astype('int32')
		
		return disag
		
	def update_weight(self, disag, train):
	
		epsilon = np.multiply(disag, train[:, 3])
		beta = epsilon.sum() / (1 - epsilon.sum())
		
		train[:, 3] = train[:, 3] * np.exp((1 - disag) * np.log(beta))
		#train[train[:, 3] < 10 ** (-7), 3] =  10 ** (-7)
			
		return epsilon.sum()
		
	def iter_boosting(self, train, niter):
	
	
		for iter in range(self.nboost):
		
			self.user_features_ = 0.1 * self.random_state.rand(self.n_user, self.n_feature)
			self.item_features_ = 0.1 * self.random_state.rand(self.n_item, self.n_feature)
			self.a = 0.05 * self.random_state.rand(self.n_user)
			self.b = 0.05 * self.random_state.rand(self.n_item)
		
			train[:, 3] = train[:, 3] / train[:, 3].sum()
			d = pd.DataFrame(train)
			d.columns = ['sid', 'cid', 'grdpts', 'weight']
			print(d.groupby('grdpts').weight.mean())
		
			self.fit_bias(train, n_iters=niter, weight=True)
			disag = self.disagreement(train, iter)
			epsilon = self.update_weight(disag, train)
			
			print(epsilon)
			
			if epsilon > 0.5: 
				break
			
			self.alpha[iter] = np.log((1 - epsilon) / epsilon)
			
			self.models[iter] = {}
			
			self.models[iter]['users'] = self.user_features_
			self.models[iter]['items'] = self.item_features_
			self.models[iter]['a'] = self.a
			self.models[iter]['b'] = self.b
	
	def predict_boosted(self, data):
	
		alpha = self.alpha / self.alpha.sum()
		pred = np.zeros(data.shape[0])
		print(alpha)
		
		for i in np.arange(alpha.shape[0]):
		

			if alpha[i] > 0:
				model = self.models[i]
		
				u_features = model['users'].take(data.take(0, axis=1).astype('int32'), axis=0)
				i_features = model['items'].take(data.take(1, axis=1).astype('int32'), axis=0)
				a_features = model['a'].take(data.take(0, axis=1).astype('int32'), axis=0)
				b_features = model['b'].take(data.take(1, axis=1).astype('int32'), axis=0)
				preds = a_features + b_features + np.sum(u_features * i_features, 1) 
				#+ self.mean_rating_
				pred += alpha[i] * preds
			
		
		if self.max_rating:
			pred[pred > self.max_rating] = self.max_rating

		#if self.min_rating:
			#pred[pred < self.min_rating] = 1
			#self.min_rating
			
		#pred[pred < 1] = 0
			
		return pred
		
		
		
	
		
	
		
		
			
			
if __name__ == "__main__":

		rec = PMFBoosting(2, 3, 3)
		
		
	
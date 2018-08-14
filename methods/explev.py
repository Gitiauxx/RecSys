import numpy as np
from methods.recommend.recommend.pmf import PMF
import pandas as pd
from numpy.random import RandomState
import copy
from scipy.spatial import cKDTree

class PMFBoosting(PMF):

	def __init__(self, n_user, n_item, n_feature, batch_size=1000, epsilon=50.0,
                 momentum=0.8, seed=None, reg=0, converge=1e-10,
                 max_rating=None, min_rating=None, delta=0.5, nboost=10):
		
		super(PMF, self).__init__()
		
		self.n_user = n_user
		self.n_item = n_item
		self.n_feature = n_feature
		
		self.random_state = RandomState(seed)
		
		# master learner -- initialize to zero function
		self.user_features_m = np.zeros((self.n_user, self.n_feature))
		self.item_features_m = np.zeros((self.n_item, self.n_feature))
		self.a_m = np.zeros(self.n_user)
		self.b_m = np.zeros(self.n_item)
		
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

		
		self.delta = delta
		self.nboost = nboost
		self.alpha = {}
		self.models = {}
			
		
	def edge(self, data):
	
		predicted = self.predict_bias(data)
		residuals = data[:, 2]
		print(predicted[residuals < 0])
		print(residuals.min())
		print(residuals.max())
		
		#edge = np.multiply(predicted, residuals).sum()
		edge = np.multiply(predicted, residuals)
		
		edge_user = np.zeros((self.n_user, 3))
		edge_user[:, 1] = 10 ** (-8)
		edge_user[:, 2] = 10 ** (-8)
		
		for i in np.arange(data.shape[0]):
			user = int(data[i, 0])
			edge_user[user, 0] += edge[i]
			if np.abs(predicted[i]) > edge_user[user, 1]:
				edge_user[user, 1] = np.abs(predicted[i])
			edge_user[user, 2] += np.abs(residuals[i])
		
		#return edge / (np.sum(np.abs(residuals)) * np.max(predicted))
		return edge_user[:, 0] / (edge_user[:, 2] * edge_user[:, 1])
		
	def compute_alpha(self, data):
	
		s = self.s
	
		e = self.edge(data)
		
		potential = data[:, 4]
		dpotential = data[:, 3]
		
		#alpha = s * np.sum(potential) + 2 * s * data.shape[0] + e * np.sum(np.abs(dpotential))
		#alpha = alpha / (s * np.sum(potential) + 2 * s * data.shape[0] - e * np.sum(np.abs(dpotential)))
		
		alpha = np.zeros(self.n_user)
		beta = np.zeros(self.n_user)
		for i in np.arange(data.shape[0]):
			user = int(data[i, 0])
			alpha[user] += s * potential[i] + e[user] * np.abs(dpotential[i]) + 2 * s
			beta[user] += s * potential[i] - e[user] * np.abs(dpotential[i]) + 2 * s
			
		return 1 / (2 * s) * np.log(alpha / beta)
			
		
	def update_residuals(self, data):
		return data[:, 2]  - self.predict_master(data) 
		
	def iter_boosting(self, train, niter):
	
		i = 0
		
		self.s = 0.7
		#np.log(train.shape[0]) / self.delta * 0.01
		s = self.s
		print(s)
		
		residuals = train[:, 2]
		residuals_max = np.inf
		
		self.mean_rating2 = np.mean(train[:, 2])
	
		while (i < self.nboost) & (residuals_max > self.delta):
		
			i += 1
			
			train_demean = np.zeros((train.shape[0], train.shape[1] + 1))
			train_demean[:, :2] = train[:, :2]
			if i == 1: 
				train_demean[:, 2] = residuals 
			else:
				train_demean[:, 2] = s * np.exp(s * residuals) - s * np.exp(-s * residuals)
				
				# add change in potential
				train_demean[:, 3] = - s* np.exp(s * residuals) + s * np.exp(-s * residuals)
			
				# add potential
				train_demean[:, 4] = np.exp(s * residuals) + np.exp(-s * residuals) - 2
		
			# initialize learner
			self.user_features_ = 0.1 * self.random_state.rand(self.n_user, self.n_feature)
			self.item_features_ = 0.1 * self.random_state.rand(self.n_item, self.n_feature)
			self.a = 0.1 * self.random_state.rand(self.n_user)
			self.b = 0.1 * self.random_state.rand(self.n_item)
		
			# base learner on residuals
			self.fit_bias(train_demean, n_iters=niter, weight=False)
			
			# compute edge
			if i == 1: 
				self.alpha[1] = 1
			else:
				self.alpha[i] = self.compute_alpha(train_demean)
	
			# update master
			self.models[i] = {}					
			self.models[i]['u'] = self.user_features_
			self.models[i]['v'] = self.item_features_
			self.models[i]['a'] = self.a
			self.models[i]['b'] = self.b
			
			# update residuals
			residuals = self.update_residuals(train)
			print(residuals.min())
			print(residuals.max())
			residuals_max = np.max(np.abs(residuals))
			
						
	def predict_master(self, data):
	
		pred = np.zeros(data.shape[0])
		A = np.zeros(self.a.shape[0])
		
		for key in self.models.keys():
			model = self.models[key]
			alpha = self.alpha[key]	
			
			u_features = model['u'].take(data.take(0, axis=1).astype('int32'), axis=0)
			i_features = model['v'].take(data.take(1, axis=1).astype('int32'), axis=0)
			a_features = model['a'].take(data.take(0, axis=1).astype('int32'), axis=0)
			b_features = model['b'].take(data.take(1, axis=1).astype('int32'), axis=0)

			if key > 1:
				alpha_data = alpha.take(data.take(0, axis=1).astype('int32'), axis=0)	
				pred = pred +  alpha_data * (a_features + b_features + np.sum(u_features * i_features, 1))
			
			else: 
				pred = pred +  a_features + b_features + np.sum(u_features * i_features, 1)
			
		
		pred[pred < 0] = 0	
		pred[pred > 1.0] = 1.0
		print(pred[data[:, 2] == 0])
		return pred 
		

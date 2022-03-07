import random
import numpy as np
import tensorflow as tf

class ResNet18:
	def __init__(self, inp, keep_prob=1):
		
		self.keep_prob = keep_prob
		self.cross_entropy = []
		self.build(inp)

	def build(self, inp, rotate=False):
		# To build the model for original ResNet18
		self.inp = inp
		self.build_conv_layers(rotate=rotate)
		self.build_fc_layers(rotate=rotate)

		# lists
		self.var_list = []
		self.hidden_list = [self.conv0_1, self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2,
							self.conv3_1, self.conv3_2, self.conv4_1,
							self.conv4_2, self.fc]
		self.input_list = [self.x, self.pool0_1, self.relu1_1, self.res_1, self.relu2_1,
							self.res_2, self.relu3_1, self.res_3, self.relu4_1, self.avgpoool]

	def build_conv_layers(self, rotate=False):
		# 7x7 Conv
		self.conv0_1 = self._conv_layer_7(self.x, "conv0_1", 64, rotate=rotate)
		self.bn0_1 = tf.nn.batch_normalization(self.conv0_1)
		self.pool0_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(self.bn0_1)
		# Conv1
		self.conv1_1 = self._conv_layer(self.pool0_1, "conv1_1", 64, rotate=rotate)
		self.bn1_1 = tf.nn.batch_normalization(self.conv1_1)
		self.relu1_1 = tf.nn.relu(self.bn1_1)
		self.conv1_2 = self._conv_layer(self.relu1_1, "conv1_2", 64, rotate=rotate)
		self.bn1_2 = tf.nn.batch_normalization(self.conv1_2)
		self.res_1 = tf.nn.relu(tf.keras.layers.add([self.bn1_2, self.pool0_1]))
		# Conv2
		self.conv2_1 = self._conv_layer(self.res_1, "conv2_1", 128, rotate=rotate)
		self.bn2_1 = tf.nn.batch_normalization(self.conv2_1)
		self.relu2_1 = tf.nn.relu(self.bn2_1)
		self.conv2_2 = self._conv_layer(self.relu2_1, "conv2_2", 128, rotate=rotate)
		self.bn2_2 = tf.nn.batch_normalization(self.conv2_2)
		self.res_2 = tf.nn.relu(tf.keras.layers.add([self.bn2_2, self.res_1]))
		# Conv3
		self.conv3_1 = self._conv_layer(self.res_2, "conv3_1", 256, rotate=rotate)
		self.bn3_1 = tf.nn.batch_normalization(self.conv3_1)
		self.relu3_1 = tf.nn.relu(self.bn3_1)
		self.conv3_2 = self._conv_layer(self.relu3_1, "conv3_2", 256, rotate=rotate)
		self.bn3_2 = tf.nn.batch_normalization(self.conv3_2)
		self.res_3 = tf.nn.relu(tf.keras.layers.add([self.bn3_2, self.res_2]))
		# Conv4
		self.conv4_1 = self._conv_layer(self.res_3, "conv4_1", 512, rotate=rotate)
		self.bn4_1 = tf.nn.batch_normalization(self.conv4_1)
		self.relu4_1 = tf.nn.relu(self.bn4_1)
		self.conv4_2 = self._conv_layer(self.relu4_1, "conv4_2", 512, rotate=rotate)
		self.bn4_2 = tf.nn.batch_normalization(self.conv4_2)
		self.res_4 = tf.nn.relu(tf.keras.layers.add([self.bn4_2, self.res_3]))

	def build_fc_layers(self, rotate=False):
		self.avgpool = tf.keras.layers.GlobalAveragePooling2D(self.res_4)
		self.fc = self._fc_layer(self.avgpoool, 10, 'fc', rotate=rotate)
		self.y = self.fc

	def _get_conv_filter(self, name, depth_in, depth_out, trainable=True):
		initializer = tf.contrib.layers.xavier_initializer()            
		return tf.get_variable(name+'_weights', shape=[3, 3, depth_in, depth_out], initializer=initializer,
								trainable=trainable, regularizer=tf.nn.l2_loss,
								collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])

	def _get_conv_filter_new(self, name, depth_in, depth_out, trainable=True):
		initializer = tf.contrib.layers.xavier_initializer()            
		return tf.get_variable(name+'_weights', shape=[1, 1, depth_in, depth_out], initializer=initializer,
								trainable=trainable, regularizer=tf.nn.l2_loss,
								collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])

	def _get_conv_bias(self, name, depth):
		initializer = tf.constant_initializer(0.0)            
		return tf.get_variable(name+'_biases', shape=[depth], initializer=initializer, trainable=True,
								collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
		
	def _conv_layer(self, bottom, name, depth, rotate=False):
		depth_in = bottom.get_shape().as_list()[-1]
		if rotate:
			with tf.variable_scope(name + "A"):
				filt = self._get_conv_filter_new(name + "A", depth_in, depth_in, trainable=False)
				conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
			with tf.variable_scope(name + "B"):
				filt = self._get_conv_filter(name + "B", depth_in, depth, trainable=True)
				conv = tf.nn.conv2d(conv, filt, [1, 1, 1, 1], padding='SAME')
			with tf.variable_scope(name + "C"):
				filt = self._get_conv_filter_new(name + "C", depth, depth, trainable=False)
				conv_out = tf.nn.conv2d(conv, filt, [1, 1, 1, 1], padding='VALID')
				conv_biases = self._get_conv_bias(name, depth)
				bias = tf.nn.bias_add(conv_out, conv_biases)
		else:
			with tf.variable_scope(name):
				filt = self._get_conv_filter(name, depth_in, depth)
				conv = tf.nn.conv2d(bottom, filt, [1,1,1,1], padding='SAME')
				conv_biases = self._get_conv_bias(name, depth)
				bias = tf.nn.bias_add(conv, conv_biases)
		return bias

	def _get_fc_weights(self, bottom, dim_in, dim_out, name, trainable=True):
		initializer = tf.truncated_normal([dim_in, dim_out], .0, .001)
		return tf.get_variable(name+'_weights', initializer=initializer, trainable=trainable,
							   regularizer=tf.nn.l2_loss,
							   collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])

	def _get_fc_bias(self, dim_out, name, trainable=True):
		initializer = tf.random_normal([dim_out], stddev=0.1)
		return tf.get_variable(name+'_bias', initializer=initializer, trainable=trainable,
							   collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])

	def _fc_layer(self, bottom, dim_out, name, trainable=True, rotate=False):
		shape_bottom = bottom.get_shape().as_list()[1:]
		dim_in = np.prod(shape_bottom)
		if rotate:
			with tf.variable_scope(name + "A"):
				weights = self._get_fc_weights(bottom, dim_in, dim_in, name + "A", trainable=False)
				fc = tf.matmul(bottom, weights)
			with tf.variable_scope(name + "B"):
				weights = self._get_fc_weights(bottom, dim_in, dim_out, name + "B", trainable=True)
				fc = tf.matmul(fc, weights)
			with tf.variable_scope(name + "C"):
				weights = self._get_fc_weights(bottom, dim_out, dim_out, name + "C", trainable=False)
				fc_out = tf.matmul(fc, weights)
				biases = self._get_fc_bias(dim_out, name)
				fc_bias = tf.nn.bias_add(fc_out, biases)
		else:
			with tf.variable_scope(name):
				weights = self._get_fc_weights(bottom, dim_in,dim_out, name,trainable)
				biases = self._get_fc_bias(dim_out, name)
				fc_bias = tf.nn.bias_add(tf.matmul(bottom, weights), biases)
		return fc_bias

	def _get_conv_filter_7(self, name, depth_in, depth_out, trainable=True):
		initializer = tf.contrib.layers.xavier_initializer()            
		return tf.get_variable(name+'_weights', shape=[7, 7, depth_in, depth_out], initializer=initializer,
								trainable=trainable, regularizer=tf.nn.l2_loss,
								collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
	
	def _get_conv_filter_new_7(self, name, depth_in, depth_out, trainable=True):
		initializer = tf.contrib.layers.xavier_initializer()            
		return tf.get_variable(name+'_weights', shape=[1, 1, depth_in, depth_out], initializer=initializer,
								trainable=trainable, regularizer=tf.nn.l2_loss,
								collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
	
	def _get_conv_bias_7(self, name, depth):
		initializer = tf.constant_initializer(0.0)            
		return tf.get_variable(name+'_biases', shape=[depth], initializer=initializer, trainable=True,
								collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])

	def _conv_layer_7(self, bottom, name, depth, rotate=False):
		depth_in = bottom.get_shape().as_list()[-1]
		if rotate:
			with tf.variable_scope(name + "A"):
				filt = self._get_conv_filter_new_7(name + "A", depth_in, depth_in, trainable=False)
				conv = tf.nn.conv2d(bottom, filt, [2, 2, 2, 2], padding='VALID')
			with tf.variable_scope(name + "B"):
				filt = self._get_conv_filter_7(name + "B", depth_in, depth, trainable=True)
				conv = tf.nn.conv2d(conv, filt, [2, 2, 2, 2], padding='SAME')
			with tf.variable_scope(name + "C"):
				filt = self._get_conv_filter_new_7(name + "C", depth, depth, trainable=False)
				conv_out = tf.nn.conv2d(conv, filt, [2, 2, 2, 2], padding='VALID')
				conv_biases = self._get_conv_bias_7(name, depth)
				bias = tf.nn.bias_add(conv_out, conv_biases)
		else:
			with tf.variable_scope(name):
				filt = self._get_conv_filter_7(name, depth_in, depth)
				conv = tf.nn.conv2d(bottom, filt, [2,2,2,2], padding='SAME')
				conv_biases = self._get_conv_bias_7(name, depth)
				bias = tf.nn.bias_add(conv, conv_biases)
		return bias

	def compute_fisher(self, data_img, data_lbl, sess, num_samples=200, eq_distrib=True):
		# initialize Fisher information for most recent task
		self.F_accum = []
		for v in range(len(self.var_list)):
			self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

		# sampling a random class from softmax
		probs = tf.nn.softmax(self.y)
		class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

		classes = np.unique(data_lbl)
		if eq_distrib:
			# equally distributed among classes samples
			indx = []
			for cl in range(len(classes)):
				tmp = np.where(data_lbl == classes[cl])[0]
				np.random.shuffle(tmp)
				indx = np.hstack((indx, tmp[0:min(num_samples, len(tmp))]))
				indx = np.asarray(indx).astype(int)
		else:
			# random non-repeating selected images
			indx = random.sample(range(0, data_img.shape[0]), num_samples * len(classes))
		
		for i in range(len(indx)):
			# select random input image
			im_ind = indx[i]

			# compute first-order derivatives
			tmp = tf.gradients(tf.log(probs[0,class_ind]), self.var_list)
			for v in range(len(tmp)):
				if tmp[v] == None:
					tmp[v] = tf.zeros([1]) 
			ders = sess.run( tmp , feed_dict={ self.x: data_img[im_ind:im_ind+1]})
			# square the derivatives and add to total
			for v in range(len(self.F_accum)):
				self.F_accum[v] += np.square(ders[v])

		# divide totals by number of samples
		for v in range(len(self.F_accum)):
			self.F_accum[v] /= len(indx)

	def compute_M_L(self, imgset,lbl, sess, num_samples=200, eq_distrib=True):
		# computer Fisher information for each parameter
		# initialize Fisher information for most recent task
		self.M_mat = []
		for v in range(len(self.hidden_list)):
			self.M_mat.append(np.zeros((self.hidden_list[v].get_shape()[-1],self.hidden_list[v].get_shape()[-1])))

		self.L_mat = []
		for v in range(len(self.input_list)):
			self.L_mat.append(np.zeros((self.input_list[v].get_shape()[-1],self.input_list[v].get_shape()[-1])))

		# sampling a random class from softmax
		probs = tf.nn.softmax(self.y)
		class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
		classes = np.unique(lbl)
		if eq_distrib:
			# random non-repeating selected images
			indx = random.sample(range(0,imgset.shape[0]),num_samples*len(classes))
		else:
			# equally distributed among classes samples
			indx = []
			for cl in range(len(classes)):
				tmp = np.where(lbl == classes[cl])[0]
				np.random.shuffle(tmp)
				indx = np.hstack((indx,tmp[0:min(num_samples, len(tmp))]))
				indx = np.asarray(indx).astype(int)

		for i in range(len(indx)):
			# select random input image
			im_ind = indx[i]
			# compute first-order derivatives
			ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.hidden_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
			for v in range(len(self.M_mat)):
				# check which type of layer is it
				if len(ders[v].shape)==2:
					# it's a fully-connected layer
					self.M_mat[v] += np.dot(ders[v].T,ders[v])
				else:
					# it's a convolutional layer
					x = ders[v]
					for w in range(x.shape[1]):
						for h in range(x.shape[2]):
							self.M_mat[v] += np.dot(x[:,w,h,:].T,x[:,w,h,:])
			
			# square the derivatives and add to total
			inp_vec = sess.run(self.input_list,feed_dict={self.x: imgset[im_ind:im_ind+1]})
			for v in range(len(self.L_mat)):
				# check which type of layer is it
				if len(inp_vec[v].shape)==2:
					# it's a fully-connected layer
					self.L_mat[v] += np.dot(inp_vec[v].T,inp_vec[v])
				else:
					# it's a convolutional layer
					x = inp_vec[v]
					for w in range(x.shape[1]):
						for h in range(x.shape[2]):
							self.L_mat[v] += np.dot(x[:,w,h,:].T,x[:,w,h,:])
				
		# divide totals by number of samples
		for v in range(len(self.M_mat)):
			self.M_mat[v] /= len(indx)
		for v in range(len(self.L_mat)):
			self.L_mat[v] /= len(indx)
 
	def star(self):
		# used for saving optimal weights after most recent task training
		self.star_vars = []

		for v in range(len(self.var_list)):
			self.star_vars.append(self.var_list[v].eval())

	def restore(self, sess):
		# reassign optimal weights for latest task
		if hasattr(self, "star_vars"):
			for v in range(len(self.var_list)):
				sess.run(self.var_list[v].assign(self.star_vars[v]))

	def set_vanilla_loss(self,lr,wght_decay,train_list):
		self.loss = self.cross_entropy  
		self.ewc  = tf.constant(0.0)
		l2_reg        = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		self.loss += l2_reg
		conv_list = [v for v in train_list if "conv"  in v.name]
		fc_list = [v for v in train_list if "fc"  in v.name]
		train_op1 = tf.train.AdamOptimizer(lr*0.01).minimize(self.loss,var_list=conv_list)
		train_op2 = tf.train.AdamOptimizer(lr).minimize(self.loss,var_list=fc_list)
		self.train_step = tf.group(train_op1, train_op2)
	
	def update_ewc_loss(self,lr,lam,wght_decay,train_list,num_nodes):
		# elastic weight consolidation
		# lam is weighting for previous task(s) constraints
		self.loss = self.cross_entropy
		self.ewc = 0.0
		for v in range(len(self.var_list)):
			 # EWC
			 if v == len(self.var_list)-2:
				  self.ewc += lam*0.5 * tf.reduce_sum(tf.multiply(self.F_accum[v][:,:num_nodes].astype(np.float32),tf.square(self.var_list[v][:,:num_nodes] - self.star_vars[v][:,:num_nodes])))
			 elif v== len(self.var_list)-1:
				  self.ewc += lam*0.5 * tf.reduce_sum(tf.multiply(self.F_accum[v][:num_nodes].astype(np.float32),tf.square(self.var_list[v][:num_nodes] - self.star_vars[v][:num_nodes])))
			 else:
				  self.ewc += lam*0.5 * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
				  
		l2_reg        = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		self.loss += l2_reg
		self.loss += self.ewc

		conv_list = [v for v in train_list if "conv"  in v.name]
		fc_list = [v for v in train_list if "fc"  in v.name]
		train_op1 = tf.train.AdamOptimizer(lr*0.01).minimize(self.loss,var_list=conv_list)
		train_op2 = tf.train.AdamOptimizer(lr).minimize(self.loss,var_list=fc_list)

		self.train_step = tf.group(train_op1, train_op2)
		

	def compute_svd(self, sess, num_task):
		# Compute svd of REWC
		self.decom_list = [v for v in self.var_list if "weights" in v.name]
		self.weights_svd = []
		if num_task < 1:
			 self.weights_w = []
			 for v in range(len(self.decom_list)):
				  self.weights_w.append(sess.run(self.decom_list[v]))
		else:  
			# Combine the filters firstly, and then do decomposition again
			tmp = []
			for v in range(len(self.decom_list)):
				tmp.append(sess.run(self.decom_list[v]))
			self.weights_w = []

			for v in range(len(self.M_mat)):
				# check which type of variable it is -- if 1 then bias
				if len(tmp[v*3+1].shape)==2:
					# it's a fully-connected layer
					W_rec = np.dot(np.dot(tmp[v*3], tmp[v*3+1]), tmp[v*3+2])
				else:
					if len(tmp[v*3+1].shape)==4:
						# it's a convolutional layer
						W_rec = np.zeros(tmp[v*3+1].shape)
						for w in range(tmp[v*3+1].shape[0]):
							for h in range(tmp[v*3+1].shape[1]):
								W_rec[w,h,:,:] = np.dot(np.dot(tmp[v*3][0,0,:,:],tmp[v*3+1][w,h,:,:]),tmp[v*3+2][0,0,:,:])
				  
				self.weights_w.append(W_rec)
		for v in range(len(self.M_mat)):
			# Apply SVD
			U2, _, _ = np.linalg.svd(self.M_mat[v], full_matrices=False)
			U1, _, _ = np.linalg.svd(self.L_mat[v], full_matrices=False)
			# Calculate rotations
			Q1 = np.dot(np.linalg.inv(np.dot(U1.T,U1)),U1.T)
			Q2 = np.dot(U2.T,np.linalg.inv(np.dot(U2,U2.T)))
			# check which type of variable it is -- if 1 then bias
			if len(self.weights_w[v].shape)==2:
				# it's a fully-connected layer
				Wp = np.dot(np.dot(Q1,self.weights_w[v]),Q2)
				self.weights_svd.append(U1)
				self.weights_svd.append(Wp)
				self.weights_svd.append(U2)
				#print np.allclose(np.dot(np.dot(U1,Wp),U2),self.weights_w[v])
			else:
				if len(self.weights_w[v].shape)==4:
					# it's a convolutional layer
					W = self.weights_w[v]
					Wp = np.zeros(W.shape)
					# fix the output dimension
					for w in range(W.shape[0]):
						for h in range(W.shape[1]):
							for o in range(W.shape[3]):
								Wp[w,h,:,o] = np.dot(Q1,W[w,h,:,o])
					# fix the input dimension
					for w in range(W.shape[0]):
						for h in range(W.shape[1]):
							for i in range(W.shape[2]):
								Wp[w,h,i,:] = np.dot(Wp[w,h,i,:],Q2)
					
					self.weights_svd.append(np.expand_dims(np.expand_dims(U1,axis=0),axis=0))
					self.weights_svd.append(Wp)
					self.weights_svd.append(np.expand_dims(np.expand_dims(U2,axis=0),axis=0))
					
					# check by fixing spatial dimensions
					W_check = np.zeros(W.shape)
					for w in range(W.shape[0]):
						for h in range(W.shape[1]):
							W_check[w,h,:,:] = np.dot(np.dot(U1,Wp[w,h,:,:]),U2)
					#print np.allclose(W_check,self.weights_w[v])

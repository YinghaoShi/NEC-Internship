import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
import pdb
import random

#the example call I used: python mainREWC.py --n_tasks 10 --lr 0.001 --data_path data --save_path results  --samples_per_task 1000 --cuda yes --seed 0 --data_file fashion_mnist_permutations_reduced.pt --meta_lambda 1000 --num_samples_per_class 1

class MLP:
    def __init__(self, x, y_, doDecom = [True, True, False]):
        self.build(x, y_)
        self.doDecom = doDecom
        self.factor = 2
    def build(self, x, y_):
          in_dim = int(x.get_shape()[1])
          out_dim = int(y_.get_shape()[1])
          self.x = x

          # Hyperparameters
          mu = 0
          sigma = 0.1

		#out = x  
		#with tf.name_scope("MLP"):  
		#	for i in range(1, len(self.sizes)):  
		#		if i < (len(self.sizes) - 1):  
		#			out = tcl.fully_connected(out, self.sizes[i], activation_fn = tf.nn.relu, \
		#				reuse=tf.AUTO_REUSE, scope = 'fully_connected'+'_'+str(i))
        #        else: 
		#			out = tcl.fully_connected(out, int(self.sizes[i]), activation_fn=None, \
		#				reuse=tf.AUTO_REUSE, scope = 'fully_connected'+'_'+str(i))

          # Layer 3: Fully Connected. Input = in_dim. Output = 100.
          self.fc1_w = tf.Variable(tf.random.truncated_normal(shape = (in_dim,100), mean = mu, stddev = sigma),name = 'fc1_w')
          self.fc1_b = tf.Variable(tf.zeros(100),name = 'fc1_b')
          self.fc1 = tf.matmul(self.x,self.fc1_w) + self.fc1_b
          self.relu3 = tf.nn.relu(self.fc1)
          
          # Layer 4: Fully Connected. Input = 120. Output = 100.
          self.fc2_w = tf.Variable(tf.random.truncated_normal(shape = (100,100), mean = mu, stddev = sigma),name = 'fc2_w')
          self.fc2_b = tf.Variable(tf.zeros(100),name = 'fc2_b')
          self.fc2 = tf.matmul(self.relu3,self.fc2_w) + self.fc2_b
          self.relu4 = tf.nn.relu(self.fc2)
          
          # Layer 5: Fully Connected. Input = 100. Output = number of features.
          self.fc3_w = tf.Variable(tf.random.truncated_normal(shape = (100,out_dim), mean = mu , stddev = sigma),name = 'fc3_w')
          self.fc3_b = tf.Variable(tf.zeros(out_dim),name = 'fc3_b')
          self.y = tf.matmul(self.relu4, self.fc3_w) + self.fc3_b

          # lists
          self.var_list = [self.fc1_w,   self.fc1_b,   self.fc2_w,   self.fc2_b,
                           self.fc3_w,   self.fc3_b]
          self.hidden_list = [self.fc1, self.fc2, self.y]
          self.input_list = [self.x, self.relu3, self.relu4]

          
          self.b_list = [self.fc1_b,  self.fc2_b, self.fc3_b]
          self.b_val_list = []
          # vanilla single-task loss
          one_hot_targets = y_
          self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_targets , logits=self.y))

          # performance metrics
          correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
          self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def rebuild_decom(self, x, y_,t):
        
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
        self.x = x

        self.var_list = []
        pos = 0

        # Hyperparameters
        mu = 0
        sigma = 0.1
        
        
        # Layer 3: Fully Connected. Input = in_dim. Output = 100. 
        if self.doDecom[0]:
            self.fc1_w1 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos])), trainable = False)
            self.fc1_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.fc1_w3 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+2])), trainable = False)
            #print("fc1_w3   ",self.fc1_w3.shape)
            #if self.factor==2:
            #    self.fc1_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[1])))
            #else:
            #    self.fc1_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[0])))
            self.fc1_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[0])))
            #print("fc1_b    ",self.fc1_b.shape)
            self.fc1 = tf.matmul(tf.matmul(tf.matmul(self.x,self.fc1_w1),self.fc1_w2),self.fc1_w3) + self.fc1_b
            #print("fc1  ",self.fc1.shape)
            self.relu3 = tf.nn.relu(self.fc1)
            self.var_list.append(self.fc1_w1)
            self.var_list.append(self.fc1_w2)            
            self.var_list.append(self.fc1_w3)
            #self.var_list.append(tf.matmul(tf.matmul(self.fc1_w1,self.fc1_w2),self.fc1_w3))
            self.var_list.append(self.fc1_b)
            pos = pos + 3
        else: 
            self.fc1_w = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos])))
            if self.factor==2:
                self.fc1_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[1])))
            else:
                self.fc1_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[0])))
            self.fc1 = tf.matmul(self.fla,self.fc1_w) + self.fc1_b
            self.relu3 = tf.nn.relu(self.fc1)
            self.var_list.append(self.fc1_w)
            self.var_list.append(self.fc1_b)
        #    pos = pos + 1
            
        # Layer 4: Fully Connected. Input = 120. Output = 100.
        if self.doDecom[1]:
            #pdb.set_trace()
            self.fc2_w1 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos])), trainable = False)
            self.fc2_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.fc2_w3 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+2])), trainable = False)
            #print("fc2_w3   ",self.fc2_w3.shape)
            #if self.factor==2:
            #    self.fc2_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[3])))
            #else:
            #    self.fc2_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[1])))
            self.fc2_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[1])))
            #print("fc2_b    ",self.fc2_b.shape)
            self.fc2 = tf.matmul(tf.matmul(tf.matmul(self.relu3,self.fc2_w1),self.fc2_w2),self.fc2_w3) + self.fc2_b
            #print("fc2  ",self.fc2.shape)
            self.relu4 = tf.nn.relu(self.fc2)
            self.var_list.append(self.fc2_w1)
            self.var_list.append(self.fc2_w2)
            self.var_list.append(self.fc2_w3)            
            #self.var_list.append(tf.matmul(tf.matmul(self.fc2_w1,self.fc2_w2),self.fc2_w3))
            self.var_list.append(self.fc2_b)
            pos = pos + 3
        else:
            self.fc2_w = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos])))
            if self.factor==2:
                self.fc2_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[3])))
            else:
                self.fc2_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[1])))
            self.fc2 = tf.matmul(self.relu3,self.fc2_w) + self.fc2_b
            self.relu4 = tf.nn.relu(self.fc2)
            self.var_list.append(self.fc2_w)
            self.var_list.append(self.fc2_b)
            pos = pos + 1
        
        # Layer 5: Fully Connected. Input = 100. Output = number of features.
        #pdb.set_trace()
        if self.doDecom[2]:
            self.fc3_w1 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos])), trainable = False)
            self.fc3_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.fc3_w3 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+2])), trainable = False)
            #print("fc3_w3   ",self.fc3_w3.shape)
            #if self.factor==2:
            #    self.fc3_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[5])))
            #else:
            #    self.fc3_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[2])))
            #print("fc3_b    ",self.fc3_b.shape)
            """ self.fc3_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[2]))) """
            self.y = tf.matmul(tf.matmul(tf.matmul(self.relu4,self.fc3_w1),self.fc3_w2),self.fc3_w3) + self.fc3_b
            #print("y    ",self.y.shape)
            self.var_list.append(self.fc3_w1)
            self.var_list.append(self.fc3_w2)
            self.var_list.append(self.fc3_w3)
            self.var_list.append(self.fc3_b)
            pos = pos + 3
        else:
            self.fc3_w = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos])))
            #if self.factor==2:
            #    self.fc3_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[5])))
            #else:
            #    self.fc3_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[2])))
            self.fc3_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[2])))    
            self.y = tf.matmul(self.relu4, self.fc3_w) + self.fc3_b
            self.var_list.append(self.fc3_w)
            self.var_list.append(self.fc3_b)
            pos = pos + 1

        # lists
        self.hidden_list = [self.fc1, self.fc2, self.y]
        self.input_list = [self.x, self.relu3, self.relu4]
        
        self.b_list = [self.fc1_b,  self.fc2_b, self.fc3_b]        

        # vanilla single-task loss
        #scores = self.y
        #new_cl = range(0,10)
        #label_new_classes =  tf.stack([y_[:,i] for i in new_cl],axis=1) 
        #pred_new_classes = tf.stack([scores[:,i] for i in new_cl],axis=1)

        #self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes,
        #                                                                            logits=pred_new_classes))
    
        one_hot_targets = y_
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_targets , logits=self.y))
        
        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.global_step = tf.compat.v1.train.get_or_create_global_step()


    def compute_fisher(self, data, sess, num_samples=200, eq_distrib=True):
        # computer Fisher information for each parameter
        
        imgset = data.images
        
        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.math.log(probs), 1)[0][0])

        classes = np.unique(data.labels)
        if eq_distrib:
            # equally distributed among classes samples
            indx = []
            for cl in range(len(classes)):
                tmp = np.where(data.labels == classes[cl])[0]
                np.random.shuffle(tmp)
                indx = np.hstack((indx,tmp[0:min(num_samples, len(tmp))]))
                indx = np.asarray(indx).astype(int)
        else:
            # random non-repeating selected images
            indx = random.sample(xrange(0,imgset.shape[0]),num_samples*len(classes))
            
        for i in range(len(indx)):
            # select random input image
            im_ind = indx[i]
            
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.math.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= len(indx)

    def compute_M_L(self, data, sess, num_samples=200, eq_distrib=True):
        # computer Fisher information for each parameter
        
        imgset = data.images
        
        # initialize Fisher information for most recent task
        self.M_mat = []
        for v in range(len(self.hidden_list)):
            self.M_mat.append(np.zeros((self.hidden_list[v].get_shape()[-1],self.hidden_list[v].get_shape()[-1])))

        self.L_mat = []
        for v in range(len(self.input_list)):
            self.L_mat.append(np.zeros((self.input_list[v].get_shape()[-1],self.input_list[v].get_shape()[-1])))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.math.log(probs), 1)[0][0])

        classes = np.unique(data.labels)
        if eq_distrib:
            # equally distributed among classes samples
            indx = []
            for cl in range(len(classes)):
                tmp = np.where(data.labels == classes[cl])[0]
                np.random.shuffle(tmp)
                indx = np.hstack((indx,tmp[0:min(num_samples, len(tmp))]))
                indx = np.asarray(indx).astype(int)
        else:
            # random non-repeating selected images
            indx = random.sample(xrange(0,imgset.shape[0]),num_samples*len(classes))
        
        for i in range(len(indx)):
            # select random input image
            im_ind = indx[i]
            
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.math.log(probs[0,class_ind]), self.hidden_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
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

    def set_vanilla_loss(self,  lr):
        self.loss = self.cross_entropy
        self.ewc  = tf.constant(0.0)
        self.train_step = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)
    
    def update_ewc_loss(self, lr, lam, num_nodes=10):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints
        self.loss = self.cross_entropy
        self.ewc = tf.constant(0.0)
        for v in range(len(self.var_list)):
            if v == len(self.var_list)-2:
                self.ewc += lam*0.5 * tf.reduce_sum(tf.multiply(self.F_accum[v][:,:num_nodes].astype(np.float32),tf.square(self.var_list[v][:,:num_nodes] - self.star_vars[v][:,:num_nodes])))
            elif v== len(self.var_list)-1:
                self.ewc += lam*0.5 * tf.reduce_sum(tf.multiply(self.F_accum[v][:num_nodes].astype(np.float32),tf.square(self.var_list[v][:num_nodes] - self.star_vars[v][:num_nodes])))
            else:
                self.ewc += lam*0.5 * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        self.loss += self.ewc
        self.train_step = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def compute_svd(self, sess,t):
        # Compute svd of REWC
        self.b_val_list = []
        for v in range(len(self.b_list)):
            self.b_val_list.append(sess.run(self.b_list[v]))

        self.decom_list = [v for v in self.var_list]

        
        self.weights_svd = []
        self.weights_w = []
        for v in range(len(self.var_list)):
            self.weights_w.append(sess.run(self.var_list[v]))
        if t<1:
            for v in range(len(self.M_mat)):
                if not self.doDecom[v]:
                    # Not use decomposition
                    self.weights_svd.append(self.weights_w[v*2])
                else:
                    # Apply decomposition
                    U2, _, _ = np.linalg.svd(self.M_mat[v], full_matrices=False)
                    U1, _, _ = np.linalg.svd(self.L_mat[v], full_matrices=False)
                    # Calculate rotations
                    Q1 = np.dot(np.linalg.inv(np.dot(U1.T,U1)),U1.T)
                    Q2 = np.dot(U2.T,np.linalg.inv(np.dot(U2,U2.T)))
                    # check which type of variable it is -- if 1 then bias
                    if len(self.weights_w[v*2].shape)==2:
                        # it's a fully-connected layer
                        Wp = np.dot(np.dot(Q1,self.weights_w[v*2]),Q2)
                        self.weights_svd.append(U1)
                        self.weights_svd.append(Wp)
                        self.weights_svd.append(U2)
        else:
            for v in range(len(self.M_mat)):
                if not self.doDecom[v]:
                    # Not use decomposition
                    self.weights_svd.append(self.weights_w[v*4])
                else:
                    # Apply decomposition
                    U2, _, _ = np.linalg.svd(self.M_mat[v], full_matrices=False)
                    U1, _, _ = np.linalg.svd(self.L_mat[v], full_matrices=False)
                    # Calculate rotations
                    Q1 = np.dot(np.linalg.inv(np.dot(U1.T,U1)),U1.T)
                    Q2 = np.dot(U2.T,np.linalg.inv(np.dot(U2,U2.T)))
                    # check which type of variable it is -- if 1 then bias
                    if len(self.weights_w[v*4].shape)==2:
                        # it's a fully-connected layer
                        temp = np.dot(np.dot(self.weights_w[v*4],self.weights_w[v*4+1]),self.weights_w[v*4+2])
                        Wp = np.dot(np.dot(Q1,temp),Q2)
                        self.weights_svd.append(U1)
                        self.weights_svd.append(Wp)
                        self.weights_svd.append(U2)

    def compute_svd_old(self, sess,num_task):
        # Compute svd of REWC
        #self.weights_svd = []
        #self.weights_w = []
        #for v in range(len(self.var_list)):
        #    self.weights_w.append(sess.run(self.var_list[v]))

        self.b_val_list = []
        for v in range(len(self.b_list)):
            self.b_val_list.append(sess.run(self.b_list[v]))
        #self.decom_list = [v for v in self.var_list if "weights" in v.name]
        #self.decom_list = [v for v in self.var_list if "_w" in v.name]
        self.decom_list = [v for v in self.var_list]# if "_w" in v.name]
        self.weights_svd = []
        #pdb.set_trace()
        if num_task < 1:
             self.weights_w = []
             for v in range(len(self.decom_list)):
                  self.weights_w.append(sess.run(self.decom_list[v]))
             self.factor = 2
        else:  
            self.factor = 1
            # Combine the filters firstly, and then do decomposition again
            tmp = []
            for v in range(len(self.decom_list)):
                tmp.append(sess.run(self.decom_list[v]))
            self.weights_w = []

            for v in range(len(self.M_mat)):
                # check which type of variable it is -- if 1 then bias
                if len(tmp[v*4+1].shape)==2:
                    # it's a fully-connected layer
                    W_rec = np.dot(np.dot(tmp[v*4], tmp[v*4+1]), tmp[v*4+2])                
                  
                
                self.weights_w.append(W_rec)
        #pdb.set_trace()
        for v in range(len(self.M_mat)):
            if not self.doDecom[v]:
                # Not use decomposition
                self.weights_svd.append(self.weights_w[v*self.factor])
            else:
                # Apply decomposition
                U2, _, _ = np.linalg.svd(self.M_mat[v], full_matrices=False)
                U1, _, _ = np.linalg.svd(self.L_mat[v], full_matrices=False)
                # Calculate rotations
                Q1 = np.dot(np.linalg.inv(np.dot(U1.T,U1)),U1.T)
                Q2 = np.dot(U2.T,np.linalg.inv(np.dot(U2,U2.T)))
                #print(Q1.shape,Q2.shape,self.weights_w[v*self.factor].shape)                
                #print(np.dot(Q1,self.weights_w[v*self.factor]).shape)
                #print(np.dot(np.dot(Q1,self.weights_w[v*self.factor]),Q2).shape)
                #print("###############")
                # check which type of variable it is -- if 1 then bias
                if len(self.weights_w[v*self.factor].shape)==2:
                    # it's a fully-connected layer
                    Wp = np.dot(np.dot(Q1,self.weights_w[v*self.factor]),Q2)
                    self.weights_svd.append(U1)
                    self.weights_svd.append(Wp)
                    self.weights_svd.append(U2)
                
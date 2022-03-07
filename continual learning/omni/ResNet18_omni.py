import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
import pdb
import random

#the example call I used: python mainREWC.py --n_tasks 10 --lr 0.001 --data_path data --save_path results  --samples_per_task 1000 --cuda yes --seed 0 --data_file fashion_mnist_permutations_reduced.pt --meta_lambda 1000 --num_samples_per_class 1

class ResNet18:
    def __init__(self, x, y_, doDecom = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False], class_number=10, class_number_list=[], nf=20,t=0):
        self.class_number_list = class_number_list
        self.class_number = class_number
        self.nf = nf
        self.t = t
        self.build(x, y_, class_number,nf,t)
        self.doDecom = doDecom
        self.factor = 2
        
    def compute_offsets(self, outputs_per_task, task):        
        nc_per_task = np.add.accumulate(outputs_per_task)
        offset1 = nc_per_task[task] - outputs_per_task[task]
        offset2 = nc_per_task[task]
        return int(offset1), int(offset2)

    def build(self, x, y_, class_number, nf, t):
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
        self.x = x
        self.nf = nf
        # Hyperparameters
        mu = 0
        sigma = 0.1
        #print("xshape:{}".format(self.x.get_shape()))
        #print("yshape:{}".format(y_.get_shape()))

        #input layer 3*3 output nf
        self.conv1_w = tf.Variable(tf.truncated_normal(shape = [3,3,1,self.nf],mean = mu, stddev = sigma),name = 'conv1_w')
        self.conv1_b = tf.Variable(tf.zeros(self.nf),name = 'conv1_b',trainable=False)
        self.conv1 = tf.nn.conv2d(self.x,self.conv1_w, strides = [1,1,1,1], padding = 'SAME') + self.conv1_b
        self.bn1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.relu1 = tf.nn.relu(self.bn1)
	
        #first basic block conv 3*3 nf stride [1,1]
        self.conv2_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf,self.nf],mean = mu, stddev = sigma),name = 'conv2_w')
        self.conv2_b = tf.Variable(tf.zeros(self.nf),name = 'conv2_b',trainable=False)
        self.conv2 = tf.nn.conv2d(self.relu1,self.conv2_w, strides = [1,1,1,1], padding = 'SAME') + self.conv2_b
        self.bn2 = tf.keras.layers.BatchNormalization()(self.conv2)
        self.relu2 = tf.nn.relu(self.bn2)
        self.conv3_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf,self.nf],mean = mu, stddev = sigma),name = 'conv3_w')
        self.conv3_b = tf.Variable(tf.zeros(self.nf),name = 'conv3_b',trainable=False)
        self.conv3 = tf.nn.conv2d(self.relu2,self.conv3_w, strides = [1,1,1,1], padding = 'SAME') + self.conv3_b
        self.bn3 = tf.keras.layers.BatchNormalization()(self.conv3)
        self.relu3 = tf.nn.relu(self.bn3)

        self.conv4_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf,self.nf],mean = mu, stddev = sigma),name = 'conv4_w')
        self.conv4_b = tf.Variable(tf.zeros(self.nf),name = 'conv4_b',trainable=False)
        self.conv4 = tf.nn.conv2d(self.relu3,self.conv4_w, strides = [1,1,1,1], padding = 'SAME') + self.conv4_b
        self.bn4 = tf.keras.layers.BatchNormalization()(self.conv4)
        self.relu4 = tf.nn.relu(self.bn4)
        self.conv5_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf,self.nf],mean = mu, stddev = sigma),name = 'conv5_w')
        self.conv5_b = tf.Variable(tf.zeros(self.nf),name = 'conv5_b',trainable=False)
        self.conv5 = tf.nn.conv2d(self.relu4,self.conv5_w, strides = [1,1,1,1], padding = 'SAME') + self.conv5_b
        self.bn5 = tf.keras.layers.BatchNormalization()(self.conv5)
        self.relu5 = tf.nn.relu(self.bn5)

        #second basic block conv 3*3 nf*2 stride [2,1]
        self.conv6_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf,self.nf*2],mean = mu, stddev = sigma),name = 'conv6_w')
        self.conv6_b = tf.Variable(tf.zeros(self.nf*2),name = 'conv6_b',trainable=False)
        self.conv6 = tf.nn.conv2d(self.relu5,self.conv6_w, strides = [1,2,2,1], padding = 'SAME') + self.conv6_b
        self.bn6 = tf.keras.layers.BatchNormalization()(self.conv6)
        self.relu6 = tf.nn.relu(self.bn6)
        self.conv7_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*2,self.nf*2],mean = mu, stddev = sigma),name = 'conv7_w')
        self.conv7_b = tf.Variable(tf.zeros(self.nf*2),name = 'conv1_b',trainable=False)
        self.conv7 = tf.nn.conv2d(self.relu6,self.conv7_w, strides = [1,1,1,1], padding = 'SAME') + self.conv7_b
        self.bn7 = tf.keras.layers.BatchNormalization()(self.conv7)

        self.conv8_w = tf.Variable(tf.truncated_normal(shape = [1,1,self.nf,self.nf*2],mean = mu, stddev = sigma),name = 'conv8_w')
        self.conv8_b = tf.Variable(tf.zeros(self.nf*2),name = 'conv1_b',trainable=False)
        self.conv8 = tf.nn.conv2d(self.relu5,self.conv8_w, strides = [1,2,2,1], padding = 'SAME') + self.conv8_b
        self.bn8 = tf.keras.layers.BatchNormalization()(self.conv8)
        self.relu7_8 = tf.nn.relu(tf.keras.layers.add([self.bn7, self.bn8]))

        self.conv9_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*2,self.nf*2],mean = mu, stddev = sigma),name = 'conv9_w')
        self.conv9_b = tf.Variable(tf.zeros(self.nf*2),name = 'conv9_b',trainable=False)
        self.conv9 = tf.nn.conv2d(self.relu7_8,self.conv9_w, strides = [1,1,1,1], padding = 'SAME') + self.conv9_b
        self.bn9 = tf.keras.layers.BatchNormalization()(self.conv9)
        self.relu9 = tf.nn.relu(self.bn9)
        self.conv10_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*2,self.nf*2],mean = mu, stddev = sigma),name = 'conv10_w')
        self.conv10_b = tf.Variable(tf.zeros(self.nf*2),name = 'conv10_b',trainable=False)
        self.conv10 = tf.nn.conv2d(self.relu9,self.conv10_w, strides = [1,1,1,1], padding = 'SAME') + self.conv10_b
        self.bn10 = tf.keras.layers.BatchNormalization()(self.conv10)
        self.relu10 = tf.nn.relu(self.bn10)

        #third basic block conv 3*3 nf*4 stride [2,1]
        self.conv11_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*2,self.nf*4],mean = mu, stddev = sigma),name = 'conv11_w')
        self.conv11_b = tf.Variable(tf.zeros(self.nf*4),name = 'conv11_b',trainable=False)
        self.conv11 = tf.nn.conv2d(self.relu10,self.conv11_w, strides = [1,2,2,1], padding = 'SAME') + self.conv11_b
        self.bn11 = tf.keras.layers.BatchNormalization()(self.conv11)
        self.relu11 = tf.nn.relu(self.bn11)
        self.conv12_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*4,self.nf*4],mean = mu, stddev = sigma),name = 'conv12_w')
        self.conv12_b = tf.Variable(tf.zeros(self.nf*4),name = 'conv12_b',trainable=False)
        self.conv12 = tf.nn.conv2d(self.relu11,self.conv12_w, strides = [1,1,1,1], padding = 'SAME') + self.conv12_b
        self.bn12 = tf.keras.layers.BatchNormalization()(self.conv12)

        self.conv13_w = tf.Variable(tf.truncated_normal(shape = [1,1,self.nf*2,self.nf*4],mean = mu, stddev = sigma),name = 'conv13_w')
        self.conv13_b = tf.Variable(tf.zeros(self.nf*4),name = 'conv13_b',trainable=False)
        self.conv13 = tf.nn.conv2d(self.relu10,self.conv13_w, strides = [1,2,2,1], padding = 'SAME') + self.conv13_b
        self.bn13 = tf.keras.layers.BatchNormalization()(self.conv13)
        self.relu12_13 = tf.nn.relu(tf.keras.layers.add([self.bn12, self.bn13]))

        self.conv14_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*4,self.nf*4],mean = mu, stddev = sigma),name = 'conv14_w')
        self.conv14_b = tf.Variable(tf.zeros(self.nf*4),name = 'conv14_b',trainable=False)
        self.conv14 = tf.nn.conv2d(self.relu12_13,self.conv14_w, strides = [1,1,1,1], padding = 'SAME') + self.conv14_b
        self.bn14 = tf.keras.layers.BatchNormalization()(self.conv14)
        self.relu14 = tf.nn.relu(self.bn14)
        self.conv15_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*4,self.nf*4],mean = mu, stddev = sigma),name = 'conv15_w')
        self.conv15_b = tf.Variable(tf.zeros(self.nf*4),name = 'conv15_b',trainable=False)
        self.conv15 = tf.nn.conv2d(self.relu14,self.conv15_w, strides = [1,1,1,1], padding = 'SAME') + self.conv15_b
        self.bn15 = tf.keras.layers.BatchNormalization()(self.conv15)
        self.relu15 = tf.nn.relu(self.bn15)
        
        #forth basic block conv 3*3 nf*10 stride [2,1]
        self.conv16_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*4,self.nf*10],mean = mu, stddev = sigma),name = 'conv16_w')
        self.conv16_b = tf.Variable(tf.zeros(self.nf*10),name = 'conv16_b',trainable=False)
        self.conv16 = tf.nn.conv2d(self.relu15,self.conv16_w, strides = [1,2,2,1], padding = 'SAME') + self.conv16_b
        self.bn16 = tf.keras.layers.BatchNormalization()(self.conv16)
        self.relu16 = tf.nn.relu(self.bn16)
        self.conv17_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*10,self.nf*10],mean = mu, stddev = sigma),name = 'conv17_w')
        self.conv17_b = tf.Variable(tf.zeros(self.nf*10),name = 'conv17_b',trainable=False)
        self.conv17 = tf.nn.conv2d(self.relu16,self.conv17_w, strides = [1,1,1,1], padding = 'SAME') + self.conv17_b
        self.bn17 = tf.keras.layers.BatchNormalization()(self.conv17)

        self.conv18_w = tf.Variable(tf.truncated_normal(shape = [1,1,self.nf*4,self.nf*10],mean = mu, stddev = sigma),name = 'conv18_w')
        self.conv18_b = tf.Variable(tf.zeros(self.nf*10),name = 'conv18_b',trainable=False)
        self.conv18 = tf.nn.conv2d(self.relu15,self.conv18_w, strides = [1,2,2,1], padding = 'SAME') + self.conv18_b
        self.bn18 = tf.keras.layers.BatchNormalization()(self.conv18)
        self.relu17_18 = tf.nn.relu(tf.keras.layers.add([self.bn17, self.bn18]))

        self.conv19_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*10,self.nf*10],mean = mu, stddev = sigma),name = 'conv19_w')
        self.conv19_b = tf.Variable(tf.zeros(self.nf*10),name = 'conv19_b',trainable=False)
        self.conv19 = tf.nn.conv2d(self.relu17_18,self.conv19_w, strides = [1,1,1,1], padding = 'SAME') + self.conv19_b
        self.bn19 = tf.keras.layers.BatchNormalization()(self.conv19)
        self.relu19 = tf.nn.relu(self.bn19)
        self.conv20_w = tf.Variable(tf.truncated_normal(shape = [3,3,self.nf*10,self.nf*10],mean = mu, stddev = sigma),name = 'conv20_w')
        self.conv20_b = tf.Variable(tf.zeros(self.nf*10),name = 'conv1_b',trainable=False)
        self.conv20 = tf.nn.conv2d(self.relu19,self.conv20_w, strides = [1,1,1,1], padding = 'SAME') + self.conv20_b
        self.bn20 = tf.keras.layers.BatchNormalization()(self.conv20)
        self.relu20 = tf.nn.relu(self.bn20)

        #print("shape relu20 build:{}".format(self.relu20.get_shape()))
        #print("shape build avg:{}".format(tf.keras.layers.AveragePooling2D(4,padding='same')(self.relu20)))
        #average pool layer
        self.avgpool =  tf.contrib.layers.flatten(tf.keras.layers.AveragePooling2D(4,padding='same')(self.relu20))
        #self.avgpool = tf.squeeze(self.avgpool_test)
        #print("shape:{}".format(self.avgpool.get_shape()))
        #self.flat = tf.contrib.layers.flatten(self.avgpool)
        #print("shape:{}".format(self.avgpool.get_shape()))

        #fully connected layer
        self.fc1_w = tf.Variable(tf.random.truncated_normal(shape = (self.nf*10,self.nf*8), mean = mu, stddev = sigma),name = 'fc1_w')
        self.fc1_b = tf.Variable(tf.zeros(self.nf*8),name = 'fc1_b')
        self.linear1 = tf.matmul(self.avgpool,self.fc1_w) + self.fc1_b
        self.relu21 = tf.nn.relu(self.linear1)

        self.fc2_w = tf.Variable(tf.random.truncated_normal(shape = (self.nf*8,self.nf*6), mean = mu, stddev = sigma),name = 'fc2_w')
        self.fc2_b = tf.Variable(tf.zeros(self.nf*6),name = 'fc1_b')
        self.linear2 = tf.matmul(self.relu21,self.fc2_w) + self.fc2_b
        self.relu22 = tf.nn.relu(self.linear2)

        self.fc3_w = tf.Variable(tf.random.truncated_normal(shape = (self.nf*6,self.class_number), mean = mu, stddev = sigma),name = 'fc3_w')
        self.fc3_b = tf.Variable(tf.zeros(self.class_number),name = 'fc3_b')
        self.linear3 = tf.matmul(self.relu22,self.fc3_w) + self.fc3_b

        self.y = self.linear3
        #self.y = tf.squeeze(self.y)
        
        print("y build shape:{}".format(self.y.get_shape()))
        # lists
        self.var_list = [self.conv1_w, self.conv1_b,
                        self.conv2_w, self.conv2_b,
                        self.conv3_w, self.conv3_b,
                        self.conv4_w, self.conv4_b,
                        self.conv5_w, self.conv5_b,
                        self.conv6_w, self.conv6_b,
                        self.conv7_w, self.conv7_b,
                        self.conv8_w, self.conv8_b,
                        self.conv9_w, self.conv9_b,
                        self.conv10_w, self.conv10_b,
                        self.conv11_w, self.conv11_b,
                        self.conv12_w, self.conv12_b,
                        self.conv13_w, self.conv13_b,
                        self.conv14_w, self.conv14_b,
                        self.conv15_w, self.conv15_b,
                        self.conv16_w, self.conv16_b,
                        self.conv17_w, self.conv17_b,
                        self.conv18_w, self.conv18_b,
                        self.conv19_w, self.conv19_b,
                        self.conv20_w, self.conv20_b,
                        self.fc1_w, self.fc1_b, 
                        self.fc2_w, self.fc2_b, 
                        self.fc3_w, self.fc3_b]
        self.hidden_list = [self.conv1, self.conv2, self.conv3,
                            self.conv4, self.conv5, self.conv6,
                            self.conv7, self.conv8, self.conv9,
                            self.conv10, self.conv11, self.conv12,
                            self.conv13, self.conv14, self.conv15,
                            self.conv16, self.conv17, self.conv18,
                            self.conv19, self.conv20, self.linear1,
                            self.linear2, self.linear3]
        self.input_list = [self.x, self.relu1, self.relu2,
                           self.relu3, self.relu4, self.relu5,
                           self.relu6, self.relu5, self.relu7_8,
                           self.relu9, self.relu10, self.relu11,
                           self.relu10, self.relu12_13, self.relu14,
                           self.relu15, self.relu16, self.relu15,  
                           self.relu17_18, self.relu19, self.avgpool, 
                            self.relu21, self.relu22
                           ]

        
        self.b_list = [self.conv1_b, self.conv2_b, self.conv3_b, 
                        self.conv4_b, self.conv5_b, self.conv6_b, 
                        self.conv7_b, self.conv8_b, self.conv9_b, 
                        self.conv10_b, self.conv11_b, self.conv12_b, 
                        self.conv13_b, self.conv14_b, self.conv15_b, 
                        self.conv16_b, self.conv17_b, self.conv18_b, 
                        self.conv19_b, self.conv20_b, 
                        self.fc1_b, self.fc2_b, self.fc3_b]
        self.b_val_list = []
        # vanilla single-task loss
        """ offset1, offset2 = self.compute_offsets(self.class_number_list,t)
        if offset1 > 0:
            tf.slice(self.y,[0])
            self.y[:, :offset1]*tf.constant([-10e10])
        if offset2 < self.class_number:
            self.y[:, int(offset2):self.class_number]*tf.constant([-10e10]) """
        
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
        
        # input layer 3*3
        if self.doDecom[0]:
            self.conv1_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv1_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv1_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv1_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[0])))
            self.conv1 = tf.nn.conv2d(self.x,self.conv1_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv1 = tf.nn.conv2d(self.conv1,self.conv1_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv1 = tf.nn.conv2d(self.conv1,self.conv1_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv1_b
            self.bn1 = tf.keras.layers.BatchNormalization()(self.conv1)
            self.relu1 = tf.nn.relu(self.bn1)
            self.var_list.append(self.conv1_w1)
            self.var_list.append(self.conv1_w2)
            self.var_list.append(self.conv1_w3)
            self.var_list.append(self.conv1_b)
            pos = pos + 3
        
        #first basic block conv 3*3 nf stride 1
        if self.doDecom[1]:
            self.conv2_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv2_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv2_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv2_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[1])))
            self.conv2 = tf.nn.conv2d(self.relu1,self.conv2_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv2 = tf.nn.conv2d(self.conv2,self.conv2_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv2 = tf.nn.conv2d(self.conv2,self.conv2_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv2_b
            self.bn2 = tf.keras.layers.BatchNormalization()(self.conv2)
            self.relu2 = tf.nn.relu(self.bn2)
            self.var_list.append(self.conv2_w1)
            self.var_list.append(self.conv2_w2)
            self.var_list.append(self.conv2_w3)
            self.var_list.append(self.conv2_b)
            pos = pos + 3

        #first basic block conv 3*3 nf stride 1
        if self.doDecom[2]:
            self.conv3_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv3_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv3_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv3_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[2])))
            self.conv3 = tf.nn.conv2d(self.relu2,self.conv3_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv3 = tf.nn.conv2d(self.conv3,self.conv3_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv3 = tf.nn.conv2d(self.conv3,self.conv3_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv3_b
            self.bn3 = tf.keras.layers.BatchNormalization()(self.conv3)
            self.relu3 = tf.nn.relu(self.bn3)
            self.var_list.append(self.conv3_w1)
            self.var_list.append(self.conv3_w2)
            self.var_list.append(self.conv3_w3)
            self.var_list.append(self.conv3_b)
            pos = pos + 3
        
        #first basic block conv 3*3 nf stride 1
        if self.doDecom[3]:
            self.conv4_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv4_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv4_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv4_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[3])))
            self.conv4 = tf.nn.conv2d(self.relu3,self.conv4_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv4 = tf.nn.conv2d(self.conv4,self.conv4_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv4 = tf.nn.conv2d(self.conv4,self.conv4_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv4_b
            self.bn4 = tf.keras.layers.BatchNormalization()(self.conv4)
            self.relu4 = tf.nn.relu(self.bn4)
            self.var_list.append(self.conv4_w1)
            self.var_list.append(self.conv4_w2)
            self.var_list.append(self.conv4_w3)
            self.var_list.append(self.conv4_b)
            pos = pos + 3
        
        #first basic block conv 3*3 nf stride 1
        if self.doDecom[4]:
            self.conv5_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv5_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv5_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv5_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[4])))
            self.conv5 = tf.nn.conv2d(self.relu4,self.conv5_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv5 = tf.nn.conv2d(self.conv5,self.conv5_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv5 = tf.nn.conv2d(self.conv5,self.conv5_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv5_b
            self.bn5 = tf.keras.layers.BatchNormalization()(self.conv5)
            self.relu5 = tf.nn.relu(self.bn5)
            self.var_list.append(self.conv5_w1)
            self.var_list.append(self.conv5_w2)
            self.var_list.append(self.conv5_w3)
            self.var_list.append(self.conv5_b)
            pos = pos + 3
        
        #second basic block conv 3*3 nf*2 stride 2
        if self.doDecom[5]:
            self.conv6_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv6_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv6_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv6_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[5])))
            self.conv6 = tf.nn.conv2d(self.relu5,self.conv6_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv6 = tf.nn.conv2d(self.conv6,self.conv6_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv6 = tf.nn.conv2d(self.conv6,self.conv6_w3, strides = [1,2,2,1], padding = 'SAME') + self.conv6_b
            self.bn6 = tf.keras.layers.BatchNormalization()(self.conv6)
            self.relu6 = tf.nn.relu(self.bn6)
            self.var_list.append(self.conv6_w1)
            self.var_list.append(self.conv6_w2)
            self.var_list.append(self.conv6_w3)
            self.var_list.append(self.conv6_b)
            pos = pos + 3
        
        #second basic block conv 3*3 nf*2 stride 1
        if self.doDecom[6]:
            self.conv7_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv7_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv7_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv7_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[6]))) 
            self.conv7 = tf.nn.conv2d(self.relu6,self.conv7_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv7 = tf.nn.conv2d(self.conv7,self.conv7_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv7 = tf.nn.conv2d(self.conv7,self.conv7_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv7_b
            self.bn7 = tf.keras.layers.BatchNormalization()(self.conv7)
            self.var_list.append(self.conv7_w1)
            self.var_list.append(self.conv7_w2)
            self.var_list.append(self.conv7_w3)
            self.var_list.append(self.conv7_b)
            pos = pos + 3

        #second basic block shortcut
        if self.doDecom[7]:
            self.conv8_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv8_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv8_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv8_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[7])))
            self.conv8 = tf.nn.conv2d(self.relu5,self.conv8_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv8 = tf.nn.conv2d(self.conv8,self.conv8_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv8 = tf.nn.conv2d(self.conv8,self.conv8_w3, strides = [1,2,2,1], padding = 'SAME') + self.conv8_b
            self.bn8 = tf.keras.layers.BatchNormalization()(self.conv8)
            self.relu7_8 = tf.nn.relu(tf.keras.layers.add([self.bn7, self.bn8]))
            self.var_list.append(self.conv8_w1)
            self.var_list.append(self.conv8_w2)
            self.var_list.append(self.conv8_w3)
            self.var_list.append(self.conv8_b)
            pos = pos + 3
        
        #second basic block conv 3*3 nf*2 stride 1
        if self.doDecom[8]:
            self.conv9_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv9_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv9_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv9_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[8])))
            self.conv9 = tf.nn.conv2d(self.relu7_8,self.conv9_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv9 = tf.nn.conv2d(self.conv9,self.conv9_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv9 = tf.nn.conv2d(self.conv9,self.conv9_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv9_b
            self.bn9 = tf.keras.layers.BatchNormalization()(self.conv9)
            self.relu9 = tf.nn.relu(self.bn9)
            self.var_list.append(self.conv9_w1)
            self.var_list.append(self.conv9_w2)
            self.var_list.append(self.conv9_w3)
            self.var_list.append(self.conv9_b)
            pos = pos + 3
        
        #second basic block conv 3*3 nf*2 stride 1
        if self.doDecom[9]:
            self.conv10_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv10_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv10_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv10_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[9])))
            self.conv10 = tf.nn.conv2d(self.relu9,self.conv10_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv10 = tf.nn.conv2d(self.conv10,self.conv10_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv10 = tf.nn.conv2d(self.conv10,self.conv10_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv10_b
            self.bn10 = tf.keras.layers.BatchNormalization()(self.conv10)
            self.relu10 = tf.nn.relu(self.bn10)
            self.var_list.append(self.conv10_w1)
            self.var_list.append(self.conv10_w2)
            self.var_list.append(self.conv10_w3)
            self.var_list.append(self.conv10_b)
            pos = pos + 3
        
        #third basic block conv 3*3 nf*4 stride 2
        if self.doDecom[10]:
            self.conv11_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv11_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv11_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv11_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[10])))
            self.conv11 = tf.nn.conv2d(self.relu10,self.conv11_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv11 = tf.nn.conv2d(self.conv11,self.conv11_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv11 = tf.nn.conv2d(self.conv11,self.conv11_w3, strides = [1,2,2,1], padding = 'SAME') + self.conv11_b
            self.bn11 = tf.keras.layers.BatchNormalization()(self.conv11)
            self.relu11 = tf.nn.relu(self.bn11)
            self.var_list.append(self.conv11_w1)
            self.var_list.append(self.conv11_w2)
            self.var_list.append(self.conv11_w3)
            self.var_list.append(self.conv11_b)
            pos = pos + 3
        
        #third basic block conv 3*3 nf*4 stride 1
        if self.doDecom[11]:
            self.conv12_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv12_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv12_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv12_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[11])))
            self.conv12 = tf.nn.conv2d(self.relu11,self.conv12_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv12 = tf.nn.conv2d(self.conv12,self.conv12_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv12 = tf.nn.conv2d(self.conv12,self.conv12_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv12_b
            self.bn12 = tf.keras.layers.BatchNormalization()(self.conv12)
            self.var_list.append(self.conv12_w1)
            self.var_list.append(self.conv12_w2)
            self.var_list.append(self.conv12_w3)
            self.var_list.append(self.conv12_b)
            pos = pos + 3
        
        #third basic block shortcut
        if self.doDecom[12]:
            self.conv13_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv13_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv13_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv13_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[12])))
            self.conv13 = tf.nn.conv2d(self.relu10,self.conv13_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv13 = tf.nn.conv2d(self.conv13,self.conv13_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv13 = tf.nn.conv2d(self.conv13,self.conv13_w3, strides = [1,2,2,1], padding = 'SAME') + self.conv13_b
            self.bn13 = tf.keras.layers.BatchNormalization()(self.conv13)
            self.relu12_13 = tf.nn.relu(tf.keras.layers.add([self.bn12, self.bn13]))
            self.var_list.append(self.conv13_w1)
            self.var_list.append(self.conv13_w2)
            self.var_list.append(self.conv13_w3)
            self.var_list.append(self.conv13_b)
            pos = pos + 3
        
        #third basic block conv 3*3 nf*4 stride 1
        if self.doDecom[13]:
            self.conv14_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv14_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv14_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv14_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[13])))
            self.conv14 = tf.nn.conv2d(self.relu12_13,self.conv14_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv14 = tf.nn.conv2d(self.conv14,self.conv14_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv14 = tf.nn.conv2d(self.conv14,self.conv14_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv14_b
            self.bn14 = tf.keras.layers.BatchNormalization()(self.conv14)
            self.relu14 = tf.nn.relu(self.bn14)
            self.var_list.append(self.conv14_w1)
            self.var_list.append(self.conv14_w2)
            self.var_list.append(self.conv14_w3)
            self.var_list.append(self.conv14_b)
            pos = pos + 3
        
        #third basic block conv 3*3 nf*4 stride 1
        if self.doDecom[14]:
            self.conv15_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv15_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv15_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv15_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[14])))
            self.conv15 = tf.nn.conv2d(self.relu14,self.conv15_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv15 = tf.nn.conv2d(self.conv15,self.conv15_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv15 = tf.nn.conv2d(self.conv15,self.conv15_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv15_b
            self.bn15 = tf.keras.layers.BatchNormalization()(self.conv15)
            self.relu15 = tf.nn.relu(self.bn15)
            self.var_list.append(self.conv15_w1)
            self.var_list.append(self.conv15_w2)
            self.var_list.append(self.conv15_w3)
            self.var_list.append(self.conv15_b)
            pos = pos + 3
        
        #forth basic block conv 3*3 nf*10 stride 2
        if self.doDecom[15]:
            self.conv16_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv16_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv16_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv16_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[15])))
            self.conv16 = tf.nn.conv2d(self.relu15,self.conv16_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv16 = tf.nn.conv2d(self.conv16,self.conv16_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv16 = tf.nn.conv2d(self.conv16,self.conv16_w3, strides = [1,2,2,1], padding = 'SAME') + self.conv16_b
            self.bn16 = tf.keras.layers.BatchNormalization()(self.conv16)
            self.relu16 = tf.nn.relu(self.bn16)
            self.var_list.append(self.conv16_w1)
            self.var_list.append(self.conv16_w2)
            self.var_list.append(self.conv16_w3)
            self.var_list.append(self.conv16_b)
            pos = pos + 3
        
        #forth basic block conv 3*3 nf*10 stride 1
        if self.doDecom[16]:
            self.conv17_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv17_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv17_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv17_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[16])))
            self.conv17 = tf.nn.conv2d(self.relu16,self.conv17_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv17 = tf.nn.conv2d(self.conv17,self.conv17_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv17 = tf.nn.conv2d(self.conv17,self.conv17_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv17_b
            self.bn17 = tf.keras.layers.BatchNormalization()(self.conv17)
            self.var_list.append(self.conv17_w1)
            self.var_list.append(self.conv17_w2)
            self.var_list.append(self.conv17_w3)
            self.var_list.append(self.conv17_b)
            pos = pos + 3
        
        #forth basic block shortcut
        if self.doDecom[17]:
            self.conv18_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv18_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv18_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv18_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[17])))
            self.conv18 = tf.nn.conv2d(self.relu15,self.conv18_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv18 = tf.nn.conv2d(self.conv18,self.conv18_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv18 = tf.nn.conv2d(self.conv18,self.conv18_w3, strides = [1,2,2,1], padding = 'SAME') + self.conv18_b
            self.bn18 = tf.keras.layers.BatchNormalization()(self.conv18)
            self.relu17_18 = tf.nn.relu(tf.keras.layers.add([self.bn17, self.bn18]))
            self.var_list.append(self.conv18_w1)
            self.var_list.append(self.conv18_w2)
            self.var_list.append(self.conv18_w3)
            self.var_list.append(self.conv18_b)
            pos = pos + 3
        
        #forth basic block conv 3*3 nf*10 stride 1
        if self.doDecom[18]:
            self.conv19_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv19_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv19_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv19_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[18])))
            self.conv19 = tf.nn.conv2d(self.relu17_18,self.conv19_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv19 = tf.nn.conv2d(self.conv19,self.conv19_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv19 = tf.nn.conv2d(self.conv19,self.conv19_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv19_b
            self.bn19 = tf.keras.layers.BatchNormalization()(self.conv19)
            self.relu19 = tf.nn.relu(self.bn19)
            self.var_list.append(self.conv19_w1)
            self.var_list.append(self.conv19_w2)
            self.var_list.append(self.conv19_w3)
            self.var_list.append(self.conv19_b)
            pos = pos + 3
        
        #forth basic block conv 3*3 nf*10 stride 1
        if self.doDecom[19]:
            self.conv20_w1 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos],axis=0),axis=0))), trainable = False)
            self.conv20_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.conv20_w3 = tf.Variable(tf.convert_to_tensor(np.float32(np.expand_dims(np.expand_dims(self.weights_svd[pos+2],axis=0),axis=0))), trainable = False)
            self.conv20_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[19])))
            self.conv20 = tf.nn.conv2d(self.relu19,self.conv20_w1, strides = [1,1,1,1], padding = 'SAME')
            self.conv20 = tf.nn.conv2d(self.conv20,self.conv20_w2, strides = [1,1,1,1], padding = 'SAME')
            self.conv20 = tf.nn.conv2d(self.conv20,self.conv20_w3, strides = [1,1,1,1], padding = 'SAME') + self.conv20_b
            self.bn20 = tf.keras.layers.BatchNormalization()(self.conv20)
            self.relu20 = tf.nn.relu(self.bn20)
            self.var_list.append(self.conv20_w1)
            self.var_list.append(self.conv20_w2)
            self.var_list.append(self.conv20_w3)
            self.var_list.append(self.conv20_b)
            pos = pos + 3

        #average pool and first fully-connected layer
        if self.doDecom[20]:
            self.avgpool = tf.contrib.layers.flatten(tf.keras.layers.AveragePooling2D(4)(self.relu20))
            self.fc1_w1 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos])),trainable = False)
            self.fc1_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.fc1_w3 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+2])),trainable = False)
            self.fc1_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[20])))
            self.linear1 = tf.matmul(tf.matmul(tf.matmul(self.avgpool,self.fc1_w1),self.fc1_w2),self.fc1_w3) + self.fc1_b
            self.relu21 = tf.nn.relu(self.linear1)
            self.var_list.append(self.fc1_w1)
            self.var_list.append(self.fc1_w2)
            self.var_list.append(self.fc1_w3)
            self.var_list.append(self.fc1_b)
            pos = pos + 3
        #print("dodecom20:{}".format(self.relu21.get_shape()))
        #second fully-connected layer
        if self.doDecom[21]:
            self.fc2_w1 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos])),trainable = False)
            self.fc2_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+1])))
            self.fc2_w3 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos+2])),trainable = False)
            self.fc2_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[21])))
            self.linear2 = tf.matmul(tf.matmul(tf.matmul(self.relu21,self.fc2_w1),self.fc2_w2),self.fc2_w3) + self.fc2_b
            self.relu22 = tf.nn.relu(self.linear2)
            self.var_list.append(self.fc2_w1)
            self.var_list.append(self.fc2_w2)
            self.var_list.append(self.fc2_w3)
            self.var_list.append(self.fc2_b)
            pos = pos + 3
        #print("dodecom21:{}".format(self.relu22.get_shape()))
        
        
        #final fully-connected layer
        if not self.doDecom[22]:
            self.fc3_w = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[pos])))
            self.fc3_b = tf.Variable(tf.convert_to_tensor(np.float32(self.b_val_list[22])))
            self.linear3 = tf.matmul(self.relu22, self.fc3_w) + self.fc3_b
            self.y = self.linear3
            self.var_list.append(self.fc3_w)
            self.var_list.append(self.fc3_b)
            pos = pos + 1

        #print("y rebuild shape:{}".format(self.y.get_shape()))

        # lists
        self.hidden_list = [self.conv1, self.conv2, self.conv3,
                            self.conv4, self.conv5, self.conv6,
                            self.conv7, self.conv8, self.conv9,
                            self.conv10, self.conv11, self.conv12,
                            self.conv13, self.conv14, self.conv15,
                            self.conv16, self.conv17, self.conv18,
                            self.conv19, self.conv20, self.linear1,
                            self.linear2, self.linear3]
        self.input_list = [self.x, self.relu1, self.relu2,
                           self.relu3, self.relu4, self.relu5,
                           self.relu6, self.relu5, self.relu7_8,
                           self.relu9, self.relu10, self.relu11,
                           self.relu10, self.relu12_13, self.relu14,
                           self.relu15, self.relu16, self.relu15,  
                           self.relu17_18, self.relu19, self.avgpool, 
                            self.relu21, self.relu22]
        
        self.b_list = [self.conv1_b, self.conv2_b, self.conv3_b, 
                        self.conv4_b, self.conv5_b, self.conv6_b, 
                        self.conv7_b, self.conv8_b, self.conv9_b, 
                        self.conv10_b, self.conv11_b, self.conv12_b, 
                        self.conv13_b, self.conv14_b, self.conv15_b, 
                        self.conv16_b, self.conv17_b, self.conv18_b, 
                        self.conv19_b, self.conv20_b, 
                        self.fc1_b, self.fc2_b, self.fc3_b]       

        # vanilla single-task loss
        #scores = self.y
        #new_cl = range(0,10)
        #label_new_classes =  tf.stack([y_[:,i] for i in new_cl],axis=1) 
        #pred_new_classes = tf.stack([scores[:,i] for i in new_cl],axis=1)

        #self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes,
        #                                                                            logits=pred_new_classes))
        """ offset1, offset2 = self.compute_offsets(self.class_number_list,t)
        if offset1 > 0:
            tf.Variable(self.y)[:, :offset1].assgin(-10e10)
        if offset2 < self.class_number:
            tf.Variable(self.y)[:, int(offset2):self.class_number].assign(-10e10) """

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
    
    def update_ewc_loss(self, lr, lam, num_nodes=20):
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
            #print("task 1 on")
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
                        if len(self.weights_w[v*2].shape)==4:
                            # it's a convolutional layer
                            W = self.weights_w[v*2]
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
                            
                            self.weights_svd.append(U1)
                            self.weights_svd.append(Wp)
                            self.weights_svd.append(U2)
                            
                            # check by fixing spatial dimensions
                            W_check = np.zeros(W.shape)
                            for w in range(W.shape[0]):
                                for h in range(W.shape[1]):
                                    W_check[w,h,:,:] = np.dot(np.dot(U1,Wp[w,h,:,:]),U2)
        else:
            #print("task {} on".format(t+1))
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
                    else:
                        if len(self.weights_w[v*4].shape)==4:
                            # it's a convolutional layer
                            W_rec = np.zeros(self.weights_w[v*4+1].shape)
                            for w in range(self.weights_w[v*4+1].shape[0]):
                                for h in range(self.weights_w[v*4+1].shape[1]):
                                    W_rec[w,h,:,:] = np.dot(np.dot(self.weights_w[v*4][0,0,:,:],self.weights_w[v*4+1][w,h,:,:]),self.weights_w[v*4+2][0,0,:,:])
                            Wp = np.zeros(W_rec.shape)
                            # fix the output dimension
                            for w in range(W_rec.shape[0]):
                                for h in range(W_rec.shape[1]):
                                    for o in range(W_rec.shape[3]):
                                        Wp[w,h,:,o] = np.dot(Q1,W_rec[w,h,:,o])
                            # fix the input dimension
                            for w in range(W_rec.shape[0]):
                                for h in range(W_rec.shape[1]):
                                    for i in range(W_rec.shape[2]):
                                        Wp[w,h,i,:] = np.dot(Wp[w,h,i,:],Q2)
                            
                            self.weights_svd.append(U1)
                            self.weights_svd.append(Wp)
                            self.weights_svd.append(U2)
                            
                            # check by fixing spatial dimensions
                            W_check = np.zeros(W_rec.shape)
                            for w in range(W_rec.shape[0]):
                                for h in range(W_rec.shape[1]):
                                    W_check[w,h,:,:] = np.dot(np.dot(U1,Wp[w,h,:,:]),U2)

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
                
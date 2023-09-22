import numpy as np
import time
from pyDOE import lhs
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import shutil
import pickle
import math
import scipy.io

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;
np.random.seed(1111)
tf.compat.v1.set_random_seed(1111)

class DeepHPM:
    # Initialize the class
    def __init__(self, Collo, snap1, snap2, vp, uv_layers, lb, ub, ExistModel=0, modelDir=''):

        self.count = 0
        self.vp = vp

        
        self.lb = lb
        self.ub = ub
        self.loss_f_rec = []
        self.loss_snap_rec=[]
        self.loss_rec=[]
        self.train_loss=[]
   

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]
        self.t_c = Collo[:, 2:3]

        self.x_snap1 = snap1[:, 0:1]
        self.y_snap1 = snap1[:, 1:2]
        self.t_snap1 = snap1[:, 2:3]
        self.u_snap1 = snap1[:, 3:4]
        
        self.x_snap2 = snap2[:, 0:1]
        self.y_snap2 = snap2[:, 1:2]
        self.t_snap2 = snap2[:, 2:3]
        self.u_snap2 = snap2[:, 3:4]
        
       
        # Define layers
        self.uv_layers = uv_layers

        # Initialize NNs
        if ExistModel== 0 :
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            self.uv_weights, self.uv_biases = self.load_NN(modelDir, self.uv_layers)
        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])    # Point for postprocessing
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float32, shape=[None, self.y_c.shape[1]])
        self.t_c_tf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])

        self.x_snap1_tf = tf.placeholder(tf.float32, shape=[None, self.x_snap1.shape[1]])
        self.y_snap1_tf = tf.placeholder(tf.float32, shape=[None, self.y_snap1.shape[1]])
        self.t_snap1_tf = tf.placeholder(tf.float32, shape=[None, self.t_snap1.shape[1]])
        self.u_snap1_tf = tf.placeholder(tf.float32, shape=[None, self.u_snap1.shape[1]])

        self.x_snap2_tf = tf.placeholder(tf.float32, shape=[None, self.x_snap2.shape[1]])
        self.y_snap2_tf = tf.placeholder(tf.float32, shape=[None, self.y_snap2.shape[1]])
        self.t_snap2_tf = tf.placeholder(tf.float32, shape=[None, self.t_snap2.shape[1]])
        self.u_snap2_tf = tf.placeholder(tf.float32, shape=[None, self.u_snap2.shape[1]])
 
        self.u_pred = self.net_uv(self.x_tf, self.y_tf, self.t_tf)
        self.u_snap2_pred  = self.net_uv(self.x_snap2_tf, self.y_snap2_tf, self.t_snap2_tf)
        self.u_snap1_pred  = self.net_uv(self.x_snap1_tf, self.y_snap1_tf, self.t_snap1_tf)
  
      
        self.f_pred_u = self.net_f_sig(self.x_c_tf, self.y_c_tf, self.t_c_tf)

        self.loss_f = tf.reduce_mean(tf.square(self.f_pred_u)) 
    
        self.loss_snap = tf.reduce_mean(tf.square(self.u_snap1_pred - self.u_snap1_tf)) \
                         + tf.reduce_mean(tf.square(self.u_snap2_pred - self.u_snap2_tf))

        self.loss = self.loss_snap + 1e-4*self.loss_f
        #self.loss = self.loss_snap + 1e-3*self.loss_f #for the second training
        #self.loss = self.loss_snap + 1e-1*self.loss_f #for the third training

        # Optimizer for solution
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 20000,
                                                                         'maxfun': 20000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1 * np.finfo(float).eps})

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        #starter_learning_rate = 0.0001 #for the second training
        #starter_learning_rate = 0.00001 #for the third training
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        10000, 0.9, staircase=False)
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss,
                                                          var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


    def save_NN(self, fileDir):
        uv_weights = self.sess.run(self.uv_weights)
        uv_biases = self.sess.run(self.uv_biases)
        with open(fileDir, 'wb') as f:
            # pickle.dump([np.array(uv_weights), np.array(uv_biases)], f)
            pickle.dump([uv_weights, uv_biases], f)
            print("Save NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)
            # print(len(uv_weights))
            # print(np.shape(uv_weights))
            # print(num_layers)

            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights)+1)

            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num])
                b = tf.Variable(uv_biases[num])
                weights.append(W)
                biases.append(b)
                print("Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        # H = X
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_uv(self, x, y, t):
        # This NN return sigma_phi
        u = self.neural_net(tf.concat([x, y, t], 1), self.uv_weights, self.uv_biases)
        u = u[:, 0:1]
        #ut = tf.gradients(u, t)[0]
        return u 

    def net_f_sig(self, x, y, t):
        
        vp =self.vp         
        u = self.net_uv(x, y, t)
        ut = tf.gradients(u, t)[0]
        u_tt = tf.gradients(ut, t)[0]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        f_u = u_tt - vp*vp*(u_xx + u_yy)
        return f_u

    def callback(self, loss):
        self.count = self.count + 1
        print('{} th iterations, Loss: {}'.format(self.count, loss))
        self.train_loss.append(loss)

    def train(self, iter, batch_num):


        loss_f = []
        loss_snap = []
        loss = []

        # The collocation point is splited into partitions of batch_num，1 epoch for training
        for i in range(batch_num):
            col_num = self.x_c.shape[0]
            idx_start = int(i * col_num / batch_num)
            idx_end = int((i + 1) * col_num / batch_num)

            tf_dict = {self.x_c_tf: self.x_c[idx_start:idx_end,:], self.y_c_tf: self.y_c[idx_start:idx_end,:], self.t_c_tf: self.t_c[idx_start:idx_end,:],
                       self.x_snap1_tf: self.x_snap1, self.y_snap1_tf: self.y_snap1, self.t_snap1_tf: self.t_snap1, self.u_snap1_tf: self.u_snap1,
                       self.x_snap2_tf: self.x_snap2, self.y_snap2_tf: self.y_snap2, self.t_snap2_tf: self.t_snap2, self.u_snap2_tf: self.u_snap2}

            for it in range(iter):

                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 10 == 0:
                    loss_value = self.sess.run(self.loss, tf_dict)
                    loss_f_value = self.sess.run(self.loss_f, tf_dict)
                    loss_snap_value = self.sess.run(self.loss_snap, tf_dict)
                    print('It: %d, Loss: %.3e, %.3e, %.3e' %
                          (it, loss_value, loss_snap_value,loss_f_value))

                self.loss_f_rec.append(self.sess.run(self.loss_f, tf_dict))
                self.loss_snap_rec.append(self.sess.run(self.loss_snap, tf_dict))
                self.loss_rec.append(self.sess.run(self.loss, tf_dict))

        return self.loss, self.loss_snap, self.loss_f

    def train_bfgs(self, batch_num):
        # The collocation point is splited into partitions of batch_num
        for i in range(batch_num):
            col_num = self.x_c.shape[0]
            idx_start = int(i*col_num/batch_num)
            idx_end = int((i+1)*col_num/batch_num)
            tf_dict = {self.x_c_tf: self.x_c[idx_start:idx_end,:], self.y_c_tf: self.y_c[idx_start:idx_end,:], self.t_c_tf: self.t_c[idx_start:idx_end,:],
                       self.x_snap1_tf: self.x_snap1, self.y_snap1_tf: self.y_snap1, self.t_snap1_tf: self.t_snap1, self.u_snap1_tf: self.u_snap1,
                       self.x_snap2_tf: self.x_snap2, self.y_snap2_tf: self.y_snap2, self.t_snap2_tf: self.t_snap2, self.u_snap2_tf: self.u_snap2}

            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss],
                                    loss_callback=self.callback)

    def predict(self, x_star, y_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        return u_star

    def probe(self, x_star, y_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})      
        return u_star
    
    def getloss(self):  # To be updated

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.x_snap1_tf: self.x_snap1, self.y_snap1_tf: self.y_snap1, self.t_snap1_tf: self.t_snap1, self.u_snap1_tf: self.u_snap1,
                   self.x_snap2_tf: self.x_snap2, self.y_snap2_tf: self.y_snap2, self.t_snap2_tf: self.t_snap2, self.u_snap2_tf: self.u_snap2}

        loss_f = self.sess.run(self.loss_f, tf_dict)
        loss_snap = self.sess.run(self.loss_snap, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
       
        return loss, loss_snap ,loss_f
    
if __name__ == "__main__":

    # Need pretraining!! (i.e. train for 10s -> 15s -> 25s)

    MAX_T = 1.2
    # Domain bounds
    lb = np.array([0.0, 0.0, 0.2])
    ub = np.array([2, 2, MAX_T])
    lb_sam = np.array([0.0, 0.5, 0.5])
    ub_sam = np.array([2, 1.5, MAX_T])

    uv_layers = [3] + 5*[64] + 3*[32]  + [1]

    # Num of collocation point in x, y, t
    N_f = 80000
    
    data_snap1 = scipy.io.loadmat('./ceng_zs0/u_t2.mat')
    data_snap2 = scipy.io.loadmat('./ceng_zs0/u_t4.mat')
    data_snap3 = scipy.io.loadmat('./ceng_zs0/u_t7.mat')
  
    u_snap1 = data_snap1['u_t0'].flatten()[:, None] 
    u_snap2 = data_snap2['u_t0'].flatten()[:, None]
    x_snap =  data_snap1['xx'].flatten()[:, None]
    y_snap =  data_snap1['zz'].flatten()[:, None]
    
    u_snap3 = data_snap3['u_t0'].flatten()[:, None] 

   
    t_snap1 = np.zeros((x_snap1.size, 1))
    t_snap1.fill(0.2)
    t_snap2 = np.zeros((x_snap2.size, 1))
    t_snap2.fill(0.3)
    t_snap3 = np.zeros((x_snap3.size, 1))
    t_snap3.fill(0.7)
    
    snap1 = np.concatenate((x_snap, y_snap, t_snap1, u_snap1), 1)
    snap2 = np.concatenate((x_snap, y_snap, t_snap2, u_snap2), 1)
    snap3 = np.concatenate((x_snap, y_snap, t_snap3, u_snap3), 1)
  
    N = x_snap1.shape[0]
    N_train = round(N/4*4)   
 # Training Data    
    idx = np.random.choice(N, N_train, replace=False)

    snap1_train = snap1[idx,:]
    snap2_train = snap2[idx,:]
  
    # Collocation point
    XYT = lb + (ub - lb) * lhs(3, N_f)
    XYT_c_sam = lb_sam + (ub_sam - lb_sam) * lhs(3, 30000)
    XYT_c = np.concatenate((XYT, XYT_c_sam), 0)
    vp = XYT_c[:,0:1]*0

    A = XYT_c[:,0:1]
    B = XYT_c[:,1:2]    
        
        
    for i in range(len(vp)):
        if B[i]<1:
            vp[i]=1.5
        else:
            vp[i]=1.9

    with tf.device('/device:GPU:0'):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        ## Load network if networks are provided
        model = DeepHPM(XYT_c, snap1_train, snap2_train, vp, uv_layers, lb, ub, ExistModel=1, modelDir='uv_NN_ceng_3snap_2.pickle')

        loss_snap, loss_f,loss = model.train(iter=60000, batch_num=1)
        model.train_bfgs(batch_num=1)

        model.save_NN('uv_NN_ceng_3snap_2_2snap_lr_1.pickle')

        model.getloss()
        

        # Output result at each time step
        x_star = np.linspace(0.0, 2, 101)
        y_star = np.linspace(0.0, 2, 101)
        x_star, y_star = np.meshgrid(x_star, y_star)
        x_star = x_star.flatten()[:, None]
        y_star = y_star.flatten()[:, None]
      
        shutil.rmtree('./output', ignore_errors=True)
        os.makedirs('./output')
        
        
        t_star_9 = np.zeros((x_star.size, 1))
        t_star_9.fill(0.9)
        u_pred_9 = model.predict(x_star, y_star, t_star_9)
        
        t_star_10 = np.zeros((x_star.size, 1))
        t_star_10.fill(1)
        u_pred_10 = model.predict(x_star, y_star, t_star_10)
        
        t_star_11 = np.zeros((x_star.size, 1))
        t_star_11.fill(1.1)
        u_pred_11 = model.predict(x_star, y_star, t_star_11)
        
    
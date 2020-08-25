import sys, math , time
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import warnings

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('INFO')

np.random.seed(1234)
tf.set_random_seed(1234)


def difference(source, target):
    return abs(source - target)


class PhysicsInformedNN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, mode):
        
        self.lb = lb
        self.ub = ub
    
        self.x_u = X_u[:,0:1]
        self.t_u = X_u[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.u = u
        
        self.layers = layers
        self.nu = nu
        self.mode = mode
        
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])        

        if self.mode == 'normal': 
            self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf) 
            self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)  
        else:
            self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf) 
            self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)      
        
        if self.mode == 'normal': 
            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        else:
            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                        tf.reduce_mean(tf.square(self.f_pred))
                          
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
                
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u
    
    def net_f(self, x, t):
        u = self.net_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        v = (- u*u_x + self.nu*u_xx)
        f = u_t + u*u_x - self.nu*u_xx
        return f, u_t, u, v, x, t
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)
                                    
    
    def predict(self, X_star, df):
        if self.mode == 'normal':    
            u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})
            f_star, u_t, u, v, x, t = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
            df['du_normal'] = u_t.flatten()
            df['u_normal'] = u.flatten()
            df['dv_normal'] = v.flatten()
            df['f_star_normal'] = f_star.flatten()
            df['x'] = x.flatten()
            df['t'] = t.flatten()
            return u_star, f_star, df
        else:
            u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:,0:1], self.t_u_tf: X_star[:,1:2]})  
            f_star, u_t, u, v, x, t = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
            df['du_physics'] = u_t.flatten()
            df['u_physics'] = u.flatten()
            df['dv_physics'] = v.flatten()
            df['f_star_physics'] = f_star.flatten()
            return u_star, f_star, df
    

if __name__ == "__main__": 
     
    nu = 0.01/np.pi
    noise = 0.00       

    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat(r'burgers_shock.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t) 
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]

    lb = xx1.min(0)
    ub = xx3.max(0)
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
        
    normal_model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu, 'normal')
    
    start_time = time.time()                
    normal_model.train()
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

    df = pd.DataFrame()
    u_pred, f_pred, df = normal_model.predict(X_star, df)
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('Normal Error u: %e' % (error_u))  

    
    physics_model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu, 'physics')
    
    start_time = time.time()                
    physics_model.train()
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    u_pred, f_pred, df = physics_model.predict(X_star, df)
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star, 2)
    print('Physics Error u: %e' % (error_u)) 
    
    df['u_actual'] = u_star.flatten()
    df['actual_normal'] = df.apply(lambda x: difference(x.u_actual, x.u_normal), axis=1)
    df['actual_physics'] = df.apply(lambda x: difference(x.u_actual, x.u_physics), axis=1)
    df['du_dv_normal'] = df.apply(lambda x: difference(x.du_normal, x.dv_normal), axis=1)
    df['du_dv_physics'] = df.apply(lambda x: difference(x.du_physics, x.dv_physics), axis=1)

    df.to_csv('data_for_plot.csv')

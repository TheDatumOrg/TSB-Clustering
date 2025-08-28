'''
Created on 2020年7月14日

@author: Shaoyu Dou
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from .rnns import dilated_encoder, single_layer_decoder
from .classification import classifier
from .kmeans import kmeans
from .utils import truncatedSVD, ri_score, cluster_using_kmeans, nmi_score
from tqdm import tqdm

class DTCR():
    def __init__(self, opts):
        self.opts = opts
        tf.reset_default_graph()
        self.creat_network()
        self.init_optimizers()

    def creat_network(self):
        opts = self.opts
        self.encoder_input = tf.placeholder(dtype=tf.float32, shape=(None, opts['input_length'], 1), name='encoder_input')
        self.decoder_input = tf.placeholder(dtype=tf.float32, shape=(None, opts['input_length'], 1), name='decoder_input')
        self.classification_labels = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='classification_labels')
        
        # seq2seq
        with tf.variable_scope('seq2seq'):
            self.D_ENCODER = dilated_encoder(opts)
            self.h = self.D_ENCODER.encoder(self.encoder_input)
            
            self.S_DECOER = single_layer_decoder(opts)
            recons_input = self.S_DECOER.decoder(self.h, self.decoder_input)
            
            self.h_fake, self.h_real = tf.split(self.h, num_or_size_splits=2, axis=0)
            
        # classifier
        with tf.variable_scope('classifier'):
            self.CLS = classifier(opts)
            output_without_softmax = self.CLS.cls_net(self.h)
        
        # K-means
        with tf.variable_scope('kmeans'):
            self.KMEANS = kmeans(opts)
            # update F
            kmeans_obj = self.KMEANS.kmeans_optimalize(self.h_real)
        
        # L-reconstruction
        self.loss_reconstruction = tf.losses.mean_squared_error(self.encoder_input, recons_input)
        # L-classification
        self.loss_classification = tf.losses.softmax_cross_entropy(self.classification_labels, output_without_softmax)
        # L-kmeans
        self.loss_kmeans = kmeans_obj
        
        
    def init_optimizers(self):
        lambda_1 = self.opts['lambda']
        
        # vars
        seq2seq_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='seq2seq')
        cls_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
        end2end_vars = seq2seq_vars + cls_vars
        
        kmeans_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='kmeans')
        
        # loss
        self.loss_dtcr = self.loss_reconstruction + self.loss_classification + lambda_1 * self.loss_kmeans
        
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-3)
        
        # update vars
        self.train_op = optimizer.minimize(self.loss_dtcr, var_list=end2end_vars)
    
    def update_kmeans_f(self, train_h):    
        new_f = truncatedSVD(train_h, self.opts['cluster_num'])
        self.KMEANS.update_f(new_f)        
        
    def train(self, cls_data, cls_label,  train_data, train_label):
        opts = self.opts
        
        # processing data and label
        cls_data = np.expand_dims(cls_data, axis=2)        
        cls_label_ = np.zeros(shape=(cls_label.shape[0], len(np.unique(cls_label))))
        cls_label_[np.arange(cls_label_.shape[0]), cls_label] = 1

        # session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        #print('vars_num: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        if cls_data.shape[0]//200 == 0:
            feed_d = {self.encoder_input: cls_data,
                    self.decoder_input: np.zeros_like(cls_data),
                    self.classification_labels: cls_label_} 

            # init:        
            train_h = sess.run(self.h_real, feed_dict=feed_d)
            self.update_kmeans_f(train_h) 
        else:
            for i in tqdm(range(cls_data.shape[0]//200)):
                feed_d = {self.encoder_input: cls_data[i*200:(i+1)*200],
                    self.decoder_input: np.zeros_like(cls_data)[i*200:(i+1)*200],
                    self.classification_labels: cls_label_[i*200:(i+1)*200]} 

                # init:        
                train_h = sess.run(self.h_real, feed_dict=feed_d)
                self.update_kmeans_f(train_h)
        
        # train
        train_list = []
        test_list = []
        best_indicator = float('inf')
        best_epoch = -1
        for epoch in range(opts['max_iter']):
            ep_loss = 0
            if cls_data.shape[0]//200 == 0:
                feed_d = {self.encoder_input: cls_data,
                        self.decoder_input: np.zeros_like(cls_data),
                        self.classification_labels: cls_label_} 

                _, loss, l_recons, l_cls, l_kmeans = sess.run([self.train_op, self.loss_dtcr, self.loss_reconstruction, self.loss_classification, self.loss_kmeans], feed_dict=feed_d)
                ep_loss = ep_loss + loss
            else:
                for k in range(cls_data.shape[0]//200):
                    feed_d = {self.encoder_input: cls_data[k*200:(k+1)*200],
                        self.decoder_input: np.zeros_like(cls_data)[k*200:(k+1)*200],
                        self.classification_labels: cls_label_[k*200:(k+1)*200]} 
                    _, loss, l_recons, l_cls, l_kmeans = sess.run([self.train_op, self.loss_dtcr, self.loss_reconstruction, self.loss_classification, self.loss_kmeans], feed_dict=feed_d)
                    ep_loss = ep_loss + loss
                    print('loss: {}, l_recons: {}, l_cls: {}, l_kmeans: {}, epoch: {}'.format(loss, l_recons, l_cls, l_kmeans, epoch))
            
            if epoch % opts['alter_iter'] == 0:
                if cls_data.shape[0]//200 == 0:
                    feed_d = {self.encoder_input: cls_data,
                        self.decoder_input: np.zeros_like(cls_data),
                        self.classification_labels: cls_label_}       
                    train_h = sess.run(self.h_real, feed_dict=feed_d)
                    self.update_kmeans_f(train_h)
                else:
                    for k in tqdm(range(cls_data.shape[0]//200)):
                        feed_d = {self.encoder_input: cls_data[k*200:(k+1)*200],
                        self.decoder_input: np.zeros_like(cls_data)[k*200:(k+1)*200],
                        self.classification_labels: cls_label_[k*200:(k+1)*200]}  
                        train_h = sess.run(self.h_real, feed_dict=feed_d)
                        self.update_kmeans_f(train_h)
            
            if cls_data.shape[0]//200 == 0:
                ep_loss = ep_loss
            else:
                ep_loss = ep_loss / (cls_data.shape[0]//200)

            train_embedding = self.test(sess, train_data)
            pred_train, pred_inertia = cluster_using_kmeans(train_embedding, opts['cluster_num'])

            if best_indicator > ep_loss:
                best_indicator = ep_loss
                best_pred = pred_train
                best_inertia = pred_inertia

        sess.close()
        return best_inertia, best_pred, pred_inertia, pred_train
                    
    def test(self, sess, test_data):
        test_data = np.expand_dims(test_data, axis=2)
        feed_d = {self.encoder_input: test_data}
        h = sess.run(self.h, feed_dict=feed_d)
        return h
'''
Created on 2020年7月14日

@author: Shaoyu Dou
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class classifier():
    def __init__(self, opts):
        self.hidden_units = opts['classifier_hidden_units']
        
    def cls_net(self, inputs):
        out = inputs
        for units in self.hidden_units:
            out = tf.layers.dense(out, units=units, use_bias=False)# (7)
        return out
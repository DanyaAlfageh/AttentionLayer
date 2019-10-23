import glob
import numpy as np
#from tqdm import tqdm
import os, json, codecs
from collections import Counter
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
from keras.layers import *
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import pandas as pd
import random
import keras

from keras import backend as K
from keras.engine.topology import Layer

#1.GlobalMaxPooling1D:
#在steps维度（也就是第二维）对整个数据求最大值。
#比如说输入数据维度是[10, 4, 10]，那么进过全局池化后，输出数据的维度则变成[10, 10]。
#2.MaxPooling1D：
#也是在steps维度（也就是第二维）求最大值。但是限制每一步的池化的大小。 比如，输入数据维度是[10, 4, 10]，池化层大小pooling_size=2，步长stride=1，那么经过MaxPooling(pooling_size=2, stride=1)后，输出数据维度是[10, 3, 10]。


class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('An attention layer should be called '
                             'on a list of 2 inputs.')
        if not input_shape[0][2] == input_shape[1][2]:
            raise ValueError('Embedding sizes should be of the same size')

        self.kernel = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        #inputs[0]  shapel = [bacth_size, col1, emb]
        #inputs[1]  shape = [batch_size, col2, emb]
        a = K.dot(inputs[0], self.kernel)  # shape = [batch_size, col1, emb]
        y_trans = K.permute_dimensions(inputs[1], (0,2,1)) #shape = [batch_size, emb, col2]
        b = K.batch_dot(a, y_trans, axes=[2,1])  #shape = [batch_size, col1, col2]
        attention = K.tanh(b) # shape = [batch_size, col1, col2]
        left_att = K.max(attention, axis = 2) # shape = (batch_size, col1)
        left_att = K.softmax(left_att) # shape = (batch_size, col1)
        out_1 = K.batch_dot(left_att, inputs[0], axes = 1)  #shape = (batch_size, emb)
        right_att = K.permute_dimensions(attention, (0, 2, 1)) #shape = (batch_size, col2, col1)
        right_att = K.max(right_att, axis = 2) #shape = (batch_size, col2)
        right_att = K.softmax(right_att) #shape = (batch_size, col2)
        out_2 = K.batch_dot(right_att, inputs[1], axes = 1) #shape = (batch_size, emb)
        return [out_1, out_2]
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][2]), (input_shape[1][0], input_shape[1][2])]

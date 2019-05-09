#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

class TRNNConfig(object):
    """RNN配置参数"""

    embedding_dim = 64     
    seq_length = 600       
    num_classes = 10        
    vocab_size = 5000       

    num_layers= 2           
    hidden_dim = 128       
    rnn = 'gru'     
    # 隐藏层层数为2
    # 选择lstm 或 gru

    dropout_keep_prob = 0.8 
    learning_rate = 1e-3   

    batch_size = 128         
    num_epochs = 10        

    print_per_batch = 100    
    save_per_batch = 10      

class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   
            return tf.nn.rnn_cell.LSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  
            return tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)

        def dropout(): 
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        # 为每一个rnn核后面加一个dropout层

        with tf.device('/gpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            # 堆叠了2层的RNN模型。

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  
            # 取最后一个时序输出作为结果，也就是最后时刻和第2层的LSTM或GRU的隐状态。

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  

        with tf.name_scope("optimize"):
            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
           
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
    
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

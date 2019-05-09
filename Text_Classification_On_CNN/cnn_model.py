# coding: utf-8

import tensorflow as tf

""" CNN配置参数 """
class TCNNConfig(object):
   
    embedding_dim = 64  
    seq_length = 600  
    num_classes = 10  
    num_filters = 256     
    kernel_size = 5  
    
    # 输入层的维度是（600，64，1）
    # 卷积核数目是256，也就是提取的特征数量是256种，决定了卷积层的通道数为256
    # 卷积核的维度是（5，64）
    # 卷积核尺寸为5，也就是一次卷多少个词，这里卷5个词，那么是5-gram。
    # 卷积层的维度是（600-5+1，1，256），如果Stride=1, n-gram=5。256是由卷积核的个数决定的。
    # 卷积层的通道数等于卷积核的个数，卷积核的通道数等于输入层的通道数。
    
    vocab_size = 5000  
    hidden_dim = 128  

    dropout_keep_prob = 0.5  
    learning_rate = 1e-3  

    batch_size = 64  
    num_epochs = 10  
    print_per_batch = 100  
    save_per_batch = 10  
   # 每100批输出一次结果。
   # 每10批存入tensorboard。

"""文本分类，CNN模型"""
class TextCNN(object):

    def __init__(self, config):
        self.config = config

        # None是bitch_size,input_x是（64，600）的维度，input_y的维度是（64，10）        
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        
        with tf.device('/gpu:0'):          
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])   
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            
            # 指定在第1块gpu上运行，如果指定是cpu则（'/cpu:0'）
            # 获取已经存在的变量，不存在则创建并随机初始化。这里的词向量是随机初始化的，embedding的维度是（5000，64）
            # embedding_inputs.shape=(64，600,64)
            
        with tf.name_scope("cnn"):

            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            
            # 使用一维卷积核进行卷积，因为卷积核的第二维与词向量维度相同，只能沿着行向下滑动。
            # 输入样本维度是(600,64)，经过(5,64)的卷积核卷积后得到(596,1)的向量（600-5+1=596），默认滑动为1步。
            # 由于有256个过滤器，于是得到256个(596,1)的向量。
            # 结果显示为(None,596,256)   
            # 用最大池化方法，按行求最大值，conv.shape=[Dimension(None), Dimension(596), Dimension(256)],留下了第1和第3维。
            # 取每个向量(596,1)中的最大值，然后就得到了256个最大值，
            # gmp.shape=（64，256） 

            with tf.name_scope("score"):
                
                fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                fc = tf.nn.relu(fc)
                # 全连接层，后面接dropout以及relu激活
                # 神经元的个数为128个，gmp为(64,256),经过这一层得到fc的维度是(64，128）    
                
                self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
                self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  
                # softmax得到的输出为[Dimension(None), Dimension(10)],是10个类别的概率
                # 然后再从中选出最大的那个值的下标，如[9,1,3...]
                # 最后得到的是（64，1）的列向量，即64个样本对应的类别。                
                
            with tf.name_scope("optimize"):
  
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
                self.loss = tf.reduce_mean(cross_entropy)
                self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
                # tf.reduce_mean(input_tensor,axis)用于求平均值，这里是求64个样本的交叉熵损失的均值。
    
            with tf.name_scope("accuracy"):

                correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
                self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                
                # 准确率的计算，tf.equal对内部两个向量的每个元素进行对比，返回[True,False,True,...]这样的向量
                # 也就是对预测类别和标签进行对比，self.y_pred_cls形如[9,0,2,3,...]
                # tf.cast函数将布尔类型转化为浮点型，True转为1.，False转化为0.，返回[1,0,1,...]
                # 然后对[1,0,1,...]这样的向量求均值，恰好就是1的个数除以所有的样本，恰好是准确率。                
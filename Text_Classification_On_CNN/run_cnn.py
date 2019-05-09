#!/usr/bin/python
# -*- coding: utf-8 -*-

#哪怕程序在2.7的版本运行，也可以用print()这种语法来打印。
from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab
# 数据处理模块为cnews_loader
# 模型搭建模块为cnn_model

base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  
#这里说是保存路径，其实这个“best_validation”是保存的文件的名字的开头，比如保存的一个文件是“best_validation.index”


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

   # round函数是对浮点数四舍五入为int，注意版本3中round(0.5)=0,round(3.567,2)=3.57。
   # timedelta是用于对间隔进行规范化输出，间隔10秒的输出为：00:00:10    


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

""" 评估在某一数据上的准确率和损失 """
def evaluate(sess, x_, y_):

    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        
        batch_len = len(x_batch)      
        feed_dict = feed_data(x_batch, y_batch, 1.0)    
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    
    return total_loss / data_len, total_acc / data_len
        
        # 1.0是dropout值，在测试和验证时不需要舍弃
        # 把feed_dict的数据传入去计算model.loss,是求出了128个样本的平均交叉熵损失
        # 把平均交叉熵和平均准确率分别乘以128个样本得到总数，不断累加得到10000个样本的总数。
        # 求出10000个样本的平均交叉熵，和平均准确率。

def train():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = 'tensorboard/textcnn'
  
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
        
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    # 用到 tf.summary 中的方法保存日志数据，用于tensorboard可视化操作。
    # 用 tf.summary.scalar 保存标量，一般用来保存loss，accuary，学习率等数据    
    
  
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    # 使用 tf.summaries.merge_all() 对所有的汇总操作进行合并
    # 将数据写入本地磁盘: tf.summary.FileWriter

 
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  
    best_acc_val = 0.0  
    last_improved = 0  
    require_improvement = 1000  
    # 如果超过1000轮未提升，提前结束训练，防止过拟合。

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)      
            
            if total_batch % config.save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  
        if flag:  
            break


def test():
    
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
      
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  
    # 在保存和恢复模型时都需要首先运行这一行：tf.train.Saver()，而不是只有保存时需要。

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
     # 返回了10000个总测试样本的平均交叉熵损失和平均准率。
    
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test) 
    num_batch = int((data_len - 1) / batch_size) + 1 
    y_test_cls = np.argmax(y_test, 1) 
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  
    for i in range(num_batch):  
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0 
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
        # 测试的时候不需要dropout神经元。

  
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
    # 可以得到准确率 、召回率和F1_score

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):       
        build_vocab(train_dir, vocab_dir, config.vocab_size)    
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    
    # 如果不存在词汇表，重建，值为False时进行重建。
    # 字典中有5000个字。
    # 返回categories：['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # 以及cat-to-id：{'体育': 0, '财经': 1, '房产': 2    , '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
    # 输出word：['<PAD>', '，', '的', '。', '一', '是', '在', '0', '有', '不', '了', '中', '1', '人', '大', '、', '国', '', '2', ...]
    # 输出word_to_id:{'<PAD>': 0, '，': 1, '的': 2, '。':     3, '一': 4, '是': 5,...}，里面还包含了一些字号、逗号和数字作为键值。  
    
    config.vocab_size = len(words)
    model = TextCNN(config)
    option='train'
    if option == 'train':
        train()
    else:
        test()

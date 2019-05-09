# CNN和RNN中文文本分类模型

## 环境

    Python 3.6.8
    TensorFlow 1.12
    numpy
    scikit-learn

## 数据集

本次训练使用了其中的10个分类，每个分类6500条数据。

类别如下：

体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐

这个子集可以在此下载：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud, 下载后放在 data这个目录下即可。

训练集、验证集和测试集的文件如下：
    cnews.train.txt: 训练集(50000条)
    cnews.val.txt: 验证集(5000条)
    cnews.test.txt: 测试集(10000条)
    
## CNN模型

第一个文件夹为CNN模型的数据和代码。

模块分为四个：数据处理模块（cnews_loader.py）,模型构建模块（cnn_model.py）,模型运行模块（run_cnn.py）,模型预测模块（predict.py）

## RNN模型

第二个文件夹为RNN模块的数据和代码。

模块分为四个：数据处理模块（cnews_loader.py）,模型构建模块（rnn_model.py）,模型运行模块（run_rnn.py）,模型预测模块（predict.py）

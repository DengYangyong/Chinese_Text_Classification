#coding: utf-8
import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False
    # 判断软件的版本，如果版本为3.6.5，那么sys.version_info的输出为：sys.version_info(major=3, minor=6, micro=5)。

"""如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
def native_word(word, encoding='utf-8'):
    if not is_py3:
        return word.encode(encoding)
    else:
        return word

"""is_py3函数当版本为3时返回True，否则返回False。if not 后面的值为False则将“utf-8”编码转换为'unicode'."""
def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

""" 常用文件操作，可在python2和python3间切换."""
def open_file(filename, mode='r'):
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

""" 读取文件数据"""
def read_file(filename): 
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:   
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels
      #  line.strip().split('\t')的输出为两个元素的列表：['体育', '黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯...']。
      # 注意这个list()函数，把一段文字转化为了列表，元素为每个字和符号：['黄', '蜂', 'v', 's', '湖', '人', '首', '发', '：', '科', '比',...]
      # contents的元素为每段新闻转化成的列表：[['黄', '蜂', 'v', 's', '湖', '人', '首', '发', '：', '科', '比',...],[],...]
      # labels为['体育', '体育',...]

"""根据训练集构建词汇表，存储"""
def build_vocab(train_dir, vocab_dir, vocab_size=5000): 
    data_train, _ = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

'''读取词汇表'''
def read_vocab(vocab_dir):    
    with open_file(vocab_dir) as fp:   
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id
# readlines()读取所有行然后把它们作为一个字符串列表返回:['头\n', '天\n', ...]。strip()函数去掉"\n"。
# words: ['<PAD>', '，', '的', '。', '一', '是', '在', '0', '有',...]
# word_to_id：{'<PAD>': 0, '，': 1, '的': 2, '。': 3, '一': 4, '是': 5,..}，每个类别对应的value值为其索引ID

"""读取分类目录"""
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id
   # cat_to_id的输出为：{'体育': 0, '财经': 1, '房产': 2, '家居': 3,...}，每个类别对应的value值为其索引ID.
   
""" 将id表示的内容转换为文字 """
def to_words(content, words):
    return ''.join(words[x] for x in content)

""" 将文件转换为id表示,进行pad """
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    #contents的形式为：[['黄', '蜂', 'v', 's', '湖', '人',...],[],[],...]，每一个元素是一个列表，该列表的元素是每段新闻的字和符号。
    #labels的形式为：['体育', '体育', '体育', '体育', '体育', ...]    
    
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  
    return x_pad, y_pad

   # word_to_id是一个字典：{'<PAD>': 0, '，': 1, '的': 2, '。': 3, '一': 4, '是': 5,...}
   # 对于每一段新闻转化的字列表，把每个字在字典中对应的索引找到：
   # data_id: 将[['黄', '蜂', 'v', 's', '湖', '人',...],[],[],...] 转化为 [[387, 1197, 2173, 215, 110, 264,...],[],[],...]的形式
   # label_id : ['体育', '体育', '体育', '体育', '体育', ...] 转化为[0, 0, 0, 0, 0, ...]
   # data_id的行数为50000，即为新闻的条数，每个元素为由每段新闻的字的数字索引构成的列表；
   # data_id长这样：[[387, 1197, 2173, 215, 110, 264,...],[],[],...]
   # 由于每段新闻的字数不一样，因此每个元素（列表）的长度不一样，可能大于600，也可能小于600，需要统一长度为600。
   # 使用keras提供的pad_sequences来将文本pad为固定长度，x_pad的形状为（50000，600）.
   # label_id是形如[0, 0, 0, 0, 0, ...]的整形数组，cat_to_id是形如{'体育': 0, '财经': 1, '房产': 2, '家居': 3,...}的字典
   # to_categorical是对标签进行one-hot编码，num-classes是类别数10，y_pad的维度是（50000，10）
   
"""生成批次数据"""
def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1    
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    
    # 样本长度为50000
    # int()可以将其他类型转化为整型，也可以用于向下取整，这里为782.
    # indices元素的范围是0-49999，形如[256,189,2,...]的拥有50000个元素的列表
    # 用indices对样本和标签按照行进行重新洗牌，接着上面的例子，把第256行(从0开始计)放在第0行，第189行放在第1行.
    
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
    
        # i=780时，end_id=781*64=49984;
        # 当i=781时，end_id=50000，因为782*64=50048>50000,所以最后一批取[49984:50000]    
        # yield是生成一个迭代器，用for循环来不断生成下一个批量。
        # 为了防止内存溢出，每次只取64个，内存占用少。        
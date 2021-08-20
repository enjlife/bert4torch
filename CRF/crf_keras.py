# -*- coding:utf-8 -*-

from keras.layers import Layer
import keras.backend as K

# This script is from `https://github.com/bojone/crf/`


class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        这块比较核心，更详细的推导可以看 `https://blog.csdn.net/weixin_40548136/article/details/119620027`
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]  # (batch_size, 4) (batch_size, 1)
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        # 指数乘法，e^(Z_t + h_{t+1}) (batch_size, output_dim, 1) + (1, output_dim, output_dim) -> (batch_size, output_dim, output_dim)
        # 针对当前k个终点累加, 所以累加的维度为1 (batch_size, output_dim, output_dim) -> ((batch_size, output_dim)
        outputs = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)
        outputs = outputs + inputs
        # mask=1 正常输出；为0，输出上一个状态
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs * labels, 2), 1, keepdims=True)  # 逐标签得分 padding [0,0,0,0,1] -> [0,0,0,0]
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        # 两个错位labels，相乘后需要计算转移矩阵的部分为1负责从转移矩阵中抽取目标转移得分
        # 举个例子[[1,0,0]]和[[0,1,0]] expand后相乘得到3x3 第0行1列的值为1，表示取转移分
        labels = labels1 * labels2  # （batch_size，seq_len-1,output_dim,output_dim）
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans * labels, [2, 3]), 1, keepdims=True)
        return point_score + trans_score  # 两部分得分之和

    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred):  # y_true需要是one hot形式
        if self.ignore_last_label:
            mask = 1 - y_true[:, :, -1:]  # (batch_size，max_len,output_dim) -> (batch_size，max_len,1) padding位置为0
        else:
            mask = K.ones_like(y_pred[:, :, :1])  # (batch_size，max_len,1) 都为1
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
        init_states = [y_pred[:, 0]]  # 初始状态
        y_pred = K.concatenate([y_pred, mask])
        log_norm, _, _ = K.rnn(self.log_norm_step, y_pred[:, 1:], init_states)  # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        return log_norm - path_score  # 即-log(分子/分母)

    def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1 - y_true[:, :, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)

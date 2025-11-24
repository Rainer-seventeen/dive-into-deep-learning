"""
从零开始实现线性回归
"""

import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    """生成 y=Wx+b+Noise"""
    # 生成随机数： 均值为 0，方差为 1，形状为 [num_examples, len(w)]
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b    # 矩阵计算
    y += torch.normal(0, 0.01, y.shape)   # 随机噪声
    return X, torch.reshape(y, (-1, 1))   #随机的x 列向量的y

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机重写顺序
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        # 最后一个批量不够，取到数据最大值
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        
        # yield 返回值函数转为生成器，每次生成数据从上一个数据继续
        yield features[batch_indices], labels[batch_indices]

def linear_reg(X, w, b):
    """线性模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方误差"""
    # ** 是次方的意思
    return (y_hat - y.reshape(y_hat.shape)) **2 / 2

def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    因为计算 loss 的时候没有均值，因此这里加上
    """
    # 更新的时候关掉梯度
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 计算完毕，清空梯度


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    features, labels = synthetic_data(true_w, true_b, 1000)
    
    batch_size  = 10

    # 每一次从中拿出一部分
    # for X, y in data_iter(batch_size, features, labels):
    #     print(X, '\n', y)
    #     break

    # 随机初始化参数
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 开始训练 这是一个通用的模板
    lr = 3e-2
    num_epoch = 3
    net = linear_reg
    loss = squared_loss
    
    for epoch in range(num_epoch):
        # 抽取一个 batch 的数据
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y) # 计算网络和实际数值之间的误差
            l.sum().backward() # 表示进行一次梯度计算，结果存放在 .grad 中
            sgd([w, b], lr, batch_size)
            
            with torch.no_grad():
                # 验证一下 w 和 b 计算出的 loss
                train_loss = loss(net(features, w, b), labels)
                # train_loss 是一个 tensor [1000, 1] 对他求和再取平均值
                print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

    print(f"w 误差 {true_w - w.reshape(true_w.shape)}")
    print(f"b 误差 {true_b - b}")

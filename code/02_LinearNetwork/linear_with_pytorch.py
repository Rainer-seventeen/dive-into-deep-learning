"""
使用 pytorch API 来进行线性回归
"""
import torch
from torch.utils import data
from torch import nn

def synthetic_data(w, b, num_examples):
    """生成 y=Wx+b+Noise"""
    # 生成随机数： 均值为 0，方差为 1，形状为 [num_examples, len(w)]
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b    # 矩阵计算
    y += torch.normal(0, 0.01, y.shape)   # 随机噪声
    return X, torch.reshape(y, (-1, 1))   #随机的x 列向量的y


def load_array(data_array, batch_size, is_train=True):
    """pytorch 数据读取"""
    # * 表示把元组直接展开为不同的数据
    dataset = data.TensorDataset(*data_array)
    # 如果是 train 就进行打乱
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2

    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    # iter 从一个可迭代的对象 iterable 中创建一个 迭代器对象 iterator
    # next(iterator) 会从中取出下一个元素
    iterator = iter(data_iter)

    net = nn.Sequential(
        nn.Linear(2, 1)
    )
    # 初始化参数, 有下划线表示原地更改数值
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0) 

    learning_rate = 3e-2
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    num_epoch = 3
    for epoch in range(num_epoch):
        # 抽取一个 batch 的数据
        for X, y in iterator:
            l = loss(net(X), y) # 计算网络和实际数值之间的误差
            trainer.zero_grad() # 清空梯度
            l.backward() # 进行一次梯度计算
            trainer.step()
            
            with torch.no_grad():
                # 验证一下 w 和 b 计算出的 loss
                train_loss = loss(net(features), labels)
                # train_loss 是一个 tensor [1000, 1] 对他求和再取平均值
                print(f'epoch {epoch + 1}, loss {train_loss:f}')


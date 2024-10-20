import torch as t
from torch.autograd import Variable as V
# 不是 jupyter 运行请注释掉下面一行，为了 jupyter 显示图片
from matplotlib import pyplot as plt
from IPython import display

t.manual_seed(1000)  # 随机数种子


def get_fake_data(batch_size=8):
    """产生随机数据：y = x * 2 + 3，同时加上了一些噪声"""
    x = t.rand(batch_size, 1) * 20
    y = x * 2 + (1 + t.randn(batch_size, 1)) * 3  # 噪声为 |3-((1 + t.randn(batch_size, 1)) * 3)|

    return x, y


# 查看 x，y 的分布情况
x, y = get_fake_data()
plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
plt.show()
# 随机初始化参数
w = V(t.rand(1, 1), requires_grad=True)
b = V(t.zeros(1, 1), requires_grad=True)

lr = 0.001  # 学习率

for i in range(8000):
    x, y = get_fake_data()
    x, y = V(x), V(y)

    # forwad：计算 loss
    y_pred = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (y_pred - y)**2
    loss = loss.sum()

    # backward：自动计算梯度
    loss.backward()

    # 更新参数
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)

    # 梯度清零，不清零则会进行叠加，影响下一次梯度计算
    w.grad.data.zero_()
    b.grad.data.zero_()

    if i % 1000 == 0:
        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 20, dtype=t.float).view(-1, 1)
        y = x.mm(w.data) + b.data.expand_as(x)
        plt.plot(x.numpy(), y.numpy(), color='red')  # 预测效果

        x2, y2 = get_fake_data(batch_size=20)
        plt.scatter(x2.numpy(), y2.numpy(), color='blue')  # 真实数据

        plt.xlim(0, 20)
        plt.ylim(0, 41)
        plt.show()
        # plt.pause(0.5)
        # break  # 注销这一行，可以看到动态效果
import numpy as np
import matplotlib.pyplot as plt


def guess_func_with_param(x, params):
    """泰勒展开"""
    y = 0
    for i, param in enumerate(params):
        y += param * x ** i
    return y


def sgd_update(params, grads, lr):
    """SGD更新"""
    for i in range(len(params)):
        params[i] -= lr * grads[i]


def train(x, y, params, lr):
    """训练"""
    # 预测y值
    y_pred = guess_func_with_param(x, params)
    # 计算差距
    loss = np.mean((y - y_pred) ** 2)
    # 计算梯度
    grads = []
    for i in range(len(params)):
        # 偏导数
        grad = np.mean((y_pred - y) * x ** i)
        grads.append(grad)
    # 更新参数
    sgd_update(params, grads, lr)
    return loss


def train_loop(x, y, params, lr, epochs):
    """训练循环"""
    for epoch in range(epochs):
        loss = train(x, y, params, lr)
        print('epoch: {}, loss: {}'.format(epoch, loss))


def real_func(x):
    """真实函数"""
    return np.sin(2 * np.pi * x)


def generate_data():
    """生成数据"""
    x = np.linspace(-1, 1, 1000)
    y = real_func(x)
    return x, y


def draw_func(func_list):
    """绘制函数图像"""
    x = np.linspace(-1, 1, 100)
    for func in func_list:
        y = func(x)
        # 颜色随机
        color = np.random.rand(3)
        plt.plot(x, y, color=color)
    plt.show()


def main():
    """主函数"""
    # 生成数据
    x, y = generate_data()
    # 随机初始化参数, 参数有n个, 表示泰勒展开到n次
    params = np.random.randn(100)
    print('params: {}'.format(params))
    # 训练
    train_loop(x, y, params, lr=1.2, epochs=10000)
    # 打印参数
    print('params: {}'.format(params))

    def guess_func(_x):
        return guess_func_with_param(_x, params)

    # 绘制函数图像
    draw_func([real_func, guess_func])


if __name__ == '__main__':
    main()

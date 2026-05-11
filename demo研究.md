# demo.ipynb 代码分析

`demo.ipynb` 是一个 MicroGrad 二分类训练演示。它使用 `make_moons` 生成二维数据集，训练一个小型 MLP，并绘制最终的分类决策边界。

## 1. 整体作用

这个 notebook 展示了如何使用项目中的 `micrograd.engine.Value` 和 `micrograd.nn.MLP`：

- 构造一个二维二分类数据集
- 初始化一个多层感知机
- 使用 hinge loss / max-margin loss 训练模型
- 通过 MicroGrad 的反向传播计算梯度
- 使用 SGD 更新参数
- 可视化训练后的决策边界

相关核心文件：

- `micrograd/engine.py`：实现标量自动求导节点 `Value`
- `micrograd/nn.py`：实现 `Neuron`、`Layer`、`MLP`

## 2. 导入依赖

notebook 开头导入：

```python
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

之后导入本项目中的核心类：

```python
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
```

其中：

- `Value` 是 MicroGrad 的标量自动求导对象
- `Neuron` 表示一个神经元
- `Layer` 表示一层神经元
- `MLP` 表示多层感知机

## 3. 固定随机种子

```python
np.random.seed(1337)
random.seed(1337)
```

这里同时固定 NumPy 和 Python 标准库 `random` 的随机种子，使数据生成、模型初始化和训练过程更容易复现。

## 4. 构造数据集

```python
from sklearn.datasets import make_moons, make_blobs
X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1
```

`make_moons` 会生成二维月牙形二分类数据。

原始标签是 `0` 和 `1`，代码将其转换为 `-1` 和 `1`：

```text
0 -> -1
1 -> 1
```

这样做是为了配合后面的 SVM 风格 max-margin loss：

```text
max(0, 1 - y * score)
```

随后代码用散点图展示数据分布：

```python
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
```

## 5. 初始化模型

```python
model = MLP(2, [16, 16, 1])
print(model)
print("number of parameters", len(model.parameters()))
```

模型结构是：

- 输入维度：2
- 第一隐藏层：16 个 ReLU 神经元
- 第二隐藏层：16 个 ReLU 神经元
- 输出层：1 个线性神经元

对应 `micrograd/nn.py` 中的实现：

```python
class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
```

最后一层 `nonlin=False`，所以输出层不使用 ReLU。

notebook 中输出参数量为：

```text
number of parameters 337
```

参数量计算如下：

```text
第一层：16 * (2 weights + 1 bias) = 48
第二层：16 * (16 weights + 1 bias) = 272
输出层：1 * (16 weights + 1 bias) = 17
总计：48 + 272 + 17 = 337
```

## 6. loss 函数

notebook 中定义了一个 `loss()` 函数：

```python
def loss(batch_size=None):
    
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    
    # forward the model to get scores
    scores = list(map(model, inputs))
    
    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)
```

### 6.1 批数据选择

如果不传 `batch_size`，使用完整数据集：

```python
Xb, yb = X, y
```

如果传入 `batch_size`，则随机抽取一个 mini-batch：

```python
ri = np.random.permutation(X.shape[0])[:batch_size]
Xb, yb = X[ri], y[ri]
```

当前 notebook 的训练循环没有传 `batch_size`，所以实际使用的是全量 batch。

### 6.2 转成 Value

```python
inputs = [list(map(Value, xrow)) for xrow in Xb]
```

MicroGrad 的计算图基于标量 `Value`，所以每个二维输入点都会被转换成两个 `Value` 对象。

例如一个输入点：

```text
[x1, x2]
```

会变成：

```text
[Value(x1), Value(x2)]
```

### 6.3 前向传播

```python
scores = list(map(model, inputs))
```

每个样本经过 MLP 后得到一个标量分数 `score`。

分类规则是：

```text
score > 0  -> 正类
score <= 0 -> 负类
```

### 6.4 Max-margin loss

```python
losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
data_loss = sum(losses) * (1.0 / len(losses))
```

这等价于：

```text
loss = max(0, 1 - y * score)
```

含义：

- 如果 `y * score >= 1`，说明分类正确且 margin 足够，loss 为 0
- 如果 `y * score < 1`，说明分类错误或 margin 不够，产生惩罚

这里的 `.relu()` 被用来实现 `max(0, x)`。

### 6.5 L2 正则化

```python
alpha = 1e-4
reg_loss = alpha * sum((p*p for p in model.parameters()))
total_loss = data_loss + reg_loss
```

L2 正则会惩罚过大的参数，帮助控制模型复杂度。

总损失为：

```text
total_loss = data_loss + reg_loss
```

### 6.6 准确率

```python
accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
```

这里使用 `scorei.data` 取出普通数值，而不是继续参与计算图。

判断方式：

```text
真实标签是否为正类 == 模型分数是否大于 0
```

notebook 初始输出：

```text
Value(data=0.8958441028683222, grad=0) 0.5
```

也就是：

- 初始 loss 约为 `0.8958`
- 初始 accuracy 为 `50%`

## 7. 训练过程

训练代码：

```python
for k in range(100):
    
    # forward
    total_loss, acc = loss()
    
    # backward
    model.zero_grad()
    total_loss.backward()
    
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
```

每一步包括：

1. 前向传播，计算 `total_loss`
2. 调用 `model.zero_grad()` 清空参数梯度
3. 调用 `total_loss.backward()` 执行反向传播
4. 用 SGD 更新所有参数
5. 打印当前 loss 和 accuracy

学习率为：

```python
learning_rate = 1.0 - 0.9*k/100
```

它会从 `1.0` 线性下降，训练到第 99 步时约为：

```text
1.0 - 0.9 * 99 / 100 = 0.109
```

notebook 保存的最后几步输出：

```text
step 96 loss 0.010980043387921506, accuracy 100.0%
step 97 loss 0.010979446081684675, accuracy 100.0%
step 98 loss 0.010978888332907229, accuracy 100.0%
step 99 loss 0.010978370135492717, accuracy 100.0%
```

说明模型已经在训练集上达到 `100%` 准确率。

## 8. 反向传播来自哪里

`total_loss.backward()` 调用的是 `micrograd/engine.py` 中 `Value.backward()`：

```python
def backward(self):

    # topological order all of the children in the graph
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # go one variable at a time and apply the chain rule to get its gradient
    self.grad = 1
    for v in reversed(topo):
        v._backward()
```

其核心逻辑是：

1. 从最终 loss 节点出发，递归遍历计算图
2. 得到拓扑排序
3. 将最终 loss 的梯度设为 `1`
4. 按反向拓扑顺序调用每个节点的 `_backward()`
5. 使用链式法则累积每个参数的梯度

这就是 MicroGrad 的核心：基于动态构建的标量计算图做反向模式自动微分。

## 9. 可视化决策边界

最后一个代码单元生成二维网格，并让模型预测每个网格点：

```python
h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)
```

然后画出分类区域和原始样本：

```python
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
```

这里的 `Z` 表示网格上每个位置属于哪一类。`contourf` 用颜色填充分类区域，`scatter` 再叠加训练样本点。

## 10. 代码特点

这个 notebook 的主要特点：

- 代码短小，适合教学
- 所有神经网络计算都基于标量 `Value`
- 没有使用 PyTorch、TensorFlow 等深度学习框架
- 自动求导逻辑完全由 `micrograd/engine.py` 实现
- 神经网络封装在 `micrograd/nn.py`
- 使用 SVM 风格 hinge loss，而不是常见的 cross entropy
- 使用全量 batch 训练
- 只评估训练集准确率，没有划分测试集

## 11. 注意事项

- notebook 中导入了 `make_blobs`，但实际没有使用。
- `loss(batch_size=None)` 支持 mini-batch，但当前训练循环没有使用 mini-batch。
- `100% accuracy` 是训练集准确率，不代表泛化能力。
- 每次计算 loss 时都会重新把输入转换成 `Value`，这对教学清晰，但不是高性能实现。
- 最终 notebook 文件中保存了 matplotlib 图片输出，因此 `.ipynb` 文件体积会比较大。

## 12. 总结

`demo.ipynb` 展示了 MicroGrad 的完整使用链路：

```text
生成数据 -> 构造 MLP -> 前向传播 -> 计算 max-margin loss -> 反向传播 -> SGD 更新 -> 可视化边界
```

它的重点不是性能，而是清楚展示一个神经网络训练过程如何从最基础的标量自动求导机制搭建出来。

# 从 micrograd 到 PyTorch：反向传播与 MLP 神经网络

## 一、从 Value 到 PyTorch：同一套机制，工业级实现

notebook 里的 torch 代码和 Value 代码做的是**完全相同的事**，只是 PyTorch 更强大：

```python
# micrograd 版（已理解）
x1 = Value(2.0, label='x1')
w1 = Value(-3.0, label='w1')
n = x1*w1 + x2*w2 + b
o = n.tanh()
o.backward()

# PyTorch 版
x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)
o.backward()
```

### 核心映射关系

| micrograd (Value) | PyTorch (Tensor) | 说明 |
|---|---|---|
| `Value.data` | `tensor.data` | 存储数值 |
| `Value.grad` | `tensor.grad` | 存储梯度 |
| `Value._prev` | `tensor.grad_fn` | 记录计算图（谁生成了我） |
| `Value._backward` | `tensor.grad_fn` 内部 | 每个操作的反向传播函数 |
| `Value.backward()` | `tensor.backward()` | 触发整个反向传播 |
| 默认追踪梯度 | `requires_grad = True` | **关键区别**：PyTorch 默认不追踪，需显式开启 |

### PyTorch 的 `requires_grad` 是关键区别

```python
x1 = torch.Tensor([2.0]).double()  # 默认 requires_grad=False
x1.requires_grad = True             # 必须显式开启，告诉 PyTorch "我要对这个变量求梯度"
```

为什么？因为在深度学习中，我们通常只对**参数（权重和偏置）**求梯度，不对**输入数据**求梯度。micrograd 里所有 Value 都追踪梯度，是因为它小到不需要优化。

### PyTorch 的 `grad_fn` 就是 Value 的 `_prev + _backward`

```python
n = x1*w1 + x2*w2 + b
print(n.grad_fn)  # <AddBackward0 object> — 告诉你 n 是由加法产生的，反向传播时用加法的梯度规则
```

Value 里的 `_op` 和 `_prev` 合并在 PyTorch 里变成了 `grad_fn` 这个对象链。

**反向传播的过程完全一样**：拓扑排序 → 从输出开始 → 逐节点调用 `_backward`/`grad_fn`。

---

## 二、MLP 神经网络：把 Value 的能力组装成学习机器

MLP 代码的架构：

```
输入(3) → Layer1(4个神经元) → Layer2(4个神经元) → Layer3(1个神经元) → 输出(1)
```

### 2.1 单个 Neuron 的工作原理

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  # n个权重
        self.b = Value(random.uniform(-1,1))                         # 1个偏置

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)  # Σ(wi*xi) + b
        out = act.tanh()                                          # 激活函数
        return out
```

一个神经元就是前面手动构建的那个计算图：

```
x1 ──→ × ──→
w1 ──→ ×      ──→ + ──→ + ──→ tanh ──→ 输出
x2 ──→ ×      ──→    ↗
w2 ──→ ×          b ──┘
```

**关键理解**：每个 Neuron 本质上就是 `wx + b` 然后 `tanh`，和手动构建 `x1*w1 + x2*w2 + b → tanh` 是一模一样的。

### 2.2 Layer = 多个并行的 Neuron

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]  # nout个独立的神经元

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]  # 每个神经元独立处理同一个输入x
        return outs[0] if len(outs) == 1 else outs
```

每个神经元**独立地**看同一个输入，产生不同的输出（因为权重不同）。Layer(3, 4) 意味着 4 个神经元，每个有 3 个权重，输入 3 维，输出 4 维。

### 2.3 MLP = Layer 串联

```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts  # [3, 4, 4, 1]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        # Layer(3,4) → Layer(4,4) → Layer(4,1)
```

数据流：`x(3维) → Layer1 → 4维 → Layer2 → 4维 → Layer3 → 1维`

**这就是一个巨大的计算图**，所有中间变量都是 Value 对象，它们之间的连接关系构成了反向传播的路径。

---

## 三、训练循环：反向传播的真正用途

```python
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], ...]  # 4个输入样本
ys = [1.0, -1.0, -1.0, 1.0]                        # 4个目标输出
```

### 训练的完整循环

```
┌──────────────────────────────────────────────────┐
│  1. 前向传播 (Forward)                            │
│     对每个样本 x，计算 n(x) 得到预测值 ypred         │
│                                                    │
│  2. 计算损失 (Loss)                                │
│     loss = Σ(ypred - y_true)²  (均方误差)          │
│     loss 也是一个 Value，它连接了所有样本的计算图     │
│                                                    │
│  3. 反向传播 (Backward)                            │
│     loss.backward()                                │
│     → 梯度从 loss 沿计算图流回每个 w 和 b           │
│     → 每个参数都得到了自己的 grad                    │
│                                                    │
│  4. 参数更新 (Update)                              │
│     for p in n.parameters():                       │
│         p.data += -0.01 * p.grad    (梯度下降)     │
│     → 沿梯度反方向微调参数，让 loss 变小            │
│                                                    │
│  5. 重复 1-4                                       │
└──────────────────────────────────────────────────┘
```

### notebook 中的 Bug 解析

```python
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
# TypeError: unsupported operand type(s) for +: 'int' and 'Value'
```

原因：Python 内置 `sum()` 的初始值默认是 `0`（一个 int），对 `Value` 对象求和时，变成了 `0 + Value(...)`，但 `int.__add__` 不知道怎么加 Value。

**修复方法**：给 `sum()` 指定起始值为 `Value(0.0)`：

```python
loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)), Value(0.0))
```

这与 Neuron 的 `__call__` 里 `sum(..., self.b)` 的写法一致——用 Value 作为起始值避免 int + Value 的问题。

---

## 四、PyTorch 版本的等价 MLP

用 PyTorch 写同样的 MLP 训练，对比理解：

```python
import torch
import torch.nn as nn

# PyTorch 的 MLP（等价于 micrograd MLP）
model = nn.Sequential(
    nn.Linear(3, 4), nn.Tanh(),   # Layer1
    nn.Linear(4, 4), nn.Tanh(),   # Layer2
    nn.Linear(4, 1), nn.Tanh(),   # Layer3
)

# 训练循环
xs_t = torch.tensor(xs)  # 输入
ys_t = torch.tensor(ys).unsqueeze(1)  # 目标

for step in range(100):
    # 1. 前向传播
    ypred = model(xs_t)

    # 2. 计算损失
    loss = ((ypred - ys_t)**2).mean()

    # 3. 反向传播
    model.zero_grad()      # ← PyTorch 特有：必须手动清零梯度！
    loss.backward()        # 和 Value.backward() 完全一样的逻辑

    # 4. 参数更新
    for p in model.parameters():
        p.data += -0.01 * p.grad
```

### PyTorch 的一个关键区别：`zero_grad()`

micrograd 里每次 `backward()` 之前都会创建全新的 Value 对象（因为前向传播重新算了），所以 grad 天然是 0。但 PyTorch 里参数张量是**持久的**，`backward()` 会**累加**梯度（`+=`），所以每次迭代前必须手动清零。

---

## 五、核心直觉总结

| 概念 | micrograd | PyTorch |
|---|---|---|
| 计算图构建 | 每次 Value 运算自动建立 | 每次 Tensor 运算自动建立（需 requires_grad=True） |
| 反向传播 | `loss.backward()` → 拓扑排序 + 逐节点 `_backward()` | 完全相同的逻辑，`loss.backward()` → 拓扑排序 + `grad_fn` 链 |
| 梯度累加 | `self.grad +=` | `tensor.grad +=` (必须手动 zero_grad) |
| 参数更新 | 手动 `p.data += -lr * p.grad` | 手动或用 `optimizer.step()` |
| MLP | Neuron → Layer → MLP | `nn.Linear` → `nn.Sequential` |

**一句话总结**：PyTorch 就是一个能跑在 GPU 上的、支持多维张量的、工业级优化的 micrograd。MLP 就是一堆 Value 计算图串联起来，通过 loss.backward() 算出每个参数该怎么调，然后梯度下降。

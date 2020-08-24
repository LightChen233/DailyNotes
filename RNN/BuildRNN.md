# 一步步建立自己的RNN

RNN对自然语言处理和其他序列任务非常有效，因为它们具有“记忆”。它们可以一次读取一个输入(例如单词)，并通过从一个time-step传递到下一个time-step的隐藏层激活记住一些信息/上下文。这允许单向RNN从过去获取信息来处理以后的输入。双向RNN可以从过去和未来中获取上下文。

## 命名约定

- 上标 $[l]$ 表示第 $l$ 层 

- 上标 $(i)$ 表示第 $i$ 个样本的相关元素

- 上标 $\langle t \rangle$ 表示第 $t$ 个time-step. 
	-  $x^{(i)\langle t \rangle}$ 表示第$i$个样本的第 $t$ 个time-step 

- 下标 $i$ 表示第 $i$ 个向量的分量.

## 导包

```python
import numpy as np
from rnn_utils import *
```

## 基础RNN的正向传播

### 实现步骤

**注：**在这个例子中，$T_x = T_y$。

1. 实现一个time-step所需的RNN计算。
2. 在$T_x$ time-steps上实现一个循环，以便处理所有输入，一次一个。

### RNN单元

RNN可以看作是一个单元的重复。首先要实现一个time-step的计算。

![](E:/Gitee/Page/RNN/img/a1.jpg)

**指导**:

1. tanh激活计算隐藏状态:

	$a^{\langle t \rangle} = \tanh(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$.

2. 使用新的隐藏状态$a^{\langle t \rangle}$，计算预测$\hat{y}^{\langle t \rangle} = softmax(W_{ya} a^{\langle t \rangle} + b_y)$。我们提供了一个函数:` softmax `。

3. 存储 $(a^{\langle t \rangle}, a^{\langle t-1 \rangle}, x^{\langle t \rangle}, parameters)$ 到cache缓存

4. 返回 $a^{\langle t \rangle}$ , $y^{\langle t \rangle}$ 和 cache

![](E:/Gitee/Page/RNN/img/a2.jpg)

我们将对$m$的例子进行向量化。因此，$x^{\langle t \rangle}$将拥有维度$(n_x,m)$，而$a^{\langle t \rangle}$将拥有维度$(n_a,m)$。

```python
# GRADED FUNCTION: rnn_cell_forward

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Implements a single forward step of the RNN-cell as described in Figure (2)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """
    
    # 取参
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    ### START CODE HERE ### (≈2 lines)
    # 用上面给出的公式计算下一个激活状态
    a_next = np.tanh(np.matmul(Wax, xt) + np.matmul(Waa, a_prev) + ba)
    # 使用上面给出的公式计算当前单元格的输出
    yt_pred = softmax(np.matmul(Wya, a_next) + by) 
    ### END CODE HERE ###
    
    # 将反向传播所需的值存储在cache中
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache
```

#### 测试

```python
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", a_next.shape)
print("yt_pred[1] =", yt_pred[1])
print("yt_pred.shape = ", yt_pred.shape)
```

#### 结果

```
a_next[4] =  [ 0.59584544  0.18141802  0.61311866  0.99808218  0.85016201  0.99980978
 -0.18887155  0.99815551  0.6531151   0.82872037]
a_next.shape =  (5, 10)
yt_pred[1] = [0.9888161  0.01682021 0.21140899 0.36817467 0.98988387 0.88945212
 0.36920224 0.9966312  0.9982559  0.17746526]
yt_pred.shape =  (2, 10)
```

### RNN前传（forward pass）

您可以将RNN看作刚才构建的单元的重复。如果您的输入数据序列携带超过10个time-step，那么您将复制RNN单元格10次。每个单元格接受来自前一个单元格的**隐藏状态**($a^{\langle t-1 \rangle}$)和当前time-step的**输入数据**($x^{\langle t \rangle}$)作为输入。它输出此time-step的**隐藏状态**($a^{\langle t \rangle}$)和**预测**($y^{\langle t \rangle}$)。

![](E:/Gitee/Page/RNN/img/a3.jpg)

**指导**:

1. 创建一个零向量($a$），它将存储由RNN计算的所有隐藏状态。
2. 将“下一个”隐藏状态初始化为$a_0$（初始隐藏状态）。
3. 开始循环每个time-step，增量索引为$t$：
	- 通过运行`rnn_cell_forward`更新“下一个”隐藏状态和缓存
	- 将“next”隐藏状态存储在$a$（第$t$个位置）中
	- 将预测存储在y中
	- 将cache添加到缓存列表中
4. 返回 $a$, $y$ 和caches缓存列表

```python
# GRADED FUNCTION: rnn_forward

def rnn_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        ba --  Bias numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # 初始化包含所有缓存列表的“caches”
    caches = []
    
    # 获得shape
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    ### START CODE HERE ###
    
    # 零初始化a和y_pred (≈2 lines)
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    
    # 测试a_next (≈1 line)
    a_next = a0
    
    # 遍历所有time-steps
    for t in range(T_x):
        # 更新下一个隐藏状态，计算预测，获取cache (≈1 line)
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # 在a中保存新的“next”隐藏状态的值 (≈1 line)
        a[:,:,t] = a_next
        # 保存预测值y (≈1 line)
        y_pred[:,:,t] = yt_pred
        # 将"cache"添加到"caches" (≈1 line)
        caches.append(cache)
        
    ### END CODE HERE ###
    
    # 存储反向传播在cache中所需的值
    caches = (caches, x)
    
    return a, y_pred, caches
```

#### 测试

```python
np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
print("a[4][1] = ", a[4][1])
print("a.shape = ", a.shape)
print("y_pred[1][3] =", y_pred[1][3])
print("y_pred.shape = ", y_pred.shape)
print("caches[1][1][3] =", caches[1][1][3])
print("len(caches) = ", len(caches))
```

#### 结果

```
a[4][1] =  [-0.99999375  0.77911235 -0.99861469 -0.99833267]
a.shape =  (5, 10, 4)
y_pred[1][3] = [0.79560373 0.86224861 0.11118257 0.81515947]
y_pred.shape =  (2, 10, 4)
caches[1][1][3] = [-1.1425182  -0.34934272 -0.20889423  0.58662319]
len(caches) =  2
```

恭喜你!您已经从头开始成功地构建了递归神经网络的前向传播。这对于某些应用程序来说已经足够好了，但是它会遇到梯度消失的问题。

因此，当从每个输出$y^{\langle t \rangle}$都主要使用“本地”上下文来估计时(意味着来自输入$x^{\langle t' \rangle}$的信息，其中$t'$与$t$不是太远)时，它的工作效果最好。

在下一部分中，您将构建一个更复杂的LSTM模型，该模型可以更好地处理渐变消失的问题。LSTM将能够更好地记住一段信息，并将其保存为多个时间步骤。

## Long Short-Term Memory (LSTM)网络

与上面的RNN示例类似，您将首先为单个时间步骤实现LSTM单元。然后可以从for循环内部迭代地调用它，让它用$T_x$ time-step处理输入。

### 关于门

#### 遗忘门

为了便于演示，假设我们正在阅读一段文本中的单词，并且希望使用LSTM来跟踪语法结构，比如主语是单数还是复数。当主语由单复数变为单复数时，我们需要想办法摆脱我们先前储存的单复数状态的记忆值。在LSTM中，遗忘门让我们这样做:

$$
\Gamma_f^{\langle t \rangle} = \sigma(W_f[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f) 
$$

这里，$W_f$是控制遗忘门行为的权重。我们连接$[a^{\langle t-1 \rangle}， x^{\langle t \rangle}]$并乘以$W_f$。上面的方程得到一个向量$\Gamma_f^{\langle t \rangle}$，其值在0到1之间。这个遗忘门向量将在元素方面与前一个单元格状态相乘$c^{\langle t-1 \rangle}$。因此，如果$\Gamma_f^{\langle t \rangle}$的其中一个值为0(或接近0)，则意味着LSTM应该删除$c^{\langle t-1 \rangle}$对应组件中的那条信息(例如，单数形式)。如果其中一个值是1，那么它将保留该信息。

#### 更新门

一旦我们忘记了正在讨论的主语是单数的，我们需要找到一种方法来更新它，以反映新的主语现在是复数了。更新之门的公式如下:

$$
\Gamma_u^{\langle t \rangle} = \sigma(W_u[a^{\langle t-1 \rangle}, x^{\{t\}}] + b_u)
$$

与遗忘门类似，这里$\Gamma_u^{\langle t \rangle}$也是0到1之间的值向量。为了计算$c^{\langle t \rangle}$，它将在元素方面与$\tilde{c}^{\langle t \rangle}$相乘。

#### 更新单元

为了更新新的主题，我们需要创建一个新的数字向量，我们可以将其添加到以前的单元格状态中。我们使用的方程是:

$$
\tilde{c}^{\langle t \rangle} = \tanh(W_c[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c)
$$

最后，新单元状态为:

$$
c^{\langle t \rangle} = \Gamma_f^{\langle t \rangle}* c^{\langle t-1 \rangle} + \Gamma_u^{\langle t \rangle} *\tilde{c}^{\langle t \rangle}
$$


#### 输出门

为了决定我们将使用哪些输出，我们将使用以下两个公式: 

$$
\Gamma_o^{\langle t \rangle}=  \sigma(W_o[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o)
$$

$$
a^{\langle t \rangle} = \Gamma_o^{\langle t \rangle}* \tanh(c^{\langle t \rangle})
$$

### LSTM单元

```python
# GRADED FUNCTION: lstm_cell_forward

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of the LSTM-cell as described in Figure (4)

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the memory value
    """

    # 取参
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    
    # 获得shape的参数
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    ### START CODE HERE ###
    # 连接 a_prev 和 xt (≈3 lines)
    concat = np.zeros((n_a + n_x, m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt

    # 利用公式计算ft、it、cct、c_next、ot、a_next的值 (≈6 lines)
    ft = sigmoid(np.matmul(Wf, concat) + bf)
    it = sigmoid(np.matmul(Wi, concat) + bi)
    cct = np.tanh(np.matmul(Wc, concat) + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.matmul(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)
    
    # 计算 LSTM 单元的预测 (≈1 line)
    yt_pred = softmax(np.matmul(Wy, a_next) + by)
    ### END CODE HERE ###

    # 存储反向传播在cache中所需的值
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache
```

#### 测试

```python
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
print("a_next[4] = ", a_next[4])
print("a_next.shape = ", c_next.shape)
print("c_next[2] = ", c_next[2])
print("c_next.shape = ", c_next.shape)
print("yt[1] =", yt[1])
print("yt.shape = ", yt.shape)
print("cache[1][3] =", cache[1][3])
print("len(cache) = ", len(cache))
```

#### 结果

```
a_next[4] =  [-0.66408471  0.0036921   0.02088357  0.22834167 -0.85575339  0.00138482
  0.76566531  0.34631421 -0.00215674  0.43827275]
a_next.shape =  (5, 10)
c_next[2] =  [ 0.63267805  1.00570849  0.35504474  0.20690913 -1.64566718  0.11832942
  0.76449811 -0.0981561  -0.74348425 -0.26810932]
c_next.shape =  (5, 10)
yt[1] = [0.79913913 0.15986619 0.22412122 0.15606108 0.97057211 0.31146381
 0.00943007 0.12666353 0.39380172 0.07828381]
yt.shape =  (2, 10)
cache[1][3] = [-0.16263996  1.03729328  0.72938082 -0.54101719  0.02752074 -0.30821874
  0.07651101 -1.03752894  1.41219977 -0.37647422]
len(cache) =  10
```

### LSTM前传（forward pass）

现在已经实现了LSTM的一个步骤，现在可以使用for循环对其进行迭代，以处理$T_x$输入序列。

```python
# GRADED FUNCTION: lstm_forward

def lstm_forward(x, a0, parameters):
    """
    Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (3).

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """

    # 初始化“caches”，来跟踪所有缓存的列表
    caches = []
    
    ### START CODE HERE ###
    # 取参(≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wy'].shape
    
    # 零初始化 (≈3 lines)
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    # 初始化a_next和c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros(a_next.shape)
    
    # 遍历所有time-steps
    for t in range(T_x):
        # 更新下一个隐藏状态，下一个记忆状态，计算预测，获取缓存 (≈1 line)
        a_next, c_next, yt, cache = lstm_cell_forward(x[:,:,t], a_next, c_next, parameters)
        # 在a中保存新的“next”隐藏状态的值 (≈1 line)
        a[:,:,t] = a_next
        # 用y表示预测值 (≈1 line)
        y[:,:,t] = yt
        # 保存下一个单元格状态的值 (≈1 line)
        c[:,:,t]  = c_next
        # 添加cache到caches (≈1 line)
        caches.append(cache)
        
    ### END CODE HERE ###
    
    caches = (caches, x)

    return a, y, c, caches
```

#### 测试

```python
np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)
print("a[4][3][6] = ", a[4][3][6])
print("a.shape = ", a.shape)
print("y[1][4][3] =", y[1][4][3])
print("y.shape = ", y.shape)
print("caches[1][1[1]] =", caches[1][1][1])
print("c[1][2][1]", c[1][2][1])
print("len(caches) = ", len(caches))
```

#### 结果

```python
a[4][3][6] =  0.17211776753291672
a.shape =  (5, 10, 7)
y[1][4][3] = 0.9508734618501101
y.shape =  (2, 10, 7)
caches[1][1[1]] = [ 0.82797464  0.23009474  0.76201118 -0.22232814 -0.20075807  0.18656139
  0.41005165]
c[1][2][1] -0.8555449167181981
len(caches) =  2
```

恭喜你!现在您已经实现了基本RNN和LSTM的正向传播。当使用深度学习框架时，实现前传就足以构建出性能优异的系统。

## RNN中的反向传播 (选做)

在现代的深度学习框架中，你只需要实现前向传递，而框架负责后向传递，所以大多数深度学习工程师不需要费心处理后向传递的细节。然而，如果你是一个微积分专家，想要看到详细的RNN反馈，你可以进行这个笔记本的可选部分。



在前面的课程中，您实现了一个简单的(完全连接的)神经网络，您使用了反向传播来计算与更新参数的代价相关的导数。类似地，在递归神经网络中，你可以计算关于代价的导数，以更新参数。反馈很复杂我们在课堂上没有推导过。不过，我们将在下面简要介绍它们。

### 基本的RNN反馈

我们将从计算基本的rnn单元的向后传递开始。

#### 推导一阶反向函数

开始计算一个RNN单元的后向传播，如下图。这里需要注意导数计算的维度是否需变化。当然这里最好自己动手求下导。

![](E:/Gitee/Page/RNN/img/a4.jpg)

```python
def rnn_cell_backward(da_next, cache):
    """
    Implements the backward pass for the RNN-cell (single time-step).

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradients of input data, of shape (n_x, m)
                        da_prev -- Gradients of previous hidden state, of shape (n_a, m)
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dba -- Gradients of bias vector, of shape (n_a, 1)
    """
    
    (a_next, a_prev, xt, parameters) = cache
    
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    ### START CODE HERE ###
    # 计算tanh对a_next的梯度 (≈1 line)
    dtanh = (1 - a_next ** 2) * da_next

    # 计算关于Wax的梯度 (≈2 lines)
    dxt = np.matmul(Wax.T, dtanh)
    dWax = np.matmul(dtanh, xt.T)

    # 计算关于Waa的梯度 (≈2 lines)
    da_prev = np.matmul(Waa.T, dtanh)
    dWaa = np.matmul(dtanh, a_prev.T)

    # 计算关于b的梯度 (≈1 line)
    dba = np.sum(dtanh, axis=1, keepdims=True)

    ### END CODE HERE ###
    
    # 将梯度存储在python字典中
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    
    return gradients
```

##### 测试

```python
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)

da_next = np.random.randn(5,10)
gradients = rnn_cell_backward(da_next, cache)
print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)
```

##### 结果

```
gradients["dxt"][1][2] = -1.3872130506020928
gradients["dxt"].shape = (3, 10)
gradients["da_prev"][2][3] = -0.15239949377395473
gradients["da_prev"].shape = (5, 10)
gradients["dWax"][3][1] = 0.41077282493545836
gradients["dWax"].shape = (5, 3)
gradients["dWaa"][1][2] = 1.1503450668497135
gradients["dWaa"].shape = (5, 5)
gradients["dba"][4] = [0.20023491]
gradients["dba"].shape = (5, 1)
```

#### RNN反向传播

计算成本对在每个时间步 $a^{<t>}$ 的梯度，将帮助梯度传播到前一个RNN单元。这样我们就需要从最后的时间步开始迭代所有时间步，在每个时间步，增加$db_a,dW_{aa},dW_{ax}$和存储 $dx$。

这里实现RNN后向传播的思路是：先用零矩阵初始化 return 变量，然后每个时间步都调用RNN单元，循环所有时间步，最后相应地更新另外的变量。

```python
def rnn_backward(da, caches):
    """
    Implement the backward pass for a RNN over an entire sequence of input data.

    Arguments:
    da -- Upstream gradients of all hidden states, of shape (n_a, m, T_x)
    caches -- tuple containing information from the forward pass (rnn_forward)
    
    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient w.r.t. the input data, numpy-array of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t the initial hidden state, numpy-array of shape (n_a, m)
                        dWax -- Gradient w.r.t the input's weight matrix, numpy-array of shape (n_a, n_x)
                        dWaa -- Gradient w.r.t the hidden state's weight matrix, numpy-arrayof shape (n_a, n_a)
                        dba -- Gradient w.r.t the bias, of shape (n_a, 1)
    """
        
    ### START CODE HERE ###
    
    # 从caches的第一个缓存(t=1)中检索值 (≈2 lines)
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    
    # 获得shape的参数 (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # 零初始化 (≈6 lines)
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    
    # 选择合适的 da_prev
    da_prevt = da[:,:,-1] 
    
    print(da_prevt.shape)
    print(da.shape)
    
    # 循环所有的time-step
    for t in reversed(range(T_x)):
        # 计算时间步长t的梯度。明智地选择“da_next”和“caches”，以便在向后传播步骤中使用。(≈1 line)
        gradients = rnn_cell_backward(da_prevt, caches[t])
        # 获取梯度 (≈ 1 line)
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        # 通过增加time-step t中的导数获取全局派生变量 w.r.t (≈4 lines)
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        
    # 将da0设为a的梯度，a在所有time-step中都是反向传播的 (≈1 line) 
    da0 = da_prevt
    ### END CODE HERE ###

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    
    return gradients
```

##### 测试

```python
np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
a, y, caches = rnn_forward(x, a0, parameters)
da = np.random.randn(5, 10, 4)
gradients = rnn_backward(da, caches)

print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
print("gradients[\"dba\"][4] =", gradients["dba"][4])
print("gradients[\"dba\"].shape =", gradients["dba"].shape)
```

##### 结果

```
(5, 10)
(5, 10, 4)
gradients["dx"][1][2] = [ 0.04036334  0.01590669  0.00395097  0.01483317]
gradients["dx"].shape = (3, 10, 4)
gradients["da0"][2][3] = -0.000705301629139
gradients["da0"].shape = (5, 10)
gradients["dWax"][3][1] = 8.45242637129
gradients["dWax"].shape = (5, 3)
gradients["dWaa"][1][2] = 1.27076517994
gradients["dWaa"].shape = (5, 5)
gradients["dba"][4] = [-0.50815277]
gradients["dba"].shape = (5, 1)
```

### 单步反馈

LSTM的反馈比前馈稍微复杂一些。我们在下面为您提供了LSTM反馈的所有方程。(如果你喜欢微积分练习，可以尝试自己从零开始推导。)

### 门相关梯度

$$
d \Gamma_o^{\langle t \rangle} = da_{next}*\tanh(c_{next}) * \Gamma_o^{\langle t \rangle}*(1-\Gamma_o^{\langle t \rangle})
$$

$$
d\tilde c^{\langle t \rangle} = dc_{next}*\Gamma_u^{\langle t \rangle}+ \Gamma_o^{\langle t \rangle} (1-\tanh(c_{next})^2) * i_t * da_{next} * \tilde c^{\langle t \rangle} * (1-\tanh(\tilde c)^2) 
$$

$$
d\Gamma_u^{\langle t \rangle} = dc_{next}*\tilde c^{\langle t \rangle} + \Gamma_o^{\langle t \rangle} (1-\tanh(c_{next})^2) * \tilde c^{\langle t \rangle} * da_{next}*\Gamma_u^{\langle t \rangle}*(1-\Gamma_u^{\langle t \rangle})
$$

$$
d\Gamma_f^{\langle t \rangle} = dc_{next}*\tilde c_{prev} + \Gamma_o^{\langle t \rangle} (1-\tanh(c_{next})^2) * c_{prev} * da_{next}*\Gamma_f^{\langle t \rangle}*(1-\Gamma_f^{\langle t \rangle})
$$

### 参数相关梯度

$$
dW_f = d\Gamma_f^{\langle t \rangle} * \begin{pmatrix} a_{prev} \\ x_t\end{pmatrix}^T 
$$

$$
dW_u = d\Gamma_u^{\langle t \rangle} * \begin{pmatrix} a_{prev} \\ x_t\end{pmatrix}^T 
$$

$$
dW_c = d\tilde c^{\langle t \rangle} * \begin{pmatrix} a_{prev} \\ x_t\end{pmatrix}^T
$$

$$
dW_o = d\Gamma_o^{\langle t \rangle} * \begin{pmatrix} a_{prev} \\ x_t\end{pmatrix}^T
$$

要计算$db_f, db_u, db_c, db_o$ 你需要在水平轴上 (axis= 1) 分别对$d\Gamma_f^{\langle t \rangle}, d\Gamma_u^{\langle t \rangle}, d\tilde c^{\langle t \rangle}, d\Gamma_o^{\langle t \rangle}$ 求和

最后，您将计算对之前的隐藏状态、之前的内存状态和输入的导数。

$$
da_{prev} = W_f^T*d\Gamma_f^{\langle t \rangle} + W_u^T * d\Gamma_u^{\langle t \rangle}+ W_c^T * d\tilde c^{\langle t \rangle} + W_o^T * d\Gamma_o^{\langle t \rangle}
$$

$$
dc_{prev} = dc_{next}\Gamma_f^{\langle t \rangle} + \Gamma_o^{\langle t \rangle} * (1- \tanh(c_{next})^2)*\Gamma_f^{\langle t \rangle}*da_{next}
$$

$$
dx^{\langle t \rangle} = W_f^T*d\Gamma_f^{\langle t \rangle} + W_u^T * d\Gamma_u^{\langle t \rangle}+ W_c^T * d\tilde c_t + W_o^T * d\Gamma_o^{\langle t \rangle}
$$

```python
def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    ### START CODE HERE ###
    # 取得维度 (≈2 lines)
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    print(xt.shape)
    print(a_next.shape)
    
    # 计算门的相关导数 (≈4 lines)
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next * cct * (1 - np.tanh(cct) ** 2)
    dit = dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next * it * (1 - it)
    dft = dc_next * c_prev + ot * (1 - np.tanh(c_next) ** 2) * c_prev * da_next * ft * (1 - ft)
    
    # Code equations (7) to (10) (≈4 lines)
    #dit = None
    #dft = None
    #dot = None
    #dcct = None

    # 拼接 a_prev 和 xt (≈3 lines)
    concat = np.zeros((n_a + n_x, m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt
    
    print(concat.shape)
    print(dot.shape)
    
    # 计算参数相关的导数。 (≈8 lines)
    dWf = np.dot(dft, np.concatenate((a_prev, xt), axis=0).T)
    dWi = np.dot(dit, np.concatenate((a_prev, xt), axis=0).T)
    dWc = np.dot(dcct, np.concatenate((a_prev, xt), axis=0).T)
    dWo = np.dot(dot, np.concatenate((a_prev, xt), axis=0).T)
    dbf = np.sum(dft, keepdims=True, axis=1)
    dbi = np.sum(dit, keepdims=True, axis=1)
    dbc = np.sum(dcct, keepdims=True, axis=1)
    dbo = np.sum(dot, keepdims=True, axis=1)

    # 计算派生变量w.r.t以前的隐藏状态，以前的记忆状态和输入。 (≈3 lines)
    da_prev = np.dot(parameters['Wf'][:,:n_a].T, dft) + np.dot(parameters['Wi'][:,:n_a].T, dit) + np.dot(parameters['Wc'][:,:n_a].T, dcct) + np.dot(parameters['Wo'][:,:n_a].T, dot)
    dc_prev = dc_next*ft + ot*(1-np.square(np.tanh(c_next)))*ft*da_next
    dxt = np.dot(parameters['Wf'][:,n_a:].T,dft)+np.dot(parameters['Wi'][:,n_a:].T,dit)+np.dot(parameters['Wc'][:,n_a:].T,dcct)+np.dot(parameters['Wo'][:,n_a:].T,dot) 
    ### END CODE HERE ###
    
    
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients
```

#### 测试

```python
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
c_prev = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)
Wy = np.random.randn(2,5)
by = np.random.randn(2,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

da_next = np.random.randn(5,10)
dc_next = np.random.randn(5,10)
gradients = lstm_cell_backward(da_next, dc_next, cache)
#print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
#print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
#print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
#print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
#print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
#print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
#print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
#print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
#print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
#print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
#print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
#print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
#print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
#print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)
```

#### 结果

```
(3, 10)
(5, 10)
(8, 10)
(5, 10)
gradients["dbf"][4] = [-2.09422168]
gradients["dbf"].shape = (5, 1)
gradients["dbi"][4] = [-2.23460331]
gradients["dbi"].shape = (5, 1)
gradients["dbc"][4] = [ 2.99973436]
gradients["dbc"].shape = (5, 1)
gradients["dbo"][4] = [ 0.13893342]
gradients["dbo"].shape = (5, 1)
```

### LSTM RNN反馈

这部分非常类似于您在上面实现的`rnn_backward`函数。您将首先创建与返回变量具有相同维度的变量。然后，您将迭代所有的时间步骤，并在每次迭代中调用为LSTM实现的一步函数。然后，您将通过分别对它们求和来更新参数。最后返回一个带有新渐变的字典。

说明:实现`lstm_backward`函数。从$T_x$开始并向后创建一个for循环。对于每一步调用` lstm_cell_backward`并通过添加新的梯度来更新你的旧梯度。注意，` dxt `没有被更新，而是被存储。

```python
def lstm_backward(da, caches):
    
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    
    ### START CODE HERE ###
    # 从da和x1的shape获取参数 (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # 零初始化 (≈12 lines)
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a+n_x))
    dWi = np.zeros((n_a, n_a+n_x))
    dWc = np.zeros((n_a, n_a+n_x))
    dWo = np.zeros((n_a, n_a+n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    
    for t in reversed(range(T_x)):
        # 使用lstm_cell_backward计算所有的梯度
        gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
        # 存储或添加梯度到参数的上一步的梯度
        dx[:,:,t] = gradients['dxt']
        dWf = dWf + gradients['dWf']
        dWi = dWi + gradients['dWi']
        dWc = dWc + gradients['dWc']
        dWo = dWo + gradients['dWo']
        dbf = dbf + gradients['dbf']
        dbi = dbi + gradients['dbi']
        dbc = dbc + gradients['dbc']
        dbo = dbo + gradients['dbo']
    # 将第一个激活的梯度设置为反向传播的梯度da_prev。
    da0 = gradients['da_prev']
    
    ### END CODE HERE ###

    gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
    return gradients
```

#### 测试 

```python
np.random.seed(1)
x = np.random.randn(3,10,7)
a0 = np.random.randn(5,10)
Wf = np.random.randn(5, 5+3)
bf = np.random.randn(5,1)
Wi = np.random.randn(5, 5+3)
bi = np.random.randn(5,1)
Wo = np.random.randn(5, 5+3)
bo = np.random.randn(5,1)
Wc = np.random.randn(5, 5+3)
bc = np.random.randn(5,1)

parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

a, y, c, caches = lstm_forward(x, a0, parameters)

da = np.random.randn(5, 10, 4)
gradients = lstm_backward(da, caches)

print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
print("gradients[\"dx\"].shape =", gradients["dx"].shape)
print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
print("gradients[\"da0\"].shape =", gradients["da0"].shape)
print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)
```



#### 结果

```
(3, 10)
(5, 10)
(8, 10)
(5, 10)
(3, 10)
(5, 10)
(8, 10)
(5, 10)
(3, 10)
(5, 10)
(8, 10)
(5, 10)
(3, 10)
(5, 10)
(8, 10)
(5, 10)
gradients["dx"][1][2] = [ 0.19467822  0.1168994  -0.54230341 -0.36925417]
gradients["dx"].shape = (3, 10, 4)
gradients["da0"][2][3] = -0.009257763384732567
gradients["da0"].shape = (5, 10)
gradients["dWf"][3][1] = -0.06981985612744009
gradients["dWf"].shape = (5, 8)
gradients["dWi"][1][2] = 0.10237182024854771
gradients["dWi"].shape = (5, 8)
gradients["dWc"][3][1] = -0.62022441735199
gradients["dWc"].shape = (5, 8)
gradients["dWo"][1][2] = 0.04843891314443013
gradients["dWo"].shape = (5, 8)
gradients["dbf"][4] = [-0.0565788]
gradients["dbf"].shape = (5, 1)
gradients["dbi"][4] = [-0.15399065]
gradients["dbi"].shape = (5, 1)
gradients["dbc"][4] = [-0.05673381]
gradients["dbc"].shape = (5, 1)
gradients["dbo"][4] = [-0.29798344]
gradients["dbo"].shape = (5, 1)
```




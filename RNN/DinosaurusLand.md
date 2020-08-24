# 字符级语言模型-恐龙之地

## 问题描述

欢迎来到恐龙岛!6500万年前，恐龙就存在了，在这次作业中，恐龙又回来了。你负责一项特殊任务。领先的生物学研究人员正在创造新的恐龙品种，并将它们带到地球上，而你的工作就是给这些恐龙命名。如果恐龙不喜欢它的名字，它可能会选择beserk，所以明智地选择!

幸运的是，你已经学会了一些深入的学习，你将使用它来拯救这一天。您的助手已经收集了一张他们能找到的所有恐龙名字的清单，并将它们汇编到dinos.txt中。要创建新的恐龙名称，您将构建一个字符级语言模型来生成新的名称。您的算法将学习不同的名称模式，并随机生成新的名称。希望这个算法能让你和你的团队远离恐龙的愤怒!

## 学习目标

- 如何存储文本数据处理使用RNN

- 如何综合数据，通过抽样预测在每个time-step，并传递它到下一个RNN单元

- 如何构建一个字符级文本生成递归神经网络

- 为什么裁剪梯度是重要的

```python
import numpy as np
from utils import *
import random
```

## 数据集和预处理

### 读取数据

运行以下单元读取恐龙名称的数据集，创建一个唯一字符列表(例如a-z)，并计算数据集和词汇表的大小。

```python
data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
```

#### 结果

```
There are 19909 total characters and 27 unique characters in your data.
```

### 生成字典

这些字符是a-z(26个字符)加上“\n”(或换行字符)，该字符在本次任务中扮演的角色类似于我们在课堂上讨论过的` < EOS >`标记，只是在这里它表示恐龙名称的结束，而不是句子的结束。在下面的单元格中，我们创建了一个python字典(即，将每个字符映射到0-26之间的索引。我们还创建了第二个python字典，它将每个索引映射回对应的字符字符。这将帮助您计算出在softmax层的概率分布输出中，什么索引对应于什么字符。下面，` char_to_ix `和` ix_to_char `是python字典。

```python
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
print(ix_to_char)
```

#### 结果

```
{0: '\n', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
```

### 模型概述

您的模型将具有以下结构:



- 初始化参数

- 运行优化循环

- 正向传播计算损耗函数

- 反向传播计算关于损失函数的梯度

- 裁剪梯度，以避免爆炸梯度

- 使用梯度，更新你的参数梯度下降更新规则。

- 返回学习的参数

在每一个time-step，RNN尝试给定之前字符的条件下预测下一个字符是什么。数据集$X = (x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, ..., x^{\langle T_x \rangle})$是训练集中的一组字符，而是$Y = (y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, ..., y^{\langle T_x \rangle})$这样的，在每一个time-step $t$，我们有$y^{\langle t \rangle} = x^{\langle t+1 \rangle}$。

## 构建模块

在本部分中，您将构建整个模型的两个重要模块:

- 梯度裁剪:避免爆炸梯度

- 抽样:用于生成字符的技术

然后您将应用这两个函数来构建模型。

### 裁剪优化循环中的梯度

在本节中，您将实现将在优化循环中调用的`clip`函数。回想一下，您的整个循环结构通常由正向传播、成本计算、反向传递和参数更新组成。在更新参数之前，您将执行梯度裁剪，当需要时，以确保您的梯度不是“爆炸”，即采取过大的值。

在下面的练习中，您将实现一个函数`clip`，它接受一个梯度字典并在需要时返回一个梯度的剪切版本。剪辑渐变有不同的方法;我们将使用一个简单的元素剪切过程，其中梯度向量的每个元素都被剪切到某个范围[-N, N]之间。更一般地，您将提供一个`maxValue`(比如10)。在这个例子中，如果梯度向量的任何一个分量大于10，它将被设为10;如果梯度向量的任何一个分量小于-10，它就会被设为-10。如果它在-10和10之间，它就不受影响。

```python
### GRADED FUNCTION: clip

def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    
    ### START CODE HERE ###
    # 裁剪以减轻爆炸梯度 (≈2 lines)
    
    clipped_gradients = {}
    for g in ['dWax', 'dWaa', 'dWya', 'db', 'dby']:
        clipped_gradients[g] = np.clip(gradients[g], -maxValue, maxValue)
    ### END CODE HERE ###
    
    return clipped_gradients
```

#### 测试

```python
np.random.seed(3)
dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, 10)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
```

#### 结果

```
gradients["dWaa"][1][2] = 10.0
gradients["dWax"][3][1] = -10.0
gradients["dWya"][1][2] = 0.2971381536101662
gradients["db"][4] = [10.]
gradients["dby"][1] = [8.45833407]
```

### 采样

实现下面的“sample”函数来对字符进行采样。您需要执行4个步骤:

- **步骤1**:向网络传递第一个“假的”输入$x^{\langle 1 \rangle} = \vec{0}$(零向量)这是生成任何字符之前的默认输入。我们还设置$a^{\langle 0 \rangle} = \vec{0}$

- **步骤2**:运行一步正向传播，以获得$a^{\langle 1 \rangle}$和$\hat{y}^{\langle 1 \rangle}$。方程如下:

$$
a^{\langle t+1 \rangle} = \tanh(W_{ax}  x^{\langle t \rangle } + W_{aa} a^{\langle t \rangle } + b)
$$

$$
z^{\langle t + 1 \rangle } = W_{ya}  a^{\langle t + 1 \rangle } + b_y 
$$

$$
\hat{y}^{\langle t+1 \rangle } = softmax(z^{\langle t + 1 \rangle })
$$

**注意**，$\hat{y}^{\langle t+1 \rangle} $是一个(softmax)概率向量(它的分量在0和1之间并且和为1). $\hat{y}^{\langle t+1 \rangle}_i$表示由“i”索引的字符是下一个字符的概率。

- **步骤3**:执行抽样:根据$\hat{y}^{\langle t+1 \rangle}$指定的概率分布选择下一个字符的索引。这意味着，如果$\hat{y}^{\langle t+1 \rangle}_i = 0.16$，您将以16%的概率选择索引“i”。要实现它，可以使用[' np.random.choice '](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html)。

下面是一个如何使用`np.random.choice()`的例子:

```python
np.random.seed(0)
p = np.array([0.1, 0.0, 0.7, 0.2])
index = np.random.choice([0, 1, 2, 3], p = p.ravel())
```

这意味着你将根据分布选择`index`:
$P(index = 0) = 0.1, P(index = 1) = 0.0, P(index = 2) = 0.7, P(index = 3) = 0.2$。

- **步骤4**:在` sample() `中实现的最后一步是覆盖变量` x `，它当前存储$x^{\langle t \rangle}$，其值为$x^{\langle t + 1 \rangle}$。您将通过创建一个与您选择作为预测的字符对应的one-hot向量来表示$x^{\langle t + 1 \rangle}$。然后，在步骤1中向前传播$x^{\langle t + 1 \rangle}$，并一直重复此过程，直到得到一个“\n”字符，表示已经到达恐龙名称的末尾。

```python
# GRADED FUNCTION: sample

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    
    # 取参
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    ### START CODE HERE ###
    # 步骤 1: 零初始化 one-hot 向量 x 作为首字母 (初始化序列生成). (≈1 line)
    x = np.zeros((vocab_size, 1))
    # 步骤 1': 零初始化 a_prev (≈1 line)
    a_prev = np.zeros((n_a, 1))
    
    # 创建一个空的索引列表，这个列表将包含要生成的字符的索引列表 (≈1 line)
    indices = []
    
    # Idx是一个用来检测换行符的标志，我们把它初始化为-1
    idx = -1 
    
    # 循环time-step t。在每个time-step，从概率分布中抽取一个字符并添加它的索引到“indices”。
    #如果达到50个字符，我们将停止(在训练得很好的模型中是非常不可能的)这有助于调试，并防止进入无限循环。 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        
        # 步骤 2: 正向传播x
        a = np.tanh(np.matmul(Wax, x) + np.matmul(Waa, a_prev) + b)
        z = np.matmul(Wya, a) + by
        y = softmax(z)
        
        # 为了逐渐变化
        np.random.seed(counter+seed) 
        
        # 步骤 3: 从概率分布y中抽取词汇表中字符的索引
        idx = np.random.choice(range(vocab_size), p=y.ravel())

        # 添加 index 到 "indices"
        indices.append(idx)
        
        # 步骤 4: 将输入字符改写为与抽样索引对应的字符。
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        # 更新 "a_prev" 为 "a"
        a_prev = a
        
        # 为了逐渐变化
        seed += 1
        counter +=1
        
    ### END CODE HERE ###

    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices
```

#### 测试

```python
np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:", indices)
print("list of sampled characters:", [ix_to_char[i] for i in indices])
```

#### 结果

```
Sampling:
list of sampled indices: [12, 17, 24, 14, 13, 9, 10, 22, 24, 6, 13, 11, 12, 6, 21, 15, 21, 14, 3, 2, 1, 21, 18, 24, 7, 25, 6, 25, 18, 10, 16, 2, 3, 8, 15, 12, 11, 7, 1, 12, 10, 2, 7, 7, 11, 17, 24, 1, 13, 0, 0]
list of sampled characters: ['l', 'q', 'x', 'n', 'm', 'i', 'j', 'v', 'x', 'f', 'm', 'k', 'l', 'f', 'u', 'o', 'u', 'n', 'c', 'b', 'a', 'u', 'r', 'x', 'g', 'y', 'f', 'y', 'r', 'j', 'p', 'b', 'c', 'h', 'o', 'l', 'k', 'g', 'a', 'l', 'j', 'b', 'g', 'g', 'k', 'q', 'x', 'a', 'm', '\n', '\n']
```

## 构建语言模型

现在是时候构建用于文本生成的字符级语言模型了。


### 梯度下降+优化器

在本节中，您将实现一个执行随机梯度下降一步(使用裁剪梯度)的函数。你会一次一个地看训练例子，所以优化算法是随机梯度下降。作为提醒，以下是一个常见的RNN优化循环的步骤:

- 通过RNN向前传播，计算损失

- 反向传播，通过time来计算有关参数的损失的梯度

- 如果需要的话，裁剪梯度

- 使用梯度下降更新参数

```python
# GRADED FUNCTION: optimize

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    
    ### START CODE HERE ###
    
    # 通过 time 正向传播(≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    
    # 通过 time 反向传播 (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    # 梯度裁剪 (≈1 line)
    gradients = clip(gradients, 5)
    
    # 更新参数 (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    ### END CODE HERE ###
    
    return loss, gradients, a[len(X)-1]
```

#### 测试

```python
np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])
```

#### 结果

```
Loss = 126.50397572165382
gradients["dWaa"][1][2] = 0.1947093153471637
np.argmax(gradients["dWax"]) = 93
gradients["dWya"][1][2] = -0.007773876032002977
gradients["db"][4] = [-0.06809825]
gradients["dby"][1] = [0.01538192]
a_last[4] = [-1.]
```

### 训练模型

给定恐龙名称数据集，我们使用数据集的每一行(一个名称)作为一个训练示例。每100步的随机梯度下降，你会抽取10个随机选择的名字来观察算法的运行情况。记住要打乱数据集，以便随机梯度下降以**随机顺序**访问示例。

```python
# GRADED FUNCTION: model

def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text, size of the vocabulary
    
    Returns:
    parameters -- learned parameters
    """
    
    # 从 vocab_size 获取 n_x and n_y 
    n_x, n_y = vocab_size, vocab_size
    
    # 初始化参数
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # 初始化损失(这是必需的，因为我们想平滑损失，不用担心)
    loss = get_initial_loss(vocab_size, dino_names)
    
    # 构建所有恐龙名称的列表(训练示例)。
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    
    # 随机化所有恐龙的名字
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # 初始化LSTM的隐藏状态
    a_prev = np.zeros((n_a, 1))
    
    #优化循环
    for j in range(num_iterations):
        
        ### START CODE HERE ###
        
        # 使用上面的提示定义一个训练示例(X,Y) (≈ 2 lines)
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]] 
        Y = X[1:] + [char_to_ix["\n"]]
        
        # 执行一个优化步骤:正向传播 ->反向传播 ->梯度裁剪->更新参数
        # 选择学习率为 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)
        
        ### END CODE HERE ###
        
        # 使用一个延迟技巧来保持平滑的损失。这是为了加速训练。
        loss = smooth(loss, curr_loss)

        # 每2000次迭代，通过sample()生成n个字符，以检查模型是否正确学习
        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # 要打印的恐龙名字的数量
            seed = 0
            for name in range(dino_names):
                
                # 打印样本索引
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                
                seed += 1  # 为了达到同样的逐渐移动目的，将种子增加一。
      
            print('\n')
        
    return parameters
```

运行以下单元格，您应该观察模型在第一次迭代时输出随机外观的字符。经过几千次迭代之后，您的模型应该学会生成外观合理的名称。 

```python
parameters = model(data, ix_to_char, char_to_ix)
```

#### 结果

```
Iteration: 0, Loss: 23.087336

Nkzxwtdmfqoeyhsqwasjkjvu
Kneb
Kzxwtdmfqoeyhsqwasjkjvu
Neb
Zxwtdmfqoeyhsqwasjkjvu
Eb
Xwtdmfqoeyhsqwasjkjvu


Iteration: 2000, Loss: 27.884160

Liusskeomnolxeros
Hmdaairus
Hytroligoraurus
Lecalosapaus
Xusicikoraurus
Abalpsamantisaurus
Tpraneronxeros


Iteration: 4000, Loss: 25.901815

Mivrosaurus
Inee
Ivtroplisaurus
Mbaaisaurus
Wusichisaurus
Cabaselachus
Toraperlethosdarenitochusthiamamumamaon


Iteration: 6000, Loss: 24.608779

Onwusceomosaurus
Lieeaerosaurus
Lxussaurus
Oma
Xusteonosaurus
Eeahosaurus
Toreonosaurus


Iteration: 8000, Loss: 24.070350

Onxusichepriuon
Kilabersaurus
Lutrodon
Omaaerosaurus
Xutrcheps
Edaksoje
Trodiktonus


Iteration: 10000, Loss: 23.844446

Onyusaurus
Klecalosaurus
Lustodon
Ola
Xusodonia
Eeaeosaurus
Troceosaurus


Iteration: 12000, Loss: 23.291971

Onyxosaurus
Kica
Lustrepiosaurus
Olaagrraiansaurus
Yuspangosaurus
Eealosaurus
Trognesaurus


Iteration: 14000, Loss: 23.382339

Meutromodromurus
Inda
Iutroinatorsaurus
Maca
Yusteratoptititan
Ca
Troclosaurus


Iteration: 16000, Loss: 23.288447

Meuspsangosaurus
Ingaa
Iusosaurus
Macalosaurus
Yushanis
Daalosaurus
Trpandon


Iteration: 18000, Loss: 22.823526

Phytrolonhonyg
Mela
Mustrerasaurus
Peg
Ytronorosaurus
Ehalosaurus
Trolomeehus


Iteration: 20000, Loss: 23.041871

Nousmofonosaurus
Loma
Lytrognatiasaurus
Ngaa
Ytroenetiaudostarmilus
Eiafosaurus
Troenchulunosaurus


Iteration: 22000, Loss: 22.728849

Piutyrangosaurus
Midaa
Myroranisaurus
Pedadosaurus
Ytrodon
Eiadosaurus
Trodoniomusitocorces


Iteration: 24000, Loss: 22.683403

Meutromeisaurus
Indeceratlapsaurus
Jurosaurus
Ndaa
Yusicheropterus
Eiaeropectus
Trodonasaurus


Iteration: 26000, Loss: 22.554523

Phyusaurus
Liceceron
Lyusichenodylus
Pegahus
Yustenhtonthosaurus
Elagosaurus
Trodontonsaurus


Iteration: 28000, Loss: 22.484472

Onyutimaerihus
Koia
Lytusaurus
Ola
Ytroheltorus
Eiadosaurus
Trofiashates


Iteration: 30000, Loss: 22.774404

Phytys
Lica
Lysus
Pacalosaurus
Ytrochisaurus
Eiacosaurus
Trochesaurus


Iteration: 32000, Loss: 22.209473

Mawusaurus
Jica
Lustoia
Macaisaurus
Yusolenqtesaurus
Eeaeosaurus
Trnanatrax


Iteration: 34000, Loss: 22.396744

Mavptokekus
Ilabaisaurus
Itosaurus
Macaesaurus
Yrosaurus
Eiaeosaurus
Trodon


```

## 结论

你可以看到，在训练接近尾声时，你的算法已经开始生成比较可信的恐龙名称。一开始，它是随机产生的角色，但最后你会看到恐龙的名字有很酷的结尾。您可以继续运行该算法，并使用超参数来查看是否可以得到更好的结果。我们的实现产生了一些非常酷的名字，比如`maconucon`、`marloralus`和`macingsersaurus`。希望你的模型还了解到，恐龙的名字往往以 `saurus`, `don`, `aura`, `tor`等结尾。

如果您的模型生成了一些不酷的名称，不要完全怪模型——不是所有实际的恐龙名称听起来都很酷。(例如，`dromaeosauroides`是一个真实的恐龙名字，它也在训练集中。)但是这个模型应该给你一组候选者，你可以从中选择最酷的!

这个任务使用了一个相对较小的数据集，因此您可以在CPU上快速训练RNN。训练英语模型需要更大的数据集，通常需要更多的计算，并且可以在gpu上运行数小时。我们给恐龙起名字已经有一段时间了，到目前为止，我们最喜欢的名字是伟大的、不可战胜的、凶猛的:Mangosaurus!

## 像莎士比亚一样书写

本笔记本的其余部分是可选的，不评分，但我们希望你们能做，因为它很有趣，也能提供信息。

一个类似的(但更复杂的)任务是生成莎士比亚的诗歌。你可以使用莎士比亚的诗歌集，而不是从恐龙名字的数据集中学习。使用LSTM单元格，您可以了解文本中跨**多个字符的长期依赖关系**——例如。，如果一个字符出现在序列的某个位置，那么它可能会影响以后在序列中出现的不同字符。这些长期的依赖关系对于恐龙的名字来说不那么重要，因为它们的名字都很短。

我们使用Keras实现了一个莎士比亚诗歌生成器。运行以下单元格以加载所需的包和模型。这可能需要几分钟。

```python
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io
```

为了节省你的时间，我们已经训练了一个模型，为1000个时代的莎士比亚诗集称为[*“十四行诗”*](shakespeare.txt)。



让我们再训练这个模型一个时代。当它完成一个epoch的训练时(这也需要几分钟)，您可以运行` generate_output `，它将提示您输入(` < `40个字符)。这首诗将以你的句子开始，我们的RNN-Shakespeare将为你完成这首诗的其余部分!例如，尝试“Forsooth this make eth no sense”(不要输入引号)。取决于是否在末尾包含空格，结果也可能不同——尝试两种方法，并尝试其他输入。

```python
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
```

### 结果

```
Epoch 1/1
31412/31412 [==============================] - 60s 2ms/step - loss: 2.5428
```

### 运行

```python
# 运行此单元格以尝试使用不同的输入，而不必重新训练模型
generate_output()
```

### 结果

```
Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: Forsooth this maketh no sense


Here is your poem: 

Forsooth this maketh no sense.
hens co betheregor there beauny thou oft
paintame caom marsing worden wise dith ind,
whench to be the ssast though in moin aed,
again, then hid ploce time the these disthing to boust.
thee i then on thuth you thle anstiuns astien,
to grow'n when lose so inof and o'h, and now,
delerge onfouch, my it?ning the wam ma rove,
thine in leapl thous for great the dos, not wor a sacons,
hin womnt in my th
```

RNN-Shakespeare模型与您为恐龙命名所构建的模型非常相似。唯一的主要区别是:

- 用LSTM代替基本的RNN来捕获更长的依赖关系

- 该模型是一个较深的多层LSTM模型(2层)

- 使用Keras而不是python来简化代码

如果您想了解更多，您还可以在GitHub上查看Keras团队的文本生成实现:https://github.com/keras team/keras/blob/master/examples/lstm_text_generation.py。

## 参考

- This exercise took inspiration from Andrej Karpathy's implementation: https://gist.github.com/karpathy/d4dee566867f8291f086. To learn more about text generation, also check out Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- For the Shakespearian poem generator, our implementation was based on the implementation of an LSTM text generator by the Keras team: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py 
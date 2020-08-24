# 利用LSTM网络即兴表演爵士独奏

## 学习目标

- 应用一个LSTM的音乐生成。

- 生成自己的爵士音乐与深入学习。

## 导包

```python
from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
```

## 问题描述

你想为朋友的生日特别创作一首爵士乐。然而，你不知道任何乐器或音乐创作。幸运的是，您了解深度学习，可以使用LSTM网络解决这个问题。

您将训练一个网络，以产生具有代表性的风格的小说爵士乐独奏的一组已完成的工作。

### 数据集

你将在爵士音乐的语料库上训练你的算法。运行下面的单元格来听一段来自训练集的音频:

```python
IPython.display.Audio('./data/30s_seq.mp3')
```

我们已经小心地对音乐数据进行了预处理，以便按照音乐“值”来呈现它。你可以非正式地把每个“值”看作一个音符，它包括一个音调和一个持续时间。例如，如果你按下一个特定的钢琴键0.5秒，那么你就刚刚弹奏了一个音符。在音乐理论中，“值”实际上要比这复杂得多——具体地说，它还捕捉了同时演奏多个音符所需的信息。例如，在演奏一段音乐时，你可能会同时按下两个钢琴键(同时演奏多个音符会产生所谓的“和弦”)。但在这次作业中，我们不需要担心音乐理论的细节。为了完成这项任务，你所需要知道的就是我们将获得一个值的数据集，并将学习一个RNN模型来生成一系列的值。

#### 测试

我们的音乐生成系统将使用78种独特的值。运行以下代码来加载原始音乐数据并将其预处理为值。这可能需要几分钟。

```python
X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)
print(indices_values)
```

#### 结果

```
shape of X: (60, 30, 78)
number of training examples: 60
Tx (length of sequence): 30
total # of unique values: 78
Shape of Y: (30, 60, 78)
{0: 'A,0.250,<P1,d-5>', 1: 'X,0.250,<m6,M2>', 2: 'S,0.250,<d1,P-5>', 3: 'C,0.333,<M2,d-4>', 4: 'S,0.333,<d7,m3>', 5: 'A,0.333,<A4,d-2>', 6: 'A,0.333,<M2,d-4>', 7: 'S,0.250', 8: 'C,0.500,<m2,P-4>', 9: 'C,0.333,<P5,A1>', 10: 'S,0.250,<d6,m2>', 11: 'C,0.500', 12: 'C,0.250,<dd5,d1>', 13: 'C,0.333,<m2,P-4>', 14: 'C,0.250,<d1,P-5>', 15: 'C,0.250,<d4,M-2>', 16: 'S,0.667,<m3,m-3>', 17: 'C,0.250,<d6,m2>', 18: 'C,0.250,<M2,d-4>', 19: 'C,0.250,<P-4,d-8>', 20: 'S,0.250,<m2,P-4>', 21: 'A,0.250,<m-2,d-6>', 22: 'X,0.250,<d1,P-5>', 23: 'S,0.250,<d5,P1>', 24: 'C,0.250,<m3,m-3>', 25: 'C,0.667,<M2,d-4>', 26: 'S,0.333,<d1,P-5>', 27: 'C,0.250,<d5,P1>', 28: 'S,0.250,<dd5,d1>', 29: 'S,0.250,<m3,m-3>', 30: 'C,0.667,<d6,m2>', 31: 'S,0.250,<d4,M-2>', 32: 'S,0.250,<P4,m-2>', 33: 'S,0.250,<P5,A1>', 34: 'S,0.500,<d1,P-5>', 35: 'C,0.333,<d1,P-5>', 36: 'C,0.750,<m3,m-3>', 37: 'S,0.250,<M-2,m-6>', 38: 'X,0.250,<A4,d-2>', 39: 'C,0.333,<m3,m-3>', 40: 'S,0.500,<m7,M3>', 41: 'A,0.250,<M3,d-3>', 42: 'A,0.250,<P4,m-2>', 43: 'C,0.250,<A4,d-2>', 44: 'C,0.250,<M-2,m-6>', 45: 'A,0.250,<d5,P1>', 46: 'S,0.250,<A4,d-2>', 47: 'A,0.333,<P1,d-5>', 48: 'C,0.250,<d2,A-4>', 49: 'C,0.250,<m-2,d-6>', 50: 'S,0.750,<m3,m-3>', 51: 'C,0.250', 52: 'S,0.667,<d5,P1>', 53: 'A,0.250,<P-4,d-8>', 54: 'C,0.333,<A4,d-2>', 55: 'C,0.500,<P4,m-2>', 56: 'C,0.333,<m7,M3>', 57: 'X,0.250,<M-2,m-6>', 58: 'C,0.250,<P4,m-2>', 59: 'S,0.250,<d2,A-4>', 60: 'C,0.250,<P11,M7>', 61: 'C,0.250,<A-4,P-8>', 62: 'C,0.333,<P1,d-5>', 63: 'S,0.250,<P1,d-5>', 64: 'C,0.500,<m6,M2>', 65: 'X,0.250,<d2,A-4>', 66: 'A,0.250,<d4,M-2>', 67: 'S,0.750,<d5,P1>', 68: 'C,0.250,<P1,d-5>', 69: 'A,0.250,<M2,d-4>', 70: 'C,0.250,<m7,M3>', 71: 'S,0.333,<m2,P-4>', 72: 'S,0.333', 73: 'C,0.250,<d3,M-3>', 74: 'C,0.250,<P5,A1>', 75: 'A,0.250,<m2,P-4>', 76: 'C,0.250,<m2,P-4>', 77: 'C,0.250,<M3,d-3>'}
```

刚刚加载了以下内容:

- ` X `:这是一个(m， $T_x$， 78)维数组。我们有m个训练例子，每个都是$T_x =30$ 音乐值的片段。在每一个time-step，输入是78个不同的可能值中的一个，用一个one-hot向量表示。例如，X[i,t,:]是一个one-hot向量，表示第i个例子在t时刻的值。
- ` Y `:本质上和` X `是一样的，只是向左移动了一步(到过去)。与恐龙问题类似，我们感兴趣的是使用前面的值来预测下一个值的网络，因此我们的序列模型将尝试给定$x^{\langle 1\rangle}, \ldots, x^{\langle t \rangle}$条件下，预测 $y^{\langle t \rangle}$ 。但是，` Y `中的数据被重新排序为$(T_y, m, 78)$维，其中$T_y = T_x$。这种格式使得以后提供给LSTM更加方便。

- `n_values`:这个数据集中不同音乐值的数量。这应该是78。

- ` indices_values `: python字典从0-77映射到音乐值。

### 模型概述

我们将从一段更长的音乐中随机抽取30个值来训练模型。因此，我们不需要设置第一个输入$x^{\langle 1 \rangle} = \vec{0}$，这是我们之前为了表示恐龙名称的开始而做的，因为现在大多数音频片段都是从一段音乐中间的某个地方开始的。我们将每个片段设置为相同的长度$T_x = 30$，以使矢量化更容易。

## 构建模型

在这一部分中，你将建立并训练一个学习音乐模式的模型。为此，需要构建一个模型，该模型接受shape为$(m, T_x, 78)$的X和shape为$(T_y, m, 78)$的Y。我们将使用具有64维隐藏状态的LSTM。设`n_a = 64`

```
n_a = 64 
```

下面介绍如何创建具有多个输入和输出的Keras模型。

如果您正在构建一个RNN，即使在测试时整个输入序列$x^{\langle 1 \rangle}， x^{\langle 2 \rangle}， \ldots, x^{\langle T_x \rangle}$是预先给定的，例如，如果输入是单词，输出是标签，那么Keras有简单的内置函数来构建模型。

但是，对于序列生成，在测试时我们不能预先知道$x^{\langle t\rangle}$的所有值;相反，我们使用$x^{\langle t \rangle} = y^{\langle t-1 \rangle}$每次生成一个。因此，代码会有点复杂，您需要实现自己的for循环来遍历不同的time-step。

函数` djmodel() `将使用for循环调用LSTM层$T_x$ 次，重要的是所有$T_x$副本具有相同的权重。即，它不应该每次都重新初始化权重——$T_x$ 步应该共享权重。在Keras中实现具有共享权重的层的关键步骤是:

1. 定义层对象(使用全局变量)。

2. 在传播输入时调用这些对象。

我们已经定义了作为全局变量的层对象。请运行下一个单元格来创建它们。请检查Keras文档，以确保您理解了这些层是什么: [Reshape()](https://keras.io/layers/core/#reshape), [LSTM()](https://keras.io/layers/recurrent/#lstm), [Dense()](https://keras.io/layers/core/#dense).

```python
reshapor = Reshape((1, 78))                        # 使用在步骤 2.B 
LSTM_cell = LSTM(n_a, return_state = True)         # 使用在步骤 2.C
densor = Dense(n_values, activation='softmax')     # 使用在步骤 2.D
```

每个 `reshapor`, `LSTM_cell` 和 `densor` 现在都是层对象，您可以使用它们来实现`djmodel()`。为了传播一个Keras张量对象X通过其中一个层，使用 `layer_object(X)` (如果它需要多个输入，使用 `layer_object([X,Y])` )例如，`reshapor(X)`将通过上面定义的`Reshape((1,78))`层传播X。

实现 `djmodel()`。你将需要执行两个步骤:

1. 创建一个空列表“outputs”，以保存LSTM单元格在每个时间步骤中的输出。
2. 循环 $t \in 1, \ldots, T_x$:

从X中选择第t个time-step向量，该选择的shape应为(78,)。为此，使用以下代码在Keras中创建一个一般的[Lambda](https://keras.io/layers/core/# Lambda)层:

```    python
        x = Lambda(lambda x: X[:,t,:])(X)
```

查看Keras文档，了解它的作用。它创建了一个“临时”或“未命名”的函数(这就是Lambda函数)，它提取出合适的one-hot向量，并使这个函数成为一个Keras `Layer`对象，应用于` X `。

B.把X reshape为(1,78)。你可能会发现`reshapor()`层(定义在下面)很有帮助。

C.通过LSTM_cell的一个步骤运行X。记住，使用前一步的隐藏状态$a$和单元格状态$c$初始化LSTM_cell。使用以下格式:

```python
    a, _, c = LSTM_cell(input_x, initial_state=[前一个隐藏状态，前一个单元格状态])
```
D.通过一个dense+softmax层使用`densor`传播LSTM的输出激活值。

E.将预测值追加到“outputs”列表中


```python
# GRADED FUNCTION: djmodel

def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras model with the 
    """
    
    # 用shape定义模型的输入
    X = Input(shape=(Tx, n_values))
    
    # 定义s0，解码器LSTM的初始隐藏状态
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    ### START CODE HERE ### 
    # 步骤 1:创建空列表，以便在迭代时追加输出 (≈1 line)
    outputs = []
    
    # 步骤 2: 循环
    for t in range(Tx):
        
        # 步骤 2.A: 从X选择第 "t"个time-step向量 
        x = Lambda(lambda X: X[:,t,:])(X)
        # 步骤 2.B: 使用 reshapor 来 reshape x 为 (1, n_values) (≈1 line)
        x = reshapor(x)
        # 步骤 2.C: 执行LSTM_cell的一步
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # 步骤 2.D: 对LSTM_Cell的隐藏状态的输出应用densor全连接
        out = densor(a)
        # 步骤 2.E: 添加 output 到 "outputs"
        outputs.append(out)
        
    # 步骤 3: 创建模型实例
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return model
```

运行以下单元格来定义您的模型。我们将使用` Tx=30`、` n_a=64 `(LSTM激活的维度)和` n_values=78 `。该单元格可能需要几秒钟才能运行完毕。

```python
model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
```

现在需要编译要训练的模型。我们将使用Adam和交叉熵损失。

```python
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
```

最后，让我们将LSTM的初始状态的` a0 `和` c0 `初始化为0。

```python
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
```

现在让我们fit模型!在这样做之前，我们将把`Y `转换为列表，因为成本函数期望` Y `以这种格式提供(每个time-step一个列表项)。因此`list(Y) `是一个包含30个条目的列表，其中每个条目的shape都是(60,78)。让我们为100个epoch而训练。这将需要几分钟。

```python
# model.fit([X, a0, c0], list(Y), epochs=100)
```

你会看到模型损失在下降。现在您已经训练了一个模型，让我们进入最后一节来实现一个推理算法，并生成一些音乐!

## 生成音乐

你现在有一个训练过的模型，它已经学会了爵士独奏者的模式。现在让我们使用这个模型来合成新的音乐。

### 预测 & 采样

在采样的每一步，您将从LSTM的前一个状态中获取激活` a `和单元状态` c `作为输入，向前传播一步，并获得一个新的输出激活和单元状态。然后可以使用新的激活` a `来生成输出，如前面一样使用` densor `。

在模型开始时，我们将初始化`x0 `以及LSTM激活和单元格值` a0 `和`c0 `为零。

实现下面的函数来对音乐值序列进行采样。下面是你需要在for循环中实现的一些关键步骤，生成$T_y$输出字符:

步骤2. A:使用` LSTM_Cell `，它输入上一步的` c `和` A `来生成当前一步的` c `和`A`。

步骤2. B:使用` densor `(全连接)计算` a `上的softmax，以获得当前步骤的输出。

步骤2. C:保存刚刚生成的output，将其添加到' outputs '中。

步骤2. D:采样x为“out”的one-hot版本(预测)，以便您可以将其传递到下一个LSTM的步骤。我们已经提供了这行代码，它使用了一个[Lambda](https://keras.io/layers/core/# Lambda)函数。

```python
x = Lambda(one_hot)(out) 
```

小技术要点:这一行代码实际上使用argmax在每一步选择一个最可能的值，而不是根据输出的概率随机采样一个值。

```python
# GRADED FUNCTION: music_inference_model

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # 按照shape 定义的模型的输入
    x0 = Input(shape=(1, n_values))
    
    # 定义s0，解码器LSTM的初始隐藏状态
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # 步骤 1:创建一个空的“outputs”列表，以便稍后存储预测值 (≈1 line)
    outputs = []
    
    # 步骤 2:循环Ty并在每一步生成一个值
    for t in range(Ty):
        
        # 步骤 2.A: 运行LSTM_cell中的一步 (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # 步骤 2.B: 对LSTM_Cell的隐藏状态的输出应用densor全连接 (≈1 line)
        out = densor(a)

        # 步骤 2.C: 添加 output 到 "outputs"。 out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # 步骤 2.D: 根据“out”选择下一个值，并设置“x”为所选值的one-hot表示，该值将在下一步作为输入传递给LSTM_cell。
        x = Lambda(one_hot)(out)
        
    # 步骤 3: 使用正确的"inputs"和"outputs"创建模型实例 (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return inference_model
```

运行下面的单元格以定义推理模型。这个模型被硬编码为生成50个值。

```
inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)
```

最后，这将创建用于初始化`x `和LSTM状态变量`a `和`c`的零值向量。

```
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
```

#### 预测采样

```python
# GRADED FUNCTION: predict_and_sample

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    ### START CODE HERE ###
    # 步骤 1: 使用推理模型来预测给定x_initializer、a_initializer和c_initializer的输出序列。
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # 步骤 2: 将“pred”转换为具有最大概率的索引的np.array()
    indices = np.argmax(np.array(pred), axis=-1)
    # 步骤 3: 将indices转换为one-hot向量，结果的shape应为(1，)
    results = to_categorical(indices, num_classes=x_initializer.shape[-1])

    ### END CODE HERE ###
    
    return results, indices
```

##### 测试

```python
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))

```

##### 结果

```
np.argmax(results[12]) = 66
np.argmax(results[17]) = 9
list(indices[12:18]) = [array([66], dtype=int64), array([1], dtype=int64), array([1], dtype=int64), array([1], dtype=int64), array([2], dtype=int64), array([9], dtype=int64)]
```

#### 生成音乐

最后，您可以准备生成音乐了。RNN生成一系列值。下面的代码通过首先调用` predict_and_sample()`函数来生成音乐。然后这些值被后处理为音乐和弦(这意味着可以同时播放多个值或音符)。

大多数计算音乐算法都使用一些后期处理，因为如果不进行这样的后期处理，就很难生成听起来不错的音乐。后期处理做的事情包括清理生成的音频，确保相同的声音不会重复太多次，两个连续的音符的音高不会相差太远，等等。有人可能会说，很多这些后处理步骤都是黑客;而且，很多音乐生成的文献也专注于手工制作的后期处理器，而且很多输出质量取决于后期处理的质量，而不仅仅是RNN的质量。但是这种后处理确实有很大的不同，所以让我们在实现中也使用它。

让我们来点音乐吧!

##### 测试

运行以下单元格生成音乐并将其录制到您的` out_stream `中。这可能需要几分钟。

```python
out_stream = generate_music(inference_model)
```

##### 结果

```
Predicting new values for different set of chords.
Generated 51 sounds using the predicted values for the set of chords ("1") and after pruning
Generated 51 sounds using the predicted values for the set of chords ("2") and after pruning
Generated 51 sounds using the predicted values for the set of chords ("3") and after pruning
Generated 51 sounds using the predicted values for the set of chords ("4") and after pruning
Generated 51 sounds using the predicted values for the set of chords ("5") and after pruning
Your generated music is saved in output/my_music.midi
```

### 生成

```python
IPython.display.Audio('./data/30s_trained_model.mp3')
```



## 参考

The ideas presented in this notebook came primarily from three computational music papers cited below. The implementation here also took significant inspiration and used many components from Ji-Sung Kim's github repository.

- Ji-Sung Kim, 2016, [deepjazz](https://github.com/jisungk/deepjazz)
- Jon Gillick, Kevin Tang and Robert Keller, 2009. [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf)
- Robert Keller and David Morrison, 2007, [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07 Proceedings/SMC07 Paper 55.pdf)
- François Pachet, 1999, [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf)

We're also grateful to François Germain for valuable feedback.


# 词向量运算

欢迎来到你本周的第一个作业!

因为训练单词嵌入在计算上非常昂贵，所以大多数ML从业者将加载一组预先培训过的嵌入。

## 学习目标

- 加载预先训练好的单词向量，并使用余弦相似度**度量相似度**
- 使用单词嵌入来解决**单词类比**问题，如男人对于女人就像国王对于**__**。
- 修改嵌入单词以减少他们的性别偏见

## 导包

```python
# encoding=utf8

import numpy as np
from w2v_utils import *
Using TensorFlow backend.
```

## 加载数据

接下来，加载词向量。我们将使用50维的GloVe向量来表示单词。运行以下单元格以加载`word_to_vec_map`.

```python
words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
```

上边这段话加载了:

- `words`:词汇表中的一组单词。

- `word_to_vec_map`:字典映射单词到他们的GloVe向量表示。

你已经看到，one-hot向量不能很好地区分哪些词是相似的。GloVe向量提供了关于单个单词的意义的更有用的信息。现在让我们看看如何使用手套向量来确定两个单词有多相似。

## 余弦相似度

为了衡量两个单词的相似程度，我们需要一种方法来衡量两个单词的嵌入向量之间的相似程度。给定两个向量$u$和$v$，余弦相似度定义如下:

$$
\text{CosineSimilarity(u, v)} = \frac {u . v} {||u||_2 ||v||_2} = cos(\theta)
$$

在 $u.v$ 是两个向量的点积(或内积)，$||u||_2$是向量$u$的范数(或长度)并且$\theta$是 $u$ 和$v$之间的夹角。

这种相似性取决于$u$和$v$之间的角度。如果$u$和$v$非常相似，它们的余弦相似度将接近于1;如果它们不相似，余弦相似度会取较小的值。

**注意**:  $u$ 的范数的定义为： $ ||u||_2 = \sqrt{\sum_{i=1}^{n} u_i^2}$

```python
# GRADED FUNCTION: cosine_similarity

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0
    
    ### START CODE HERE ###
    # 计算u和v的点积 (≈1 line)
    dot = np.dot(u, v)
    # 计算u的 L2 范数 (≈1 line)
    norm_u = np.linalg.norm(u)
    # 计算v的 L2 范数 (≈1 line)
    norm_v = np.linalg.norm(v)
    
    # 计算公式定义的余弦相似度 (≈1 line)
    cosine_similarity = dot / (norm_u * norm_v)
    ### END CODE HERE ###
    
    return cosine_similarity
```

### 测试

```python
father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))
```

### 结果

```
cosine_similarity(father, mother) =  0.890903844289
cosine_similarity(ball, crocodile) =  0.274392462614
cosine_similarity(france - paris, rome - italy) =  -0.675147930817
```

在得到正确的预期输出后，请随意修改输入并测量其他单词对之间的余弦相似度!摆弄其他输入的余弦相似度会让你更好地了解单词向量的行为。

## 词类比任务

在词语类比任务中，我们完成句子 <font color='brown'>"*a* is to *b* as *c* is to **____**"</font>. 一个例子是"*man* is to *woman* as *king* is to *queen*"。详细地，我们尝试找到一个单词*d*，使关联的单词向量$e_a, e_b, e_c, e_d$以以下方式关联:$e_b - e_a \approx e_d - e_c$。我们将使用余弦相似度度量$e_b - e_a$和$e_d - e_c$之间的相似度。

```python
# GRADED FUNCTION: complete_analogy

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    
    # 将单词转换为小写
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    ### START CODE HERE ###
    # 获得词嵌入 v_a, v_b 和 v_c (≈1-3 lines)
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    ### END CODE HERE ###
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # 初始化max_cosine_sim为一个很大的负数
    best_word = None                   # 用None初始化best_word，用于帮助跟踪要输出的单词

    # 循环整个词向量集
    for w in words:        
        # 为了避免best_word成为输入词之一，略过它。
        if w in [word_a, word_b, word_c] :
            continue
        
        ### START CODE HERE ###
        # 计算向量(e_b - e_a)与向量((w的向量表示)- e_c之间的余弦相似度  (≈1 line)
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)
        
        # 如果cosine_sim大于目前为止看到的max_cosine_sim，
            # 那么:将新的max_cosine_sim设置为当前的cosine_sim，将best_word设置为当前的word (≈3 lines)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        ### END CODE HERE ###
        
    return best_word
```

运行下面的单元测试代码，这可能需要1-2分钟。

```python
triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
```

结果

```
italy -> italian :: spain -> spanish
india -> delhi :: japan -> tokyo
man -> woman :: boy -> girl
small -> smaller :: large -> larger
```

一旦你得到正确的预期输出，请随意修改上面的输入单元来测试你自己的类比。试着找到一些其他有用的类比对，但也要找到一些算法没有给出正确答案的地方:例如，您可以尝试把small->smaller as big->?。

 ## 恭喜你!

你已经完成了这项任务。以下是你应该记住的要点:

- 余弦相似度比较对词向量之间的相似度的一个好方法。(尽管L2距离也适用。)

- 对于NLP应用程序，使用一组从互联网上获取的预先训练的词向量通常是一个很好的开始。

## 去偏词向量 (选做)

在下面的练习中，您将检查可以反映在一个单词嵌入中的性别偏见，并探索减少这种偏见的算法。除了学习去偏的主题，这个练习也将帮助你磨练关于单词向量正在做什么的直觉。这一节涉及到一点线性代数，尽管即使你不是线性代数方面的专家，你也可以完成它，我们鼓励你尝试一下。

### GloVe词嵌入与性别相关

首先让我们看看GloVe词嵌入是如何与性别相关的。首先计算一个向量$g = e_{woman}-e_{man}$，其中$e_{woman}$表示对应于单词*woman*的词向量，$e_{man}$对应于对应于单词*man*的单词向量。得到的向量$g$大致编码了“性别”的概念。(如果计算$g_1 = e_{mother}-e_{father}$， $g_2 = e_{girl}-e_{boy}$等并对它们求平均，可能会得到更准确的表示。但是现在只要使用$e_{woman}-e_{man}$就可以得到足够好的结果。)

#### 测试

```python
g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g)
```

#### 结果

```
[-0.087144    0.2182     -0.40986    -0.03922    -0.1032      0.94165
 -0.06042     0.32988     0.46144    -0.35962     0.31102    -0.86824
  0.96006     0.01073     0.24337     0.08193    -1.02722    -0.21122
  0.695044   -0.00222     0.29106     0.5053     -0.099454    0.40445
  0.30181     0.1355     -0.0606     -0.07131    -0.19245    -0.06115
 -0.3204      0.07165    -0.13337    -0.25068714 -0.14293    -0.224957
 -0.149       0.048882    0.12191    -0.27362    -0.165476   -0.20426
  0.54376    -0.271425   -0.10245    -0.32108     0.2516     -0.33455
 -0.04371     0.01258   ]
```

### 理解余弦相似度值

#### 测试

现在，你将考虑不同单词的余弦相似度与*g*。考虑一个正的相似度值与一个负的余弦相似度是什么意思。

```python
print ('List of names and their similarities with constructed vector:')

# girls and boys name
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
```

#### 结果

```
List of names and their similarities with constructed vector:
john -0.23163356146
marie 0.315597935396
sophie 0.318687898594
ronaldo -0.312447968503
priya 0.17632041839
rahul -0.169154710392
danielle 0.243932992163
reza -0.079304296722
katy 0.283106865957
yasmin 0.233138577679
```

正如你所看到的，女性名字与我们构建的向量*g*趋向于具有**正余弦相似度**，而男性名字趋向于具有**负余弦相似度**。这并不意外，结果似乎也可以接受。

#### 其他情况

##### 测试

但是让我们尝试一些其他的单词。

```python
print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
```

##### 结果

```
Other words and their similarities:
lipstick 0.276919162564
guns -0.18884855679
science -0.0608290654093
arts 0.00818931238588
literature 0.0647250443346
warrior -0.209201646411
doctor 0.118952894109
tree -0.0708939917548
receptionist 0.330779417506
technology -0.131937324476
fashion 0.0356389462577
teacher 0.179209234318
engineer -0.0803928049452
pilot 0.00107644989919
computer -0.103303588739
singer 0.185005181365
```



你注意到什么令人惊讶的事情了吗?令人惊讶的是，这些结果反映了某些不健康的性别刻板印象。例如，“computer”更接近“man”，而“literature”更接近“woman”。哎哟!

下面我们将看到如何减少这些向量的偏差，使用基于[Boliukbasi et al., 2016](https://arxiv.org/abs/1607.06520)的算法。需要注意的是，像"actor"/"actress"或"grandmother"/"grandfather"这样的词组合应该保持性别特异性，而像"receptionist"或"technology"这样的词应该中性化，即不与性别相关。去偏时，你必须**区别对待**这两种类型的单词。

### 消除对非性别特定词汇的偏见

如果您使用的是50维的字嵌入，那么50维的空间可以分为两部分:

偏置方向$g$和其余的49维，我们将其称为$g_{\perp}$。在线性代数中，我们说49维的$g_{\perp}$与$g$垂直(或“正交”)，这意味着它与$g$是90度。中性化步骤是取一个向量，例如，并将$e_{receptionist}$ *g*方向的分量归零，然后返回给我们$e_{receptionist}^{debiased}$。

使用 `neutralize()` 来消除 "receptionist" 或 "scientist"的偏差。给定输入嵌入 $e$，你可以使用如下公式计算 $e^{debiased}$: 

$$
e^{bias\_component} = \frac{e \cdot g}{||g||_2^2} * g
$$
$$
e^{debiased} = e - e^{bias\_component}
$$

如果您是线性代数方面的专家，您可能会认为$e^{bias\_component}$是$e$在$g$方向上的投影。如果你不是线性代数方面的专家，不用担心这个。

```python
def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.
    
    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.
    
    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    
    ### START CODE HERE ###
    # 选择“word”的词向量表示形式。使用word_to_vec_map。 (≈ 1 line)
    e = word_to_vec_map[word]
    
    # 用上述公式计算e_biascomponent。 (≈ 1 line)
    e_biascomponent = np.dot(e, g) / (np.linalg.norm(g) ** 2) * g
 
    # 通过减去e_biascomponent来中和偏见
    # e_debias应该等于它的正交投影。 (≈ 1 line)
    e_debiased = e - e_biascomponent
    ### END CODE HERE ###
    
    return e_debiased
```

#### 测试

```python
e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))
```

#### 结果

```
cosine similarity between receptionist and g, before neutralizing:  0.330779417506
cosine similarity between receptionist and g, after neutralizing:  -3.26732746085e-17
```

### 性别词汇的均衡算法

接下来，让我们看看如何去偏也适用于单词对，如"actress" 和 "actor"。均等化适用于你希望仅通过性别属性来区分的一对单词。举个具体的例子，假设"actress"比"actor"更接近"babysit"。通过对"babysit"进行中性化，我们可以减少与保姆相关的性别刻板印象。但这仍然不能保证"actress" 和 "actor"与"babysit"的距离是相等的。均衡算法解决了这个问题。

均衡化背后的关键思想是确保特定的一对单词与49维$g_\perp$之间是等距的。均衡化步骤还确保两个均衡化步骤现在到$e_{receptionist}^{debiased}$与任何其他已被中和的工作之间的距离相同。


用线性代数的推导来做这个有点复杂。 (See Bolukbasi et al., 2016 for details.) 但关键的方程式是:

$$
\mu = \frac{e_{w1} + e_{w2}}{2}
$$

$$
\mu_{B} = \frac {\mu \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
$$

$$
\mu_{\perp} = \mu - \mu_{B}
$$

$$
e_{w1B} = \frac {e_{w1} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
$$
$$
e_{w2B} = \frac {e_{w2} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
$$


$$
e_{w1B}^{corrected} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{e_{\text{w1B}} - \mu_B} {|(e_{w1} - \mu_{\perp}) - \mu_B)|} 
$$


$$
e_{w2B}^{corrected} = \sqrt{ |{1 - ||\mu_{\perp} ||^2_2} |} * \frac{e_{\text{w2B}} - \mu_B} {|(e_{w2} - \mu_{\perp}) - \mu_B)|} 
$$

$$
e_1 = e_{w1B}^{corrected} + \mu_{\perp}
$$
$$
e_2 = e_{w2B}^{corrected} + \mu_{\perp}
$$

#### 测试

```
x = np.arange(-10,10)
x
np.abs(x)
```

#### 结果

```
array([10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  1,  2,  3,  4,  5,  6,
        7,  8,  9])
```

#### 性别均衡

```python
def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.
    
    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors
    
    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    
    ### START CODE HERE ###
    # 步骤 1: 选择“word”的词向量表示形式。使用word_to_vec_map。 (≈ 2 lines)
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    # 步骤 2: 计算 e_w1 和 e_w2 的均值(≈ 1 line)
    mu = (e_w1 + e_w2) / 2

    # 步骤 3: 计算mu在偏置轴和正交轴上的投影 (≈ 2 lines)
    mu_B = np.dot(mu, bias_axis) / (np.linalg.norm(bias_axis) ** 2) * bias_axis
    mu_orth = mu - mu_B

    # 步骤 4: 利用公式计算e_w1B、e_w2B (≈2 lines)
    e_w1B = np.dot(e_w1, bias_axis) / (np.linalg.norm(bias_axis) ** 2) * bias_axis
    e_w2B = np.dot(e_w2, bias_axis) / (np.linalg.norm(bias_axis) ** 2) * bias_axis
        
    # 步骤 5: 利用上述公式调整e_w1B和e_w2B的偏置部分(≈2 lines)
    corrected_e_w1B = np.sqrt(np.abs(1 - np.linalg.norm(mu_orth) ** 2)) * (e_w1B - mu_B) / np.abs((e_w1 - mu_orth) - mu_B)
    corrected_e_w2B = np.sqrt(np.abs(1 - np.linalg.norm(mu_orth) ** 2)) * (e_w2B - mu_B) / np.abs((e_w2 - mu_orth) - mu_B)
 
    # 步骤 6: 通过使e1和e2与它们校正后的投影之和相等来消除偏置 (≈2 lines)
    e1 = corrected_e_w1B + mu_orth 
    e2 = corrected_e_w2B + mu_orth 
                                                                
    ### END CODE HERE ###
    
    return e1, e2
```

##### 测试

```python
print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
cosine similarities before equalizing:
cosine_similarity(word_to_vec_map["man"], gender) =  -0.117110957653
cosine_similarity(word_to_vec_map["woman"], gender) =  0.356666188463
```

##### 结果

```
cosine similarities after equalizing:
cosine_similarity(e1, gender) =  -0.716572752584
cosine_similarity(e2, gender) =  0.739659647493
```

请随意使用上面单元格中的输入单词，将均衡应用到其他成对的单词上。

这些去偏算法对减少偏置有很大的帮助，但并不完美，不能消除所有的偏置痕迹。例如，这个实现的一个缺点是，偏差方向$g$只使用了一对单词_woman_和_man_来定义。如前所述，如果$g$是由计算$g_1 = e_{woman} - e_{man}$;$g_2 = e_{mother} - e_{father}$;$g_3 = e_{girl} - e_{boy}$;共同定义的。那么，对它们进行平均，你会得到一个在50维单词嵌入空间中对“性别”维度的更好估计。您也可以随意使用这些变体。

## 参考

- The debiasing algorithm is from Bolukbasi et al., 2016, [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)
- The GloVe word embeddings were due to Jeffrey Pennington, Richard Socher, and Christopher D. Manning. (https://nlp.stanford.edu/projects/glove/)
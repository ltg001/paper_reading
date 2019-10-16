# Generative Code Modeling With Graphs

题目：Generative Code Modeling With Graphs  
作者：MarcBrockschmidt, MiltiadisAllamanis, AlexanderGaunt, OleksandrPolozov  
单位：Microsift Research  
出版：ICLR-2019  

## 问题

**TODO** 为每个任务配上例子

主要任务：根据给出的上下文信息 c 来产生符合相应描述的代码段。  

> The most general form of the code generation task is to produce a (partial) program in a programming language given some context information c.

* 上下文信息的形式
  * 自然语言
  * 输入输出的样例
  * 部分代码

提出的新任务 ExprGen：聚焦于让模型基于上下文信息产生少量但是语义复杂的表达式。

> a new code generation task focused on generating small but semantically complex expressions conditioned on source code context.

### 研究背景

1. 基于自然语言和形式语言的研究  
   自然语言类似的方法可以在源代码上取得一定的效果。在软件工程和语义分析上取得一定成就。  
   > shown successes on important software engineering tasks(Raychevetal.,2015;Bichseletal.,2016; Allamanis et al., 2018b) and semantic parsing (Yin & Neubig, 2017; Rabinovich et al., 2017).

   **TODO** 取得什么成就 & 评价指标  

    * 缺陷：只能满足部分语法要求，不能区分给定训练样本中的相似程序的不同之处。上下文的语义信息有缺失。
        > as they cannot distinguish unlikely from likely
        > sometimes fail to produce syntactically correct code

    **TODO** 反例

2. 抽象语法树相关的研究
   因为语法树保证了语法信息的正确性，所以可以通过目标语言的语法来构建抽象语法树来解决语法信息的正确问题。
   > using the target language’s grammar to generate abstract syntax trees

    **TODO** 图示  

    本文使用了建立抽象语法树的基本思路，并依据编程语言的语法来有序扩展语法树，通过每次扩展语法树最底层，最左边的非终结节点来有序构造。因为每次扩展的节点的相对位置一定，所以作者将代码产生问题简化为了树扩展序列的分类问题。
    > The key idea is to construct the AST a sequentially, by expanding one node at a time using production rules from the underlying programming language grammar. This simpliﬁes the code generation task to a sequence of classiﬁcation problems ...

## 方案

本文使用了经典的 encoder-decoder 结构，将提取到的上下文信息先表示为一个向量，再逐步将其展开并生成目标代码。原理图如下：

![img](model_sketch.png)

此结构同样保证了在生成 t 时刻的信息时考虑到前 t-1 时刻的信息，因为 decoder 同样使用了 RNN 的结构，保证了表示 t 时刻的单元接受到了前 t-1 时刻的状态信息，而在开始进行 decode 之前，encoder 已经提取出了上下文信息 c 并将其传入了 decoder。即原文中表述的公式：
$$p(a|c) = \prod_{t} p(a_t|c,a_{<t})$$

### encoder

* Method1 -- Seq  
    使用 NLP 领域中经典的信息抽取方式 Seq 使用两层双向的 GRU 单元来学习代码空缺处的上下文信息。使用双向的 GRU 单元同时学习空缺前和空缺后的语义信息，选取最后一个单元的状态信息作为传入的上下文信息 c。  
    作者使用了第二个上述的结构来学习变量在空缺前后的变化特征，并使用了上述结构来进行变量的表示学习，即将第二层的 GRU 最终状态信息经过平均池化后作为每个变量的向量表示。所以每一层的 GRU 单元数就是描述变量的个数。  
    ![img](seq.png) // **TODO** 换张图表示嵌入

  * 为什么使用 GRU?
    * RNN：每一层的基本单元只进行 tanh 或 relu 操作，如果网络层次太深的话，此时会产生梯度消失或梯度下降问题。这种神经网络带有环，可以将信息持久化。但是不能解决信息的持久化问题。
    * LSTM：可以在解决梯度消失和梯度爆炸的问题，还可以从语料中学习到长期依赖关系。但是参数较多，不容易训练且容易过拟合。
    * GRU：将遗忘门和输入门合并成为单一的“更新门”。相对 LSTM 引入了更少的参数，所以网络不容易过拟合。

* Method2 -- G

### decoder

## 原理

## 实现

## 评价

## 局限

1. 产生代码的类型受限，不支持用户自定的类型。
   >  we restrict ourselves to expressions that have Boolean, arithmetic or string type, or arraysof such types, excluding expressions of other typesor expressionsthat use project-speciﬁc APIs

## 展望

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

### encoder

* Method1 -- Seq
    使用 NLP 领域中经典的信息抽取方式 Seq 使用两层双向的 GRU 单元来学习代码空缺处的上下文信息。  
    作者使用了第二个上述的结构来学习变量在空缺前后的变化特征，并使用了上述结构来进行变量的表示学习，即将第二层的 GRU 最终状态信息经过平均池化后作为每个变量的向量表示。所以每一层的 GRU 单元数就是描述变量的个数。  
    ![img](seq.png) // **TODO** 换张图表示嵌入

  * 为什么使用 GRU **TODO** complete

* Method2 -- G

### decoder

## 原理

## 实现

## 评价

## 局限

1. 产生代码的类型受限，不支持用户自定的类型。
   >  we restrict ourselves to expressions that have Boolean, arithmetic or string type, or arraysof such types, excluding expressions of other typesor expressionsthat use project-speciﬁc APIs

## 展望

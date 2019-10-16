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

## 方案

## 原理

## 实现

## 评价

## 局限

1. 产生代码的类型受限，不支持用户自定的类型。
   >  we restrict ourselves to expressions that have Boolean, arithmetic or string type, or arraysof such types, excluding expressions of other typesor expressionsthat use project-speciﬁc APIs

## 展望

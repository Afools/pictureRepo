# Lecture Note for Data Visualization

## Overview 2024-8-12

- Prof Jianmin ZHENG Office:N4-02c-82 Tel:67906257 Email: ASJMZheng@ntu.edu.sg
- Prof Yong WANG Office:N4-02c-106 Tel 65183446 Email: yong-wang@ntu.edu.sg

15% 平时分 32% Final Quiz 22% Presentation

## Introduction

### Background

- Information Explosion
- Classification of Visualization

  - Data visualization
    - Scientific Visualization: Often scalar or vector fields from computer simulations.
    - Information Visualization: Abstract data (e.g., has no inherent physical form)
    - Software Visualization

- Sequence of Steps
  - data acquisition: conversion, formatting, cleaning
  - data enrichment: transformation, resampling, filtering
  - data mapping
  - rendering

### Syllabus

- Value of data visualization
- Data representation
- Visual encoding
- Visualization tools
- Basic plots and charts
- Visual perception
- Visualization design
- Interactive visualization
- Exploratory data analysis (correlation analysis)
- Exploratory data analysis (time series analysis)
- Geospatial visualization
- Abstract data visualization
- Scientific data visualization

## Data Representations

### Data models

- Data foundation
  - Data model
  - Conceptual model
  - Type of Data:
    - continues
    - discrete
  - Data attributes:
    - dimensions
    - measures
  - Level of measurement:NOIR

#### Level of measurement: NOIR

名义尺度（Nominal）、顺序尺度（Ordinal）、区间尺度（Interval）和比率尺度（Ratio）。这些等级代表了数据在测量和分类时的不同层次和复杂性。以下是对每个级别的解释：

- 名义尺度 (Nominal Scale)  
  定义：名义尺度用于对数据进行分类或命名，数据之间没有固有的顺序或大小比较。例如，性别（男性、女性）、血型（A 型、B 型、AB 型、O 型）、国家名称等。  
  特征：数据只能用于识别或区分不同类别，不能进行排序或算术运算。

- 顺序尺度 (Ordinal Scale)  
  定义：顺序尺度不仅可以对数据进行分类，还可以对其进行排序。数据间存在顺序关系，但间距不一定相等。例如，教育程度（小学、中学、大学）、满意度等级（非常不满意、不满意、中立、满意、非常满意）。  
  特征：数据可以进行排序，但不能确定两者之间的差值或做算术运算。
  <br/><br/>
- 区间尺度 (Interval Scale)  
  定义：区间尺度不仅具有顺序性，还具有相等的间隔，但没有绝对零点。例如，温度（摄氏度、华氏度）、智商（IQ）得分。  
  特征：数据之间的差值具有意义，可以进行加减运算，但由于没有绝对零点，不能进行乘除运算。

- 比率尺度 (Ratio Scale)  
  定义：比率尺度是最复杂的测量等级，具有所有前述尺度的特征，同时存在一个绝对零点。这意味着数据不仅可以排序和比较间隔，还可以进行所有算术运算（加、减、乘、除）。例如，重量、长度、年龄、收入。  
  特征：可以进行各种数学运算，因为比率尺度数据之间的比例是有意义的。

**当用 NOQ 进行区分时，Interval 和 Ratio 归类为 Q**

---
title: Learner Prerequisites
---

This page collects questions that learners can answer in order to judge if they bring all prequisites to the course. 

# Required Pre-Knowledge

- **Basic Machine Learning Knowledge** – Data Cleaning, Train & Test Split, Overfitting & Underfitting, Metrics (Accuracy, Recall, etc.),
- **Python** – Previous Experience programming in python is required (Refer to Python Data Carpentry Lesson )
- **Pandas** – Knowledge of the Pandas python package

# Pre-Knowledge Survey

## Data Cleaning Techniques 1

You are given a dataset from experiments that you want to use for machine learning (13 columns with 25000 rows). One column is particularly useful and is encoded as real numbers in a range from `-15` to `12`. You would like to normalize this data so that it fits into the range of real numbers between `0` and `1`. How would you do this?

1. I've performed such an operation multiple times. If you give me an editor, I can code this up in a minute.
2. I've done similar things in the past, I can copy & paste this over.
3. I am not sure what to do. I'd consult a colleague or google.
4. I don't know what to do.

## Data Cleaning Techniques 2

A collaborator collected a large dataset of 30 000 images in total. All images have been annotated with one class label by volunteers and hence the dataset is very precious to your colleague. However, the RGB images presented to you have differing sizes: 70% of them are of shape `1728x1492`, 25% are of shape `1920x1080` and 5% are of shape `1800x1600`. To start using machine learning, you are considering to make the dataset homogenous.

1. I've performed such an operation multiple times. If you give me an editor, I can code this up in a minute.
2. I've done similar things in the past, I can copy & paste this over.
3. I am not sure what to do. I'd consult a colleague or google.
4. I don't know what to do.

## Train & Test Split 

A fundamental ingredient of machine learning, is training an algorithm in such a way, that is can represent the distribution of the data as best as possible. For this, a training dataset is typically split up in several parts. You are receiving a piece of code from a colleague which performs this task on a dataset. For this, your peer has taken a table of 50 000 rows. The table was split at row ID `45000` and all rows up to this point have been stored as the training dataset. The remaining rows have been used as the test set. This design choice makes you uneasy.

1. I've observed this behavior in the past. If that happens, I discuss the matter with them immediately and suggest a solution.
2. I don't recall the exact reason, by I know from where I can copy & paste a fix.
3. I am not sure why this can produce an error. I'd consult a colleague or a search engine of my liking.
4. I am unclear why this is an error.

## Overfitting & Underfitting

A student intern project is handed over to you. While going through the project report, you notice the claim, that the machine learning algorithm used works optimally on the training data. The report describes, that the CNN in use has 450707 parameters. It was trained on 4000 samples. Each sample having 32 `float32` values to it. The prediction of the network fits perfectly on each datum of the test set. Moreover, the predictions even follow some of the noise in the data. This troubles your mind.

1. I've observed this behavior in the past. If that happens, I discuss the matter with them immediately and suggest a solution.
2. I don't recall the exact reason, by I know from where I can copy & paste a fix.
3. I am not sure why this can produce an error. I'd consult a colleague or a search engine of my liking.
4. I am unclear why this is an error.

## Metrics

You are given the task to classify data into three categories. After the first round of training your algorithm, your accuracy turns out at `61%` using only default parameters from example code taken from the web. You talk to a collaborator. While explaining your results casually over lunch, you are presented with the comment "Ah, this machine learning stuff is not worth a dime!". 

1. I've observed this behavior in the past. If that claim is brought forward, I explain why it is potentially based on wrong assumptions.
2. I don't recall the exact reason, by I know from where I can look up how to estimate the accuracy here.
3. I am not sure. I'd consult a colleague or a search engine of my liking.
4. I am unclear why this statement yields a misconception.

{% include links.md %}

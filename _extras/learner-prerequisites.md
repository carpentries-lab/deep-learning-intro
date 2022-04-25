---
title: Learner Prerequisites
---

This page collects questions that learners can answer in order to judge if they bring all prequisites to the course. Note this page is not meant as a full survey. It is rather provided to offer a resource to pick cherries from.

# Required Pre-Knowledge

- **Basic Machine Learning Knowledge** – Data Cleaning, Train & Test Split, Overfitting & Underfitting, Metrics (Accuracy, Recall, etc.),
- **Python** – Previous Experience programming in python is required (Refer to Python Data Carpentry Lesson )
- **Pandas** – Knowledge of the Pandas python package

# Pre-Knowledge Survey

For a motivation of this survey type, see [Greg Wilsons Template in Teaching Tech Together](https://teachtogether.tech/en/index.html#s:checklists-preassess).

## Python

### Lists 1

You are provided with a python list of integer values. The list has length 1024 and you would like to obtain all entries from index 50 to 101. How would you do this? 

1. I can do that. Give me an editor or notebook and I'll show you.
2. I'd need to look up the syntax in a cheatsheet or some old code and I'm good to do this.
3. I am unclear about this, I'd have to consult a colleague or a search engine to do this.
4. I am not sure what to do.


### Lists 2

You are provided a list of 512 random float values. These values range between 0 and 100. You would like to remove any entry in this list that is larger than 90 or smaller than 10. 

1. I can do that. Give me something that understands python and I'll show you.
2. I'd need to look up the syntax in a cheatsheet or some old code and I'm good to do this.
3. I am unclear about this, I'd have to consult a colleague or a search engine to do this.
4. I am not sure what to do.


## Pandas

### Data Cleaning Techniques 1

You are given a dataset from experiments that you want to use for machine learning (13 columns with 25000 rows). One column is particularly useful and is encoded as real numbers in a range from `-15` to `12`. You would like to normalize this data so that it fits into the range of real numbers between `0` and `1`. How would you do this?

1. If you give me an editor, I can code this up in a minute.
2. I've done similar things in the past, I can copy & paste this over.
3. I am not sure what to do. I'd consult a colleague or google.
4. I don't know what to do.

### Data Cleaning Techniques 2

A collaborator collected a large dataset of 30 000 images in total. All images have been annotated with one class label by volunteers and hence the dataset is very precious to your colleague. However, the RGB images presented to you have differing sizes: 70% of them are of shape `1728x1492`, 25% are of shape `1920x1080` and 5% are of shape `1800x1600`. To start using machine learning, you are considering to make the dataset homogenous.

1. If you give me an editor, I can code this up in a minute.
2. I've done similar things in the past, I can copy & paste this over.
3. I am not sure what to do. I'd consult a colleague or google.
4. I don't know what to do.

### CSV Files 1

You are provided with a CSV file. The file has 35000 rows. The file has 45 columns. Each column is separated by the "," symbol from the other. You would like to open the file in python, calculate the arithmetic mean, the minimum and maximum of column number 5, 12 and 39.   

1. If you give me an editor, I can code this up in a minute.
2. I've done similar things in the past, I can copy & paste this over.
3. I am not sure what to do. I'd consult a colleague or google.
4. I don't know what to do.

### CSV Files 2

You are provided with a CSV file. The file has 35000 rows. The file has 45 columns. Each column is separated by the "," symbol from the other. You would like to open the file in python, remove all entries where the value of column 21 is larger than 50 and replace them with `0`.  

1. If you give me an editor, I can code this up in a minute.
2. I've done similar things in the past, I can copy & paste this over.
3. I am not sure what to do. I'd consult a colleague or google.
4. I don't know what to do.

### Pragmatic Stats 

You are provided with a CSV file. The file has 35000 rows. The file has 45 columns. Each column is separated by the "," symbol from the other. When you load the file and plot the histogram of column 40, you are suspicious that the floating point values are not normally distributed. But, the producer of the CSV file assures you that all columns are normally distributed. To make sure, you sit down to code a function which tests any given column if it is normally distributed. 

1. If you give me an editor, I can code this up in a minute.
2. I've done similar things in the past, I can copy & paste this over.
3. I am not sure what to do. I'd consult a colleague or google.
4. I don't know what to do.


## ML

### Overfitting & Underfitting

A student intern project is handed over to you. While going through the project report, you notice the claim, that the machine learning algorithm used works optimally on the training data. The report describes, that the CNN in use has `450707` parameters. It was trained on `4000` samples. Each sample having 32 `float32` values to it. The prediction of the network fits perfectly on each datum of the test set. Moreover, the predictions even follows some of the noise in the data. 

1. There is something wrong with the network design or training procedure, and I can suggest a solution
2. There is something wrong with the network design or training procedure, but I don't know how to solve it
3. The network design and training procedure seem good, as it learned the training data well
4. I don't know how to judge whether the network design and training procedure are good

1. I've observed this behavior in the past. If that happens, I discuss the matter with them immediately and suggest a solution.
2. I don't recall the exact reason, by I know from where I can copy & paste a fix.
3. I am not sure why this can produce an error. I'd consult a colleague or a search engine of my liking.
4. I am unclear why this is an error.

### Metrics

You are given the task to classify data into three categories. After the first round of training your algorithm, your accuracy turns out at `61%` using only default parameters from example code taken from the web. You talk to a collaborator. While explaining your results casually, you are presented with the comment "Ah, this machine learning stuff is not worth a dime!". 

1. I disagree with this claim based on the accuracy value alone, and can argue why it is potentially based on wrong assumptions
2. I suspect this accuracy value alone is not enough to come this conclusion, but I have to look up how we should further evaluate
3. The collaborator is right, this accuracy is too low.
4. I have no idea whether the collaborator is right.

### Clustering

You are helping to organize a conference of more than 1000 attendants. All participants have already paid and are expecting to pick up their conference t-shirt on the first day. Your team is in shock as it discovers that t-shirt sizes have not been recorded during online registration. However, all participants were asked to provide their age, gender, body height and weight. To help out, you sit down to write a python script that predicts the t-shirt size for each participant using a clustering algorithm.

1. I can do that. Give me something that understands python and I'll show you.
2. I'd need to look up the syntax in a cheatsheet or some old code and I'm good to do this.
3. I am unclear about this, I'd have to consult a colleague or a search engine to do this.
4. I am not sure what to do.

{% include links.md %}

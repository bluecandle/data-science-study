# [Week 2] Deep Neural Networks for Time Series

---

## #Deep Neural Networks for Time Series

---

### &A conversation with Andrew Ng

---

Yeah, we'll start with a relatively simple DNN, and if you remember DNNs all the way back like at the beginning of the specialization, so you're going to be able to bring those skills to bear on time-series data. So we're going to build a very simple DNN just like a three-layer DNN.

Then one technique though that's I find really useful that you're going to learn this week, is being able to tune the learning rate of the optimizer. We'll all spent a lot of time hand-tuning learning rate.

So I think giving people a systematic way to do this, will be very useful. Yeah.

### &Preparing features and labels

---

First of all, as with any other ML problem, we have to divide our data into features and labels.

n this case our feature is effectively a number of values in the series, with our label being the next value. We'll call that number of values that will treat as our feature, the window size, where we're taking a window of the data and training an ML model to predict the next value.

So for example, if we take our time series data, say, 30 days at a time, we'll use 30 values as the feature and the next value is the label. Then over time, we'll train a neural network to match the 30 features to the single label.

additional parameter on the window called drop_remainder. if we set this to true, it will truncate the data by dropping all of the remainders. Namely, this means it will only give us windows of five items.

For each item in the list it kind of makes sense to have all of the values but the last one to be the feature, and then the last one can be the label. And this can be achieved with mapping, like this, where we split into everything but the last one with :-1, and then just the last one itself with -1:.

**[shuffle method]**

Typically, you would shuffle their data before training. And this is possible using the shuffle method.

**[batch method]**

Finally, we can look at batching the data, and this is done with the batch method.

It'll take a size parameter, and in this case it's 2. So what we'll do is we'll batch the data into sets of two

즉, y 값이 label 이고 x 값이 feature 라는거네 (원래 알던거랑 같은 이야기임.)

### &Preparing features and labels _ programming example

---

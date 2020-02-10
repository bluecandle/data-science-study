# [Week 3] Recurrent Neural Networks for Time Series

RNN LSTM 은 이전 코스에서 다루었다. (자연어 처리 과정에서) ⇒ 찾아서 듣기.

---

## #Recurrent Neural Networks for time series

---

### &Week 3 - A conversation with Andrew Ng

---

But time series is temporal data, seems like you should be applying a sequence model, like an RNN or an LCM to that.

So being able to use RNNs and LSTMs might factor that in to our data to give us a much more accurate prediction. >> Yeah, that's right, looking over a much bigger windows and carrying context from far away. >> Yeah, yeah, exactly, and you know my old favorite LSTMs, and the way they have that cell state, that allows you, we should call it L state, after me.

So for example, like financial data, today's closing price has probably got a bigger impact on tomorrow's closing price than the closing price from 30 days ago, or 60 days ago, or 90 days ago.

**So being able to use recurrent networks and LSTMs I think it will help us be able to be much more accurate in predicting seasonal data**

**[lambda layer]**

And so, but in Tensorflow and with Keras, Lambda layers allow us to write effectively an arbitrary piece of code as a layer in the neural network.

Basically a Lambda function, an unnamed function, but implemented as a layer in the neural network that resend the data, scales it.

### &Conceptual Overview

---

**[RNN, Recurrent Neural Network]**

This week, we're going to look at RNNs for the task of prediction. A Recurrent Neural Network, or RNN is a neural network that contains recurrent layers.

As you saw in the previous course, they could've been used for predicting text. Here we'll use them to process the time series. This example, will build an RNN that contains two recurrent layers and a final dense layer, which will serve as the output.

With an RNN, you can feed it in batches of sequences, and it will output a batch of forecasts, just like we did last week.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/07f1096b-0d55-4f79-ad3e-e25a240f74f9/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/07f1096b-0d55-4f79-ad3e-e25a240f74f9/Untitled.png)

At each time step, the memory cell takes the input value for that step. So for example, it is zero at time zero, and zero state input. It then calculates the output for that step, in this case Y0, and a state vector H0 that's fed into the next step. H0 is fed into the cell with X1 to produce Y1 and H1, which is then fed into the cell at the next step with X2 to produce Y2 and H2.

because the values recur due to the output of the cell, a one-step being fed back into itself at the next time step.

this is really helpful in determining states. The location of a word in a sentence can determine it semantics. Similarly, for numeric series, things such as closer numbers in the series might have a greater impact than those further away from our target value.

### &Shape of the inputs to the RNN

---

if we have a window size of 30 timestamps and we're batching them in sizes of four, the shape will be 4 times 30 times 1, and each timestamp, the memory cell input will be a four by one matrix, like this.

Now, in some cases, you might want to input a sequence, but you don't want to output on and you just want to get a single vector for each instance in the batch. This is typically called a sequence to vector RNN. But in reality, all you do is ignore all of the outputs, except the last one. When using Keras in TensorFlow, this is the default behavior. So if you want the recurrent layer to output a sequence, you have to specify returns sequences equals true when creating the layer. You'll need to do this when you stack one RNN layer on top of another.

### &Outputting a sequence

---

But notice the input_shape, it's set to None and 1. TensorFlow assumes that the first dimension is the batch size, and that it can have any size at all, so you don't need to define it. Then the next dimension is the number of timestamps, which we can set to none, which means that the RNN can handle sequences of any length. The last dimension is just one because we're using a unit vary of time series.

### &Lambda layers

---

layers that use the Lambda type. This type of layer is one that allows us to perform arbitrary operations to effectively expand the functionality of TensorFlow's kares, and we can do this within the model definition itself.

Since the time series values are in that order usually in the 10s like 40s, 50s, 60s, and 70s, then scaling up the outputs to the same ballpark can help us with learning.
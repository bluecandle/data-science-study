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

**[shuffle method]**

Next of course, is to shuffle the data. This is achieved with the shuffle method. This helps us to rearrange the data so as not to accidentally introduce a sequence bias. Multiple runs will show the data in different arrangements because it gets shuffled randomly.

**[Sequence Bias]**

Sequence bias is when the order of things can impact the selection of things. For example, if I were to ask you your favorite TV show, and listed "Game of Thrones", "Killing Eve", "Travellers" and "Doctor Who" in that order, you're probably more likely to select 'Game of Thrones' as you are familiar with it, and it's the first thing you see. Even if it is equal to the other TV shows. So, when training data in a dataset, we don't want the sequence to impact the training in a similar way, so it's good to shuffle them up.

### &Feeding **windowed dataset** into neural network

---

**[shuffle buffer]**

Once it's flattened, it's easy to shuffle it. You call a shuffle and you pass it the shuffle buffer. Using a shuffle buffer speeds things up a bit. So for example, if you have 100,000 items in your dataset, but you set the buffer to a thousand. It will just fill the buffer with the first thousand elements, pick one of them at random. And then it will replace that with the 1,000 and first element before randomly picking again, and so on. This way with super large datasets, the random element choosing can choose from a smaller number which effectively speeds things up.

### &Single layer neural network

---

I'm then going to create a single dense layer with its input shape being the window size. For linear regression, that's all you need. I'm using this approach. By passing the layer to a variable called L0, because later I'm want to print out its learned weights, and it's a lot easier for me to do that if I have a variable to refer to the layer for that.

*sgd ⇒ stochastic gradient descent*

I'd use this methodology instead of the raw string, so I can set parameters on it to initialize it such as the learning rate or LR and the momentum.

### &Machine learning on time windows

---

you consider the input window to be 20 values wide, then let's call them x0, x1, x2, etc, all the way up to x19. But let's be clear. That's not the value on the horizontal axis which is commonly called the x-axis, it's the value of the time series at that point on the horizontal axis. So the value at time t0, which is 20 steps before the current value is called x0, and t1 is called x1, etc. Similarly, for the output, which we would then consider to be the value at the current time to be the y.

### &Prediction

---

[**Machine Learned Linear Regression model]**

So we've trained our model to say that when it sees 20 values like this, the predicted next value is 49.08478. So if we want to plot our forecasts for every point on the time-series relative to the 20 points before it where our window size was 20, we can write code like this

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e715d42a-cdac-48f1-aad5-44186429422c/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e715d42a-cdac-48f1-aad5-44186429422c/Untitled.png)

### &More on single layer neural network

---

### &Deep neural network training, tuning and prediction

---

Wouldn't it be nice if we could pick the optimal learning rate instead of the one that we chose? We might learn more efficiently and build a better model. Now let's look at a technique for that that uses callbacks that you used way back in the first course.

What it will do is change the learning rates to a value based on the epoch number. So in epoch 1, it is 1 times 10 to the -8 times 10 to the power of 1 over 20.

We can then try to pick the lowest point of the curve where it's still relatively stable like this, and that's right around 7 times 10 to the -6.

So let's set that to be our learning rate and then we'll retrain.

but it's somewhat skewed by the fact that the earlier losses were so high. If we cropped them off and plot the loss for epochs after number 10 with code like this, then the chart will tell us a different story. We can see that the loss was continuing to decrease even after 500 epochs. And that shows that our network is learning very well indeed.

### &Deep neural network _ Programming example \ Quiz

---

### What’s the correct line of code to split an n column window into n-1 columns for features and 1 column for a label

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a2ab3e0c-fc24-45ff-b31e-18e1500c5393/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a2ab3e0c-fc24-45ff-b31e-18e1500c5393/Untitled.png)

### If time values are in time[], series values are in series[] and we want to split the series into training and validation at time 1000, what is the correct code?

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aad9b5d2-5209-4029-9332-af9c359e64fa/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aad9b5d2-5209-4029-9332-af9c359e64fa/Untitled.png)

### If you want to inspect the learned parameters in a layer after training, what’s a good technique to use?

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56f5b2b9-f296-4bcb-83b6-3ed85caf159b/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/56f5b2b9-f296-4bcb-83b6-3ed85caf159b/Untitled.png)

### If you want to amend the learning rate of the optimizer on the fly, after each epoch, what do you do?

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f391a79-d01f-42e9-a13a-ca91c2ed8bad/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f391a79-d01f-42e9-a13a-ca91c2ed8bad/Untitled.png)

### What does ‘drop_remainder=true’ do?

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38dfe31d-3345-4be7-b9ec-fe689190c6cb/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38dfe31d-3345-4be7-b9ec-fe689190c6cb/Untitled.png)

## #Exercise

---

### &answer

    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    print(tf.__version__)
    
    def plot_series(time, series, format="-", start=0, end=None):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(False)
    
    def trend(time, slope=0):
        return slope * time
    
    def seasonal_pattern(season_time):
        """Just an arbitrary pattern, you can change it if you wish"""
        return np.where(season_time < 0.1,
                        np.cos(season_time * 6 * np.pi),
                        2 / np.exp(9 * season_time))
    
    def seasonality(time, period, amplitude=1, phase=0):
        """Repeats the same pattern at each period"""
        season_time = ((time + phase) % period) / period
        return amplitude * seasonal_pattern(season_time)
    
    def noise(time, noise_level=1, seed=None):
        rnd = np.random.RandomState(seed)
        return rnd.randn(len(time)) * noise_level
    
    time = np.arange(10 * 365 + 1, dtype="float32")
    baseline = 10
    series = trend(time, 0.1)  
    baseline = 10
    amplitude = 40
    slope = 0.005
    noise_level = 3
    
    # Create the series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=51)
    
    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    
    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000
    
    plot_series(time, series)
    
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
      dataset = tf.data.Dataset.from_tensor_slices(series)
      dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
      dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
      dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
      dataset = dataset.batch(batch_size).prefetch(1)
      return dataset
    
    dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, input_shape=[window_size], activation="relu"), 
        tf.keras.layers.Dense(10, activation="relu"), 
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
    model.fit(dataset,epochs=100,verbose=0)
    
    forecast = []
    for time in range(len(series) - window_size):
      forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
    
    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]
    
    
    plt.figure(figsize=(10, 6))
    
    plot_series(time_valid, x_valid)
    plot_series(time_valid, results)
    
    tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
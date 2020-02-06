# [Week 1] Sequences and Prediction

---

Hi Learners and welcome to this course on sequences and prediction! In this course we'll take a look at some of the unique considerations involved when handling sequential time series data -- where values change over time, like the temperature on a particular day, or the number of visitors to your web site. We'll discuss various methodologies for predicting future values in these time series, building on what you've learned in previous courses!

## #Introduction

---

### &Introduction, A conversation with Andrew Ng

---

So you have that seasonality of data. You can also, in some cases, have trends of data, like whether it probably doesn't really trend although we could argue that it strangely enough idea with climate change, but like a stock data may trend upwards over time or downwards over some other times, and then of course the random factor that makes it hard to predict is noise.

So in this course, you'll start by learning about sequence models, a time series data, first practicing these skills and building these models on artificial Data, and then at the end of this course, you get to take all these ideas and apply them to the exciting problem of molding sunspot activity.

## #Sequences and Prediction

---

### &Time Series Examples

---

We'll go through some examples of different types of time series, as well as looking at basic forecasting around them.

You'll also start preparing time series data for machine learning algorithms. For example, how do you split time series data into training, validation, and testing sets? We'll explore some best practices and tools around that to get you ready for week 2, where you'll start looking at forecasting using a dense model, and how it differs from more naive predictions based on simple numerical analysis of the data.

In week 3, we'll get into using recurrent neural networks to forecast time series. We'll see the stateless and stateful approaches, training on windows of data, and you'll also get hands-on in forecasting for yourself.

Finally, in week 4, you'll add convolutions to the mix and put everything you've worked on together to start forecasting some real world data, and that's measurements of sunspot activity over the last 250 years.

**[univariate]**

there is a single value at each time step, and as a results, the term univariate is used to describe them.

**[Multivariate]**

You may also encounter time series that have multiple values at each time step. As you might expect, they're called Multivariate Time Series. Multivariate Time Series charts can be useful ways of understanding the impact of related data.

### &Machine learning applied to time series

---

**[Imputation]**

In some cases, you might also want to project back into the past to see how we got to where we are now.

This process is called imputation. Now maybe you want to get an idea for what the data would have been like had you been able to collect it before the data you already have.

### &Common patterns in time series

---

Time-series come in all shapes and sizes, but there are a number of very common patterns. So it's useful to recognize them when you see them.

There's no trend and there's no seasonality. The spikes appear at random timestamps. You can't predict when that will happen next or how strong they will be. But clearly, the entire series isn't random. Between the spikes there's a very deterministic type of decay. We can see here that the value of each time step is 99 percent of the value of the previous time step plus an occasional spike.This is an auto correlated time series. Namely it correlates with a delayed copy of itself often called a lag.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/deb20575-43e9-4e66-9507-ac2efc49df41/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/deb20575-43e9-4e66-9507-ac2efc49df41/Untitled.png)

**[Stationary & non-stationary]**

Their behavior can change drastically over time. For example, this time series had a positive trend and a clear seasonality up to time step 200. But then something happened to change its behavior completely. If this were stock, price then maybe it was a big financial crisis or a big scandal or perhaps a disruptive technological breakthrough causing a massive change.

After that the time series started to trend downward without any clear seasonality. We'll typically call this a non-stationary time series. To predict on this we could just train for limited period of time.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/049e8c0d-eaca-40f2-b4b2-6175b73b2c6b/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/049e8c0d-eaca-40f2-b4b2-6175b73b2c6b/Untitled.png)

But for time series forecasting it really depends on the time series. If it's stationary, meaning its behavior does not change over time, then great. The more data you have the better. But if it's not stationary then the optimal time window that you should use for training will vary.

### &Introduction to time series

---

- seasonal pattern
- autocorrelation : 자기상관(自己相關) _ 시간적으로 배열된 관측치의 계열의 값 사이의 내부 상관; {xt} 라 할 때, xt와 xt + xt + k 사이의 상관 계수.

### &Train, validation and test sets

---

**[Fixed Partitioning]**

For example, one year, or two years, or three years, if the time series has a yearly seasonality. You generally don't want one year and a half, or else some months will be represented more than others. While this might appear a little different from the training validation test, that you might be familiar with from non-time series data sets. Where you just picked random values out of the corpus to make all three, you should see that the impact is effectively the same. Next you'll train your model on the training period, and you'll evaluate it on the validation period.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1d34bcb-7c57-4399-8ea4-c8cbc6cf360a/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a1d34bcb-7c57-4399-8ea4-c8cbc6cf360a/Untitled.png)

**[Roll-forward partitioning]**

We start with a short training period, and we gradually increase it, say by one day at a time, or by one week at a time. At each iteration, we train the model on a training period. And we use it to forecast the following day, or the following week, in the validation period. And this is called roll-forward partitioning. You could see it as doing fixed partitioning a number of times, and then continually refining the model as such. F

### &Metrics for evaluating performance

---

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ac319a80-0c8f-4c15-90fd-b4683f2a40af/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ac319a80-0c8f-4c15-90fd-b4683f2a40af/Untitled.png)

- mse : mean squared error ( most common )
    - Well, the reason for this is to get rid of negative values. So, for example, if our error was two above the value, then it will be two, but if it were two below the value, then it will be minus two. These errors could then effectively cancel each other out, which will be wrong because we have two errors and not none. But if we square the error of value before analyzing, then both of these errors would square to four, not canceling each other out and effectively being equal.
- rmsed : root mse
- mae : mean absolute error
    - it's also called the main absolute deviation or mad. And in this case, instead of squaring to get rid of negatives, it just uses their absolute value.
    - For example, if large errors are potentially dangerous and they cost you much more than smaller errors, then you may prefer the mse. But if your gain or your loss is just proportional to the size of the error, then the mae may be better.
- mape : mean absolute percentage error _ mean ratio between the absolute error and the absolute value, this gives an idea of the size of the errors compared to the values.

### &Moving average and differencing

---

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/081140c4-8544-4f90-b8a7-d97aac8c7a66/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/081140c4-8544-4f90-b8a7-d97aac8c7a66/Untitled.png)

The idea here is that the yellow line is a plot of the average of the blue values over a fixed period called an averaging window, for example, 30 days.

**[Differencing]**

One method to avoid this is to remove the trend and seasonality from the time series with a technique called differencing.

So instead of studying the time series itself, we study the difference between the value at time T and the value at an earlier period.

You may have noticed that our moving average removed a lot of noise but our final forecasts are still pretty noisy. Where does that noise come from? Well, that's coming from the past values that we added back into our forecasts. So we can improve these forecasts by also removing the past noise using a moving average on that. If we do that, we get much smoother forecasts.

### &Trailing versus centered windows

---

Then moving averages using centered windows can be more accurate than using trailing windows. But we can't use centered windows to smooth present values since we don't know future values. However, to smooth past values we can afford to use centered windows.

### &Forecasting

---

가장 간단한 방법을 사용하여 forecast 를 한 후에, 거기서 나온 수치를 baseline 으로 잡고 더 나은 방법들을 시도해보는 식으로 진행한다!

Remember, for errors lower as better.

So now, if we calculate a moving average on this data, we'll see a relatively smooth moving average not impacted by seasonality. Then if we add back the past values to this moving average, we'll start to see a pretty good prediction. The orange line is quite close to the blue one

But all we did was just add in the raw historic values which are very noisy. What if, instead, we added in the moving average of the historic values, so we're effectively using two different moving averages?

Now, our prediction curve is a lot less noisy and the predictions are looking pretty good.

### &Quiz

---

noise : unexpected change in time-series data 로 이해하는게 가장 올바른 접근이다.

## #Weekly Exercise-Create and predict synthetic data

---
# [Week 1] Introduction

---

### &A primer in machine learning

---

*primer : 기본

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/67e89add-6ba9-4638-8fbc-5c4efc8d5cde/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/67e89add-6ba9-4638-8fbc-5c4efc8d5cde/Untitled.png)

pattern recognition

A neural network is just a slightly more advanced implementation of machine learning and we call that deep learning. But fortunately it's actually very easy to code.

### &The ‘Hello World’ of neural networks

---

A neural network is basically a set of functions which can learn patterns.

In keras, you use the word dense to define a layer of connected neurons. There's only one dense here. So there's only one layer and there's only one unit in it, so it's a single neuron. Successive layers are defined in sequence, hence the word sequential. But as I've said, there's only one. So you have a single neuron.

but the nice thing for now about TensorFlow and keras is that a lot of that math is implemented for you in functions. There are two function roles that you should be aware of though and these are loss functions and optimizers. This code defines them. I like to think about it this way. The neural network has no idea of the relationship between X and Y, so it makes a guess. Say it guesses Y equals 10X minus 10. It will then use the data that it knows about, that's the set of Xs and Ys that we've already seen to measure how good or how bad its guess was.

*optimizers

Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses. How you should change your weights or learning rates of your neural network to reduce the losses is defined by the optimizers you use.

⇒ generate a new and improved guess

*What does a Loss function do?

Measures how good the current 'guess' is

the optimizer is SGD which stands for stochastic gradient descent. If you want to learn more about these particular functions, as well as the other options that might be better in other scenarios, check out the TensorFlow documentation.

When using neural networks, as they try to figure out the answers for everything, they deal in probability. You'll see that a lot and you'll have to adjust how you handle answers to fit.

Now while this might seem very simple, you’ve actually gotten the basics for how neural networks work. As your applications get more complex, you’ll continue to use the same techniques.

### &Working through ‘Hello World’ in TensorFlow and Python

---

Now I'm going to compile the neural network using the loss function and the optimizer. Remember, these help the neural network guess the pattern, measure how well or how badly the guess performed, before trying again on the next epoch, and slowly getting more accurate.

These are fed in using NumPy arrays.

*Numpy array

A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

    import numpy as np
    
    # Create the following rank 2 array with shape (3, 4)
    # [[ 1  2  3  4]
    #  [ 5  6  7  8]
    #  [ 9 10 11 12]]
    a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    
    # Use slicing to pull out the subarray consisting of the first 2 rows
    # and columns 1 and 2; b is the following array of shape (2, 2):
    # [[2 3]
    #  [6 7]]
    b = a[:2, 1:3]
    
    # A slice of an array is a view into the same data, so modifying it
    # will modify the original array.
    print(a[0, 1])   # Prints "2"
    b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
    print(a[0, 1])   # Prints "77"
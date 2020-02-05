# [Week2]Introduction to Computer Vision

---

## #Introduction to Computer Vision

---

### &A Conversation with Andrew Ng

---

One of the non-intuitive things about vision is that it's so easy for a person to look at you and say, you're wearing a shirt, it's so hard for a computer to figure it out. Because it's so easy for humans to recognize objects, it's almost difficult to understand why this is a complicated thing for a computer to do.

### &An Introduction to computer vision

---

Machine Learning depends on having good data to train a system with.

### &Writing code to load training data

---

While this image is an ankle boot, the label describing it is the number nine. Now, why do you think that might be? There's two main reasons. First, of course, is that computers do better with numbers than they do with texts. Second, importantly, is that this is something that can help us reduce bias. If we labeled it as an ankle boot, we would be of course biasing towards English speakers. But with it being a numeric label, we can then refer to it in our appropriate language be it English, Chinese, Japanese, or here, even Irish Gaelic.

### &Coding a Computer Vision Neural Network

---

Remember last time we had a sequential with just one layer in it. Now we have three layers. The important things to look at are the first and the last layers. The last layer has 10 neurons in it because we have ten classes of clothing in the dataset.They should always match. The first layer is a flatten layer with the input shaping 28 by 28.

You can also tune the neural network by adding, removing and changing layer size to see the impact.

### &Walk through a Notebook for computer vision

---

But a better measure of performance can be seen by trying the test data. These are images that the network has not yet seen. You would expect performance to be worse, but if it's much worse, you have a problem.

As you can see, it's about 0.345 loss, meaning it's a little bit less accurate on the test set. It's not great either, but we know we're doing something right.

url : [https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course 1 - Part 4 - Lesson 2 - Notebook.ipynb](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb)

[Google Colaboratory](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb)

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

**Sequential**: That defines a SEQUENCE of layers in the neural network

**Flatten**: Remember earlier where our images were a square, when you printed them out? Flatten just takes that square and turns it into a 1 dimensional set.

**Dense**: Adds a layer of neurons

Each layer of neurons need an **activation function** to tell them what to do. There's lots of options, but just use these for now.

**Relu** effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.

**Softmax** takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] -- The goal is to save a lot of coding!

    model.compile(optimizer = tf.train.AdamOptimizer(),
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(training_images, training_labels, epochs=5)
    
    model.evaluate(test_images, test_labels)

### &Using Callbacks to control training

---

What we'll now do is write a callback in Python.

It's implemented as a separate class, but that can be in-line with your other code. It doesn't need to be in a separate file. In it, we'll implement the on_epoch_end function, which gets called by the callback whenever the epoch ends. It also sends a logs object which contains lots of great information about the current state of training. For example, the current loss is available in the logs, so we can query it for certain amount. For example, here I'm checking if the loss is less than 0.4 and canceling the training itself.

First, we instantiate the class that we just created, we do that with this code. Then, in my model.fit, I used the callbacks parameter and pass it this instance of the class.

### &Walk through a notebook with Callbacks

---

It's good practice to do this, because with some data and some algorithms, the loss may vary up and down during the epoch, because all of the data hasn't yet been processed. So, I like to wait for the end to be sure.

### &Quiz

---

What parameter to you set in your fit function to tell it to use callbacks?

⇒ callbacks = <~>

## #NN to recognize handwritten digits

---

### &Exercise 2 (Handwriting Recognition)

    import tensorflow as tf
    from os import path, getcwd, chdir
    
    # DO NOT CHANGE THE LINE BELOW. If you are developing in a local
    # environment, then grab mnist.npz from the Coursera Jupyter Notebook
    # and place it inside a local folder and edit the path to that location
    path = f"{getcwd()}/../tmp2/mnist.npz"
    
    # GRADED FUNCTION: train_mnist
    def train_mnist():
        # Please write your code only where you are indicated.
        # please do not remove # model fitting inline comments.
    
        # YOUR CODE SHOULD START HERE
        class myCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if(logs.get('acc')>0.99):
                      print("\nReached 99% accuracy so cancelling training!")
                      self.model.stop_training = True
        # YOUR CODE SHOULD END HERE
    
        mnist = tf.keras.datasets.mnist
    
        (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
        # YOUR CODE SHOULD START HERE
        callbacks = myCallback()
        # YOUR CODE SHOULD END HERE
        model = tf.keras.models.Sequential([
            # YOUR CODE SHOULD START HERE
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            # YOUR CODE SHOULD END HERE
        ])
    
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # model fitting
        history = model.fit(# YOUR CODE SHOULD START HERE
                x_train, y_train, epochs=10, callbacks=[callbacks]
                  # YOUR CODE SHOULD END HERE
        )
        # model fitting
        return history.epoch, history.history['acc'][-1]
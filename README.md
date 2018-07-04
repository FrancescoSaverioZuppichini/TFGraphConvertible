
# Serialize Class to TensorFlow Graph
### Francesco Saverio Zuppichini
Would it be cool to automatically bind class fields to tensorflow variables in a graph and restore them without manually get each variable back from it?

Image you have a `Model` class


```python
import tensorflow as tf

class Model():
    def __init__(self):
        self.variable = None
    def __call__(self):
        self.variable = tf.Variable([1], name='variable')
```

    /usr/local/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


Usually, you first **build** your model and then you **train** it. After that, you want to **get** from the saved graph the old variables without rebuild the whole model from scratch.

```python
tf.reset_default_graph()

model = Model()
model() # now  model.variable exists
print(model.variable)
```

    <tf.Variable 'variable:0' shape=(1,) dtype=int32_ref>


Now, imagine we have just trained our model and we want to store it. The usual pattern is


```python
EPOCHS = 10

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(EPOCHS):
        # train
        pass
    saver.save(sess,'/tmp/model.ckpt')
```

Now you want to perform **inference**, aka get your stuff back, by loading the stored graph. In our case, we want the variable named `variable`


```python
# reset the graph
tf.reset_default_graph()

with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format('/tmp/model.ckpt'))
        saver.restore(sess, '/tmp/model.ckpt')
```

    INFO:tensorflow:Restoring parameters from /tmp/model.ckpt


Now we can get back our `variable` from the graph


```python
graph = tf.get_default_graph()
variable = graph.get_operation_by_name('variable')
print(variable)
```

    name: "variable"
    op: "VariableV2"
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 1
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
    


But, what if we want to use our `model` class again? If we try now to call `model.variable` we get None


```python
model = Model() # recreate the model
print(model.variable)
```

    None


One solution is to **build again** the whole model and restore the graph after that


```python
# reset the graph
tf.reset_default_graph()

with tf.Session() as sess:
        model = Model()
        model()
        saver = tf.train.import_meta_graph("{}.meta".format('/tmp/model.ckpt'))
        saver.restore(sess, '/tmp/model.ckpt')
        print(model.variable)
```

    INFO:tensorflow:Restoring parameters from /tmp/model.ckpt
    <tf.Variable 'variable:0' shape=(1,) dtype=int32_ref>


You can already see that is a big waste of time.  We can bind `model.variable` directly to the correct graph node by


```python
model = Model()
model.variable  = graph.get_operation_by_name('variable')
print(model.variable)

```

    name: "variable"
    op: "VariableV2"
    attr {
      key: "container"
      value {
        s: ""
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT32
      }
    }
    attr {
      key: "shape"
      value {
        shape {
          dim {
            size: 1
          }
        }
      }
    }
    attr {
      key: "shared_name"
      value {
        s: ""
      }
    }
    


Now image we have a very big model with nested variables. In order to correct restore each variable pointer in the model you need to:

* name each variable
* get the variables back from the graph 

Would it be cool if we can automatically retrieve all the variables setted as a field in the Model class? 

## TFGraphConvertible

I have created a class, called `TFGraphConvertible`. You can use the `TFGraphConvertible` to automatically **serialize** and **deserialize**" a class.

Let's recreate our model


```python
from TFGraphConvertible import TFGraphConvertible
class Model(TFGraphConvertible):
    def __init__(self):
        self.variable = None
    def __call__(self):
        self.variable = tf.Variable([1], name='variable')
        
model = Model()
model()
```

It exposes two methods: `to_graph` and `from_graph` 

### Serialize - to_graph
In order to **serialize a class** you can call the **to_graph** method that creates a dictionary of field names -> tensorflow variables name. You need to pass a `fields` arguments, a dictionary of what field we want to serialize. In our case, we can just pass all of them.


```python
serialized_model = model.to_graph(model.__dict__)
print(serialized_model)
```

    {'variable': 'variable_2:0'}


It will create a dictionary with all the fields as keys  and the corresponding tensorflow variables name as values

### Deserialize - from_graph
 In order to **deserialize a class** you can call the **from_graph** method that takes the previous created dictionary and bind each class fields to the correct tensorflow variables


```python
model = Model() # simulate an empty model
print(model.variable)
model.from_graph(serialized_model, tf.get_default_graph())
model.variable # now it exists again
```

    None





    <tf.Tensor 'variable_2:0' shape=(1,) dtype=int32_ref>



And now you have your `model` back!

## Full Example

Let's see a more interesting example! We are going to train/restore a model for the MNIST dataset


```python
class MNISTModel(Model):

    def __call__(self, x, y, lr=0.001):
        self.x = tf.cast(x, tf.float32)
        self.x = tf.expand_dims(self.x, axis=-1)  # add grey channel

        self.lr = lr

        self.y = tf.one_hot(y, N_CLASSES, dtype=tf.float32)

        out = tf.layers.Conv2D(filters=32, kernel_size=5, activation=tf.nn.relu, padding="same", )(self.x)
        out = tf.layers.MaxPooling2D(2, strides=2)(out)
        out = tf.layers.Dropout(0.2)(out)
        out = tf.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu, padding="same", )(out)
        out = tf.layers.MaxPooling2D(2, strides=2)(out)
        out = tf.layers.Dropout(0.2)(out)
        out = tf.layers.flatten(out)
        out = tf.layers.Dense(units=512, activation=tf.nn.relu)(out)
        out = tf.layers.Dropout(0.2)(out)
        self.forward_raw = tf.layers.Dense(units=N_CLASSES)(out)
        forward = tf.nn.softmax(out)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.forward_raw, -1), tf.argmax(self.y, -1)), tf.float32))
        
        self.loss = self.get_loss()
        self.train_step = self.get_train()
        
        return forward

    def get_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.forward_raw))
        return loss

    def get_train(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss)

mnist_model = MNISTModel()
```

Let's get the dataset!


```python
from keras.datasets import mnist

tf.reset_default_graph()

N_CLASSES = 10

train, test = mnist.load_data()

x_, y_ = tf.placeholder(tf.float32, shape=[None, 28, 28]), tf.placeholder(tf.uint8, shape=[None])

train_dataset = tf.data.Dataset.from_tensor_slices((x_, y_)).batch(64).shuffle(10000).repeat()
test_dataset = tf.data.Dataset.from_tensor_slices((x_, y_)).batch(64).repeat()

iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                       train_dataset.output_shapes)
x, y = iter.get_next(name='iter_next')

train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)

```

    Using TensorFlow backend.


Now it is time to train it


```python

with tf.Session() as sess:

    mnist_model(x, y) # build the model
    
    sess.run(tf.global_variables_initializer())
    sess.run(train_init_op, feed_dict={x_: train[0], y_: train[1]})
    
    saver = tf.train.Saver()

    for i in range(150):
        acc, _ = sess.run([mnist_model.accuracy, mnist_model.train_step])
        if i % 15 == 0:
            print(acc)
            saver.save(sess,'/tmp/model.ckpt')
```

    0.125
    0.46875
    0.8125
    0.953125
    0.828125
    0.890625
    0.796875
    0.9375
    0.953125
    0.921875


Perfect! Let's store the serialized model in memory


```python
serialized_model = mnist_model.to_graph(mnist_model.__dict__)
print(serialized_model)
```

    {'x': 'ExpandDims:0', 'y': 'one_hot:0', 'forward_raw': 'dense_1/BiasAdd:0', 'accuracy': 'Mean:0', 'loss': 'Mean_1:0', 'train_step': 'Adam'}


Then we reset the graph and recreat the model


```python
tf.reset_default_graph()

mnist_model = MNISTModel()

with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format('/tmp/model.ckpt'))
        saver.restore(sess, '/tmp/model.ckpt')
        graph = tf.get_default_graph()

```

    INFO:tensorflow:Restoring parameters from /tmp/model.ckpt


Of course, our variables in the `mnist_model` do not exist


```python
mnist_model.accuracy
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-21-9def5e0d8f6c> in <module>()
    ----> 1 mnist_model.accuracy
    

    AttributeError: 'MNISTModel' object has no attribute 'accuracy'


Let's recreate them by calling the `from_graph` method.


```python
mnist_model.from_graph(serialized_model, tf.get_default_graph())
mnist_model.accuracy
```




    <tf.Tensor 'Mean:0' shape=() dtype=float32>



Now `mnist_model` is ready  to go, let's see the accuracy on a bacth of the test set


```python
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("{}.meta".format('/tmp/model.ckpt'))
    saver.restore(sess, '/tmp/model.ckpt')
    graph = tf.get_default_graph()
    x, y = graph.get_tensor_by_name('iter_next:0'), graph.get_tensor_by_name('iter_next:1')
    print(sess.run(mnist_model.accuracy, feed_dict={x: test[0][0:64], y: test[1][0:64]}))
```

    INFO:tensorflow:Restoring parameters from /tmp/model.ckpt
    1.0


## Conclusion
With this tutorial we have seen how to serialize a class and bind each field back to the correct tensor in the tensorflow graph. Be aware that you can store the `serialized_model` in `.json` format and load it directly where you need. In this way, you can directly create your model by using Object Oriented Programming and retrieve all the variales inside them without having to rebuild them.

Thank you for reading

Francesco Saverio Zuppichini

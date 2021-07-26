import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import activations
from tensorflow.keras import regularizers

# Python-Klasse erzeugt eine Schicht, in der die Gewichte bzw. Verschiebungsvektoren manuel als trainierbar oder nicht trainierbar 
# eingestellt werden können. Trainierbare Gewichte bzw. Verschiebungsvektoren können mit matW bzw. vecB eingestellt werden.
class SparseDense(Layer):

   def __init__(self,matW,vecB,activation=None,kernel_regularizer=None,bias_regularizer=None,**kwargs):
       super(SparseDense, self).__init__()
       self.matW = matW
       self.vecB = vecB
       self.activation = activations.get(activation)
       self.kernel_regularizer = regularizers.get(kernel_regularizer)
       self.bias_regularizer = regularizers.get(bias_regularizer)


   def add_w(self, trainable=1):
       if trainable == 1:
           return self.add_weight(shape=(1,), initializer=tf.keras.initializers.random_uniform(0,0.05), trainable=True)
       if trainable == 0:
           return self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=0), trainable=False)

   def add_b(self, trainable=1):
       if trainable == 1:
           return self.add_weight(shape=(1,), initializer=tf.keras.initializers.random_uniform(0,0.05), trainable=True)
       if trainable == 0:
           return self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=0), trainable=False)


   # jede Schicht wird gemäß der Eingabe matW,vecB aufgebaut
   def build(self, input_shape=None):
       self.sW = [[self.add_w(t) for t in T] for T in self.matW]
       self.sB = [self.add_b(t) for t in self.vecB]

   def call(self, inputs):
       self.w = tf.transpose(tf.concat(self.sW, 1))
       self.b = tf.concat(self.sB, 0)
       return self.activation(tf.matmul(inputs, self.w) + self.b)

   def get_config(self):
       config = super(SparseDense, self).get_config()
       config.update({
           'bias_regularizer': regularizers.serialize(self.bias_regularizer),
           'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)
       })
       return config

# erzeugt eine dünn besetzte Schicht
# trainierbare Gewichte  mit matW  eingestellt werden
class OutputDense(Layer):

   def __init__(self,matW,activation=None,kernel_regularizer=None,bias_regularizer=None,**kwargs):
       super(OutputDense, self).__init__()
       self.matW = matW
       self.activation = activations.get(activation)
       self.kernel_regularizer = regularizers.get(kernel_regularizer)
       self.bias_regularizer = regularizers.get(bias_regularizer)


   def add_w(self, trainable=1):
       if trainable == 1:
           return self.add_weight(shape=(1,), initializer=tf.keras.initializers.random_uniform(0,0.05), trainable=True)
       if trainable == 0:
           return self.add_weight(shape=(1,), initializer=tf.keras.initializers.Constant(value=0), trainable=False)

   # jede Schicht wird gemäß der Eingabe matW aufgebaut
   def build(self, input_shape=None):
       self.sW = [[self.add_w(t) for t in T] for T in self.matW]

   def call(self, inputs):
       self.w = tf.transpose(tf.concat(self.sW, 1))
       return tf.matmul(inputs, self.w)

   def get_config(self):
       config = super(OutputDense, self).get_config()
       config.update({
           'bias_regularizer': regularizers.serialize(self.bias_regularizer),
           'kernel_regularizer': regularizers.serialize(self.kernel_regularizer)
       })
       return config

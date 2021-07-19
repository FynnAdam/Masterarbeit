import tensorflow as tf
import numpy as np
from tensorflow.python.keras import activations
from DenseNew import SparseDense
from DenseNew import OutputDense

# Klasse für neuronales Netz mit Tiefe L, Schichten p, anzahl nicht-null Parameter s (im EW)
class NeuronalNetwork():
    # sequential deep network

  def __init__(self,L,p,s,**kwargs):
    super(NeuronalNetwork, self).__init__(**kwargs)
      # L: Tiefe des Netzes (Anzahl verdeckte Schicheten)
      # p: Knoten je Schicht (Vektor mit L+2 Einträgen)
      # s: nicht-null Parameter (Zahl zwischen >>1 und Anzahl Gesamtparameter)


    [d,*hidden,output] = p
    T = sum([(p[i-1]+1)*p[i] for i in range(1,L+2,1)])-output # Anzahl Gesamtparameter
    T=T-d*hidden[0]-hidden[-1]
    s= s-d*hidden[0]-hidden[-1]-L # Ausgleich, da Gewichte der ersten und letzte Schicht wird im Netz auf trainierbar gesetzt

    self.NN = tf.keras.Sequential()

    # Eingabeschicht
    self.NN.add(tf.keras.layers.InputLayer(input_shape=(d,)))

    # 1. verdeckte Schicht
    matW = np.random.binomial(1, 1, d*hidden[0]).reshape((d, hidden[0]))
    vecB = np.random.binomial(1, min(1,s / T), hidden[0])
    self.NN.add(SparseDense(matW=matW, vecB=vecB, activation=activations.relu))

    # verdeckte Schichten
    for width in hidden[:-1]:
        matW = np.random.binomial(1,min(1,s / T),width*width).reshape((width,width))
        matW[0][0] = 1
        vecB = np.random.binomial(1,min(1,s / T),width)
        self.NN.add(SparseDense(matW=matW,vecB=vecB,activation = activations.relu))

    # letzte verdeckte Schicht
    matW = np.random.binomial(1, 1, hidden[-1]).reshape((hidden[-1], 1))
    self.NN.add(OutputDense(matW=matW))

  def compile(self,optimizer,loss,metric):
      self.NN.compile(loss=loss, optimizer=optimizer, metrics=metric)

  def fit(self,dataX,dataY,epochs,batch):
      self.NN.fit(dataX,dataY,epochs=epochs,batch_size=batch,use_multiprocessing=False)
      weights_fit = [np.array(max(min(w, 1),-1), dtype=np.float32).reshape(w.shape) for w in self.NN.get_weights()]
      self.NN.set_weights(weights_fit)

class NeuronalNetwork_standard():
    # sequential deep network

  def __init__(self,L,p,**kwargs):
    super(NeuronalNetwork_standard, self).__init__(**kwargs)
      # L: Tiefe des Netzes (Anzahl verdeckte Schicheten)
      # p: Knoten je Schicht

    [d,*hidden,output] = p
    T = sum([(p[i-1]+1)*p[i] for i in range(1,L+2,1)])-output # Anzahl Gesamtparameter
    T=T-d*hidden[0]-hidden[-1]

    self.NN = tf.keras.Sequential()
    # Eingabeschicht
    self.NN.add(tf.keras.layers.InputLayer(input_shape=(d,)))
    # verdeckte Schichten
    for width in hidden:
        self.NN.add(tf.keras.layers.Dense(width,'relu',kernel_initializer=tf.keras.initializers.random_uniform(0,0.05),bias_initializer=tf.keras.initializers.zeros))
    # Ausgabeschicht
    self.NN.add(tf.keras.layers.Dense(1))

  def compile(self,optimizer,loss,metric):
      self.NN.compile(loss=loss, optimizer=optimizer, metrics=metric)

  def fit(self,dataX,dataY,epochs,batch):
      self.NN.fit(dataX,dataY,epochs=epochs,batch_size=batch,use_multiprocessing=False)

def sig(a,b,size):
    return np.random.uniform(a,b,size=size) # uniform distributed

def noisy_sig(x,sigma):
    return x + np.random.normal(0,sigma,len(x)) #error

# Klasse für neuronales Netz mit Tiefe L, Schichten p
class NeuronalNetwork_standard2():
    # sequential deep network

  def __init__(self,L,p,**kwargs):
    super(NeuronalNetwork_standard2, self).__init__(**kwargs)
      # L: Tiefe des Netzes (Anzahl verdeckte Schicheten)
      # p: Knoten je Schicht (Vektor mit L+2 Einträgen)
      # s: nicht-null Parameter (Zahl zwischen >>1 und Anzahl Gesamtparameter)


    [d,*hidden,output] = p
    T = sum([(p[i-1]+1)*p[i] for i in range(1,L+2,1)])-output # Anzahl Gesamtparameter
    T=T-d*hidden[0]-hidden[-1]


    self.NN = tf.keras.Sequential()

    # Eingabeschicht
    self.NN.add(tf.keras.layers.InputLayer(input_shape=(d,)))

    # 1. verdeckte Schicht
    matW = np.random.binomial(1, 1, d*hidden[0]).reshape((d, hidden[0]))
    vecB = np.random.binomial(1, 1, hidden[0])
    self.NN.add(SparseDense(matW=matW, vecB=vecB, activation=activations.relu))

    # verdeckte Schichten
    for width in hidden[:-1]:
        matW = np.random.binomial(1,1,width*width).reshape((width,width))
        matW[0][0] = 1
        vecB = np.random.binomial(1,1,width)
        self.NN.add(SparseDense(matW=matW,vecB=vecB,activation = activations.relu))

    # letzte verdeckte Schicht
    matW = np.random.binomial(1, 1, hidden[-1]).reshape((hidden[-1], 1))
    self.NN.add(OutputDense(matW=matW))

  def compile(self,optimizer,loss,metric):
      self.NN.compile(loss=loss, optimizer=optimizer, metrics=metric)

  def fit(self,dataX,dataY,epochs,batch):
      self.NN.fit(dataX,dataY,epochs=epochs,batch_size=batch,use_multiprocessing=False)
#### Vor. Dauer: 7 Stunden mit 32 Kernen
from NeuronalNetwork import *
import pandas as pd
import numpy as np
import multiprocessing as mp

#Model
#trainiere Model in Funktion um zu Parallelisierung
# n_train: größe des Samples fürs Training
# s: nicht-null Parameter des neuronalen Netzes
# L: Tiefe
# p_i: Anzahl Neuronen je Schicht
def netz(n_train,L,p_i):
    # erzeuge lineare Regressionsfunktion mit Dimension d und und Summanden mit Hölderkonstante K, beta =1
    d = 2
    K = 15
    f_coeff = list(tuple(map(lambda x: K, list(range(1, d + 1)))))
    f0 = lambda *x: sum(list(map(lambda x1, x2: x1 * x2, *x, f_coeff)))

    # Datensample (trainingsdatensatz) mit std.norm.vert. Fehler
    x = np.random.uniform(0, 1, size=(n_train, d))
    y = noisy_sig(np.fromiter(map(f0, x), dtype=np.int), 1)

    n_test = 5000
    x_orig = np.random.uniform(0, 1, size=(n_test, d))
    y_orig = np.fromiter(map(f0, x_orig), dtype=np.int)

    # modifiziere Parameter neuronales Netz
    p = list(tuple(map(lambda x: p_i, list(range(1, L + 1)))))
    p.append(1)
    p.insert(0, d)

    # netzwerk
    NN1 = NeuronalNetwork_standard(L, p)
    NN1.NN.compile("adam", 'mse', tf.keras.metrics.MeanSquaredError())
    NN1.NN.summary()
    NN1.fit(x, y, epochs=100, batch=100)

    y_pred = NN1.NN.predict(x_orig)
    NN1.NN.evaluate(x_orig, y_orig)

    # Analysiere Netz/zähle nullen
    T = sum([(p[i-1]+1)*p[i] for i in range(1,L+2,1)])-1

    # speichere Daten
    mse = tf.keras.losses.MeanSquaredError()
    return [n_train, n_test, L,  T, p_i,
                        mse(y_orig, y_pred).numpy() / len(y_orig), d, K]



if __name__ == '__main__':
    pool = mp.Pool(3)
    list_n_train = list(range(100,251,10))+list(range(300, 501, 50))+list(range(600,1501,100))
    results = [pool.apply_async(netz, args = (n,20,20)) for n in list_n_train*100]#477 komb.
    results = [result.get() for result in results]
    pool.close()
    df = pd.DataFrame(results, columns=['n_train','n_test','L','anzahl Gewichte gesamt','p_i','mean_squared_error_test','d','K'])
    writer = pd.ExcelWriter('output.xlsx')
    df.to_excel(writer)
    writer.save()
    writer.close()
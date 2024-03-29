# tiefes Lernen

from NeuronalNetwork import *
import pandas as pd
import numpy as np
import multiprocessing as mp
import math

#Model abhängig von n
def netz(n):
    # erzeuge zufällig lineare Regressionsfunktion mit Dimension d und und Summanden mit Hölderkonstante K, beta =1
    d = 2
    K = 15
    f_coeff = list(tuple(map(lambda x: K, list(range(1, d + 1)))))
    f0 = lambda *x: sum(list(map(lambda x1, x2: x1 * x2, *x, f_coeff)))

    # Datensample (trainingsdatensatz) mit std.norm.vert. Fehler
    n_train = n
    x = np.random.uniform(0, 1, size=(n_train, d))
    y = noisy_sig(np.fromiter(map(f0, x), dtype=np.int), 1)

    n_test = 5000
    x_orig = np.random.uniform(0, 1, size=(n_test, d))
    y_orig = np.fromiter(map(f0, x_orig), dtype=np.int)

    # Parameter neuronales Netz
    L = math.ceil((2 * np.log2(3 * 2 * d) * np.log2(n_train)))
    p_i = math.ceil(n_train ** (1 / 3))
    p = list(tuple(map(lambda x: p_i, list(range(1, L + 1)))))
    s = math.ceil(50*n_train ** (1 / 3) * np.log(n_train))
    p.append(1)
    p.insert(0, d)
    # netzwerk
    NN1 = NeuronalNetwork(L, p, s)
    NN1.NN.compile(tf.keras.optimizers.Adam(lr=0.0001), 'mse', tf.keras.metrics.MeanSquaredError())
    NN1.NN.summary()
    NN1.fit(x, y, epochs=20, batch=32)

    y_pred = NN1.NN.predict(x_orig)
    NN1.NN.evaluate(x_orig, y_orig)

    # Analysiere Netz/zähle nullen

    weights = NN1.NN.get_weights()
    max_abs_weight = max(abs(min(weights).sum()), abs(max(weights).sum()))
    F = max((K + 1) * d, (p_i + 1) * L * max_abs_weight)
    T = sum([(p[i-1]+1)*p[i] for i in range(1,L+2,1)])-1

    # speichere Daten
    mse = tf.keras.losses.MeanSquaredError()
    return [n_train, n_test, L, s, T, p_i,
                        mse(y_orig, y_pred).numpy() / len(y_orig), d, K, F,
                        n_train ** (1 / 3) * np.log(n_train),
                        (n ** (-2 / 3)) * (np.log(n)) ** 3, max_abs_weight]



if __name__ == '__main__':
    pool = mp.Pool(3)
    list_iterable = list(range(100,2001,100))*10
    results = [pool.apply_async(netz, args = (n,)) for n in list_iterable]
    results = [result.get() for result in results]
    pool.close()
    df = pd.DataFrame(results, columns=['n_train','n_test','L','s','anzahl Gewichte gesamt','p_i','mean_squared_error_test','d','K','geschätztes F','s erwartet','Schranke nach Theorem','maximales Gewicht'])
    writer = pd.ExcelWriter('output.xlsx')
    df.to_excel(writer)
    writer.save()
    writer.close()



#plt.plot(df['n_train'], df["mean_squared_error_test"], '.')





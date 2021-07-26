from NeuronalNetwork import *
import pandas as pd
import numpy as np
import multiprocessing as mp

# Methode lernt neuronales Netzwerk und gibt den Fehler mit weiteren Ergebnissen zurück. 
# Zu der einer vorgegebenen Größe für die Dimension des Trainingsamples (n_train) wird ein 
# neuronales Netz mit s nicht-null Parametern (im EW), der Tiefe L und p_i Neuronen je Schicht gelernt.
def netz(n_train,s,L,p_i):
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
    NN1 = NeuronalNetwork(L, p, s)
    NN1.NN.compile("adam", 'mse', tf.keras.metrics.MeanSquaredError())
    NN1.NN.summary()
    NN1.fit(x, y, epochs=100, batch=100)

    y_pred = NN1.NN.predict(x_orig)
    NN1.NN.evaluate(x_orig, y_orig)

    # Analysiere Netz/zähle nullen
    weights = NN1.NN.get_weights()
    max_abs_weight = max(abs(min(weights).sum()), abs(max(weights).sum()))
    F = max((K+1)*d,(p_i + 1) * L * max_abs_weight)
    T = sum([(p[i-1]+1)*p[i] for i in range(1,L+2,1)])-1

    # speichere Daten
    mse = tf.keras.losses.MeanSquaredError()
    return [n_train, n_test, L, s, T, p_i,
                        mse(y_orig, y_pred).numpy() / len(y_orig), d, K, F, max_abs_weight]



if __name__ == '__main__':
    pool = mp.Pool(12)
    list_s = [250,500,750,1000]
    list_n_train = list(range(100,251,10))+list(range(300, 501, 50))+list(range(600,1501,100))
    results = [pool.apply_async(netz, args = (n,s,20,10)) for s in list_s for n in list_n_train*25]
    results = [result.get() for result in results]
    pool.close()
    df = pd.DataFrame(results, columns=['n_train','n_test','L','s','anzahl Gewichte gesamt','p_i','mean_squared_error_test','d','K','geschätztes F','maximales Gewicht'])
    writer = pd.ExcelWriter('output.xlsx')
    df.to_excel(writer)
    writer.save()
    writer.close()

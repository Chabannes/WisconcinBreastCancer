import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from helper import *


def main():

    data_visualisation = True
    pca_visualisation = True
    clustering = True
    modeling = True

    # IMPORT DATA
    df = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
    df.columns = ['id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion', 'single_epith_size',
                  'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses', 'class']

    # CLEANING THE DATA
    df.drop(['id'], inplace=True, axis=1)
    df.replace('?', 99999, inplace=True)    # the scaler will handle the created outliers
    # mapping class to binary values. 1 of cancerous, 0 otherwise
    df['class'] = df['class'].map(lambda x: 1 if x == 4 else 0)
    X = np.array(df.drop(['class'], axis=1))
    y = np.array(df['class'])

    # SCALING THE DATA
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # VISUALISING THE DATA
    if data_visualisation:
        data_plot(df, X_train, y_train)

    # VISUALISING THE PCA
    if pca_visualisation:
        plot_pca(df)
        pca, df_cleaned, df_reduced = pca_cleaning()
        biplot(df_cleaned, df_reduced, pca)

    # CLUSTERING
    if clustering:
        sil_coeff2(12, df_reduced)
        gmm = sklearn.mixture.GaussianMixture(n_components=2, covariance_type='full').fit(df_reduced.values)
        plot_resultsGMM(df_reduced.values, gmm.predict(df_reduced.values), gmm.means_, gmm.covariances_, 0, 'Clustering with Gaussian Mixture',\
                        gmm.predict(df_reduced), df)

    # MODELING
    if modeling:
        plot_RandomForest(X, y)
        plot_NeuralNetwork(X, y)

    plt.show()

if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn import mixture
from sklearn.model_selection import learning_curve
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
import tensorflow as tf


def pca_results(good_data, pca):

	dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	components = pd.DataFrame(np.round(pca.components_, 4), columns=list(good_data.keys()))
	components.index = dimensions

	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	fig, ax = plt.subplots(figsize = (14,8))

	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)

	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	return pd.concat([variance_ratios, components], axis = 1)


def data_plot(df, X_train, y_train):

    corr = df.corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, square=True)


    grid = sns.FacetGrid(df, col='class', height = 6)
    grid.map(plt.hist, 'unif_cell_size', alpha=.5, bins=25).set_titles("Cancerous state : {col_name}")
    grid.add_legend();

    grid = sns.FacetGrid(df, col='class', height = 6)
    grid.map(plt.hist, 'mitoses', alpha=.5, bins=25).set_titles("Cancerous state : {col_name}")
    grid.add_legend();

    grid = sns.FacetGrid(df, col='class', height = 6)
    grid.map(plt.hist, 'single_epith_size', alpha=.5, bins=25).set_titles("Cancerous state : {col_name}")
    grid.add_legend()

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    features = ['clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion',
           'single_epith_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli',
           'mitoses']
    plt.figure()
    plt.title('Feature Importance using RandomForest')
    feat_importances = pd.Series(rf.feature_importances_, index=features)
    feat_importances.nlargest(9).plot(kind='barh')


def plot_pca(df):
    scaler = preprocessing.MinMaxScaler()
    df = df.drop('class', axis=1)
    df_PCA = scaler.fit_transform(df)
    x_scaled = scaler.fit_transform(df)
    df_cleaned = pd.DataFrame(x_scaled)
    df_cleaned.columns = ['clump_thickness','unif_cell_size','unif_cell_shape','marg_adhesion','single_epith_size','bare_nuclei','bland_chrom','norm_nucleoli', 'mitoses']
    pca = PCA(n_components = 2, random_state=0)
    pca.fit(df_PCA)

    pca_results(df_cleaned, pca)

def biplot(good_data, reduced_data, pca):

	#inpired by https://github.com/teddyroland/python-biplot

	fig, ax = plt.subplots(figsize=(25, 15))
	ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'],
			   facecolors='b', edgecolors='b', s=70, alpha=0.5)
	feature_vectors = pca.components_.T

	arrow_size, text_pos = 0.8, 1,

	# projection of original features
	for i, v in enumerate(feature_vectors):
		ax.arrow(0, 0, arrow_size * v[0], arrow_size * v[1],
				 head_width=0.02, head_length=0.02, linewidth=2, color='red')
		ax.text(v[0] * text_pos, v[1] * text_pos, good_data.columns[i], color='black',
				ha='center', va='center', fontsize=10)

	ax.set_xlabel("Dimension 1", fontsize=12)
	ax.set_ylabel("Dimension 2", fontsize=12)
	ax.set_title("Projection of the original features on the two first principal components", fontsize=15);
	return ax


def pca_cleaning():

	df_for_PCA = pd.read_csv(
		'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
	df_for_PCA.columns = ['id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion',
						  'single_epith_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mitoses', 'class']

	df_for_PCA.drop(['id', 'bare_nuclei', 'class', 'mitoses'], inplace=True, axis=1)
	df_for_PCA.replace('?', np.nan, inplace=True)
	df_for_PCA.apply(lambda x: x.fillna(x.mean()), axis=0)

	scaler = preprocessing.MinMaxScaler()
	df_for_PCA = scaler.fit_transform(df_for_PCA)
	x_scaled = scaler.fit_transform(df_for_PCA)
	df_cleaned = pd.DataFrame(x_scaled)
	df_cleaned.columns = ['clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion', 'single_epith_size',
						  'bland_chrom', 'norm_nucleoli']

	pca = PCA(n_components=2, random_state=0)
	pca.fit(df_for_PCA)

	df_reduced = pca.transform(df_cleaned)
	df_reduced = pd.DataFrame(df_reduced, columns=['Dimension 1', 'Dimension 2'])
	return pca, df_cleaned, df_reduced


def sil_coeff2(no_clusters, df_reduced):
	print("Determination of the best number of clusters with silhouette coefficient: \n")
	for i in range(2, no_clusters + 1):
		# Apply your clustering algorithm of choice to the reduced data
		clusterer_2 = sklearn.mixture.GaussianMixture(n_components=i, random_state=0)
		clusterer_2.fit(df_reduced)

		# predicting the cluster for each data point
		preds_2 = clusterer_2.predict(df_reduced)

		# finds the cluster centers
		centers_2 = clusterer_2.means_

		score = silhouette_score(df_reduced, preds_2)

		print("silhouette coefficient for `{}` clusters => {:.4f}".format(i, score))


def plot_resultsGMM(X, Y_, means, covariances, index, title, preds, df):
	fig = plt.figure(figsize=(20, 15))
	splot = plt.subplot(2, 1, 1 + index)
	labels = ['benign','cancerous']
	for i, (mean, covar) in enumerate(zip(means, covariances)):
		v, w = np.linalg.eigh(covar)
		v = 2. * np.sqrt(2.) * np.sqrt(v)
		u = w[0] / np.linalg.norm(w[0])

		if not np.any(Y_ == i):
			continue
		plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 2, label=labels[i])

		angle = np.arctan(u[1] / u[0])
		angle = 180. * angle / np.pi  # convert to degrees
		ell = matplotlib.patches.Ellipse(mean, v[0], v[1], 180. + angle)
		ell.set_clip_box(splot.bbox)
		ell.set_alpha(0.5)
		splot.add_artist(ell)

	num_cases = preds.shape[0]
	preds = pd.DataFrame(preds, columns=['Cluster'])
	df['Cluster'] = preds
	num_correct_clustering = (df[(df['class'] == df['Cluster'])]).shape[0]
	score = num_correct_clustering/num_cases*100

	plt.legend()
	plt.xticks(())
	plt.yticks(())
	plt.suptitle(title, fontsize=18)
	plt.title("Percentage of cases in the right cluster : %i%%" %score, fontsize=13)


def plot_RandomForest(X, y):

	# create CV training and test scores for different training set sizes
	train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(),
															X,
															y,
															cv=10,
															scoring='f1',
															n_jobs=-1,
															train_sizes=np.linspace(0.01, 0.3, 50))

	# Create means and standard deviations of training set scores
	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)

	# Create means and standard deviations of test set scores
	test_mean = np.mean(test_scores, axis=1)
	test_std = np.std(test_scores, axis=1)

	# Draw lines
	plt.figure()
	plt.plot(train_sizes, train_mean, '--', color="red", label="Training score")
	plt.plot(train_sizes, test_mean, color="red", label="Cross-validation score")

	# Draw bands
	plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="red", alpha=0.5)
	plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="blue", alpha=0.5)

	# Create plot
	plt.title("Learning Curve with F1 score")
	plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")
	plt.tight_layout()


def plot_NeuralNetwork(X,y):

	model = Sequential()

	model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	history = model.fit(X, y, validation_split=0.33, epochs=400, batch_size=32, verbose=0)

	plt.figure()
	plt.plot(history.history['accuracy'], label='train')
	plt.plot(history.history['val_accuracy'], label='test')
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend()
	plt.show()

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')


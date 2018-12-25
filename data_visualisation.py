import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import itertools


data = pd.read_csv('/Users/rohit/BreastCancer/breast_cancer.csv')
data = df.drop('Unnamed: 32', axis=1)

M = data[(data['diagnosis'] != 0)]
B = data[(data['diagnosis'] == 0)]

# CountPlot
count_plot = sns.countplot(x='diagnosis', data=data);
count_plot_fig = count_plot.get_figure()
count_plot_fig.savefig('diagnosis_count.jpg')

# Correlation Matrix
corr = data.corr()
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
corr_matrix = sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr_matrix_fig = corr_matrix.get_figure()
corr_matrix_fig.savefig('Correlation_matrix.jpg')

# Plotting some HIGHLY Correlated Features
# 1. perimeter_mean_vs_radius_worst
pm_v_rw = sns.scatterplot(x='perimeter_mean',y='radius_worst',data = data, hue='diagnosis')
pm_v_rw_fig = pm_v_rw.get_figure()
pm_v_rw_fig.savefig('perimeter_mean_vs_radius_worst.jpg')

# 2. area_mean_vs_radius_worst
a_v_rw = sns.scatterplot(x='area_mean',y='radius_worst',data = data, hue='diagnosis')
a_v_rw_fig = a_v_rw.get_figure()
a_v_rw_fig.savefig('area_mean_vs_radius_worst.jpg')

# 3. texture_mean_vs_texture_worst
tm_v_tw = sns.scatterplot(x='texture_mean',y='texture_worst',data = data, hue='diagnosis')
tm_v_tw_fig = tm_v_tw.get_figure()
tm_v_tw_fig.savefig('texture_mean_vs_texture_worst.jpg')

# 4. area_mean_vs_area_worst
a_v_aw = sns.scatterplot(x='area_mean',y='area_worst',data = data, hue='diagnosis')
a_v_aw_fig = a_v_aw.get_figure()
a_v_aw_fig.savefig('area_mean_vs_area_worst.jpg')

# 5. All 4 in one figure

palette ={'M' : 'lightblue', 'B' : 'gold'}
edgecolor = 'grey'

fig = plt.figure(figsize=(12,12))

plt.subplot(221)
ax1 = sns.scatterplot(x = data['perimeter_mean'], y = data['radius_worst'], hue = "diagnosis",
                    data = data, palette = palette, edgecolor=edgecolor)
plt.title('perimeter mean vs radius worst')
plt.subplot(222)
ax2 = sns.scatterplot(x = data['area_mean'], y = data['radius_worst'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('area mean vs radius worst')
plt.subplot(223)
ax3 = sns.scatterplot(x = data['texture_mean'], y = data['texture_worst'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('texture mean vs texture worst')
plt.subplot(224)
ax4 = sns.scatterplot(x = data['area_worst'], y = data['area_worst'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('area mean vs radius worst')

fig.suptitle('Positive correlated features', fontsize = 20)
plt.savefig('positive_correlated_features.jpg')
plt.show()


# Uncorrelated Features
fig = plt.figure(figsize=(12,12))

plt.subplot(221)
ax1 = sns.scatterplot(x = data['smoothness_mean'], y = data['texture_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('smoothness mean vs texture mean')
plt.subplot(222)
ax2 = sns.scatterplot(x = data['radius_mean'], y = data['fractal_dimension_worst'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('radius mean vs fractal dimension_worst')
plt.subplot(223)
ax3 = sns.scatterplot(x = data['texture_mean'], y = data['symmetry_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('texture mean vs symmetry mean')
plt.subplot(224)
ax4 = sns.scatterplot(x = data['texture_mean'], y = data['symmetry_se'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('texture mean vs symmetry se')

fig.suptitle('Uncorrelated features', fontsize = 20)
plt.savefig('uncorrelated_features.jpg')
plt.show()

# Negative Correlated Features

fig = plt.figure(figsize=(12,12))

plt.subplot(221)
ax1 = sns.scatterplot(x = data['area_mean'], y = data['fractal_dimension_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('smoothness mean vs fractal dimension mean')
plt.subplot(222)
ax2 = sns.scatterplot(x = data['radius_mean'], y = data['fractal_dimension_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('radius mean vs fractal dimension mean')
plt.subplot(223)
ax2 = sns.scatterplot(x = data['area_mean'], y = data['smoothness_se'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('area mean vs fractal smoothness se')
plt.subplot(224)
ax2 = sns.scatterplot(x = data['smoothness_se'], y = data['perimeter_mean'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('smoothness se vs perimeter mean')

fig.suptitle('Negative correlated features', fontsize = 20)
plt.savefig('negative_correlated_features.jpg')
plt.show()

from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

# Learning curve
def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None,
                        n_jobs = 1, train_sizes = np.linspace(.1, 1.0, 5)):
    """
    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha = 0.1, color = "g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color = "r",
             label = "Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color = "g",
             label = "Cross-validation score")
    plt.legend(loc = "best")
    return plt

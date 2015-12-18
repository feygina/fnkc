import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn import svm, datasets

df = pd.read_csv('2008_100_300.csv', sep=',')
df['Zytogen'].fillna(1, inplace=True)
df['Region'].fillna(77, inplace=True)
df.to_csv('fnkc_without_nan.csv', sep=',', mode='w')

df_new_type = df
df_new_type = df_new_type.drop('ID', axis=1)
df_new_type['Age'] = pd.cut(df['Age'], 8, labels=[1, 2, 3, 4, 5, 6, 7, 8])
df_new_type['Leuc'] = pd.cut(df['Leuc'], 2, labels=[1, 2])
df_new_type['Leber'] = pd.cut(df['Leber'], 2, labels=[1, 2])
df_new_type['weight'] = pd.cut(df['weight'], 3, labels=[1, 2, 3])
df_new_type['height'] = pd.cut(df['height'], 3, labels=[1, 2, 3])

target = df_new_type.ix[:, [-1]]
df_new_type = df_new_type.drop('Better', axis=1)
# print(df_new_type.dtypes)
column_names = df_new_type.columns.values
for col in column_names:
    df_new_type[col] = df_new_type[col].astype('category')
df_binary = pd.get_dummies(df_new_type)
df_binary['Better'] = pd.Series(target.values[:, 0], index=df_binary.index)
df_binary.to_csv('fnkc_binary.csv', sep=',', mode='w')

test_df = df_binary.loc[df['Better'] == 0]
train_df = df_binary.loc[df['Better'] != 0]


# features that will be used for training:
x = train_df.iloc[:, 0:20].values

# target classes
y = train_df.values[:, -1]

# train model
#clf = svm.SVC(kernel='linear', C=1)
#clf = svm.SVC(kernel='poly', degree=3, C=1)
clf = svm.SVC(kernel='rbf', gamma=0.7, C=1)

# cross_val_score
scores = cross_validation.cross_val_score(clf, x, y, cv=10)
# results of prediction
print(np.mean(scores))
# train model again
clf.fit(x, y)
# test features
features = train_df.iloc[:, 0:20].values
# results of prediction
print(clf.predict(features))

# h = .02
# C = 1.0  # SVM regularization parameter
# svc = svm.SVC(kernel='linear', C=C).fit(x, y)
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x, y)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x, y)
# lin_svc = svm.LinearSVC(C=C).fit(x, y)
#
# create a mesh to plot in
# x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
# y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))

# title for the plots
# titles = ['SVC with linear kernel',
#           'LinearSVC (linear kernel)',
#           'SVC with RBF kernel',
#           'SVC with polynomial (degree 3) kernel']
#
# for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, m_max]x[y_min, y_max].
#     plt.subplot(2, 2, i + 1)
#     plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#
#     # Plot also the training points
#     plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
#     plt.xlabel('Sepal length')
#     plt.ylabel('Sepal width')
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.xticks(())
#     plt.yticks(())
#     plt.title(titles[i])
#
# plt.show()
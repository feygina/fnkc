from svm import *
from lazy_fca import *

df = read_csv('2008_100_300.csv')
df = replace_nan_with_mean(df)
df = category_to_binary(df)
test_df, train_df = divide_test_train(df)

#df_for_test = train_df.ix[:, [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, -1]]
#df_for_test = df_for_test.tail(40)
df_for_test = train_df.tail(40)

#try_svm_method(train_df, test_df, 2)

lazy_fca(df_for_test, 2)

# kfkfk = k_fold(train_df, 2)
# for i, j in kfkfk:
#     print(i, j)



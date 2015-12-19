from svm import *


df = read_csv('2008_100_300.csv')
df = replace_nan_with_mean(df)
df = category_to_binary(df)
test_df, train_df = divide_test_train(df)
try_svm_method(train_df, test_df)

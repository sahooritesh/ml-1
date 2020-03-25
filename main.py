import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('data.csv')
print(df.head())
df.tail()
df.info()
# df['diffBreath'].value_counts()
df.describe()
def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

np.random.permutation(7)
train,test=data_split(df,0.2)
print(train)

X_train=train[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
x_test=test[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()

Y_train=train[['infectionProb']].to_numpy().reshape(2060,)
Y_test=test[['infectionProb']].to_numpy().reshape(515,)

clf=LogisticRegression()
clf.fit(X_train,Y_train)

infProb=clf.predict_proba([[98,0,1,1,0]])[0][1]

print(infProb)
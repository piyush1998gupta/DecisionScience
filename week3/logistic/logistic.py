import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



def sig(x):
    return 1 / (1 + np.exp(-x)) ;



df = pd.read_csv("Admission_Predict.csv",index_col=False)
x=df.iloc[:,1:-2]
X = np.c_[np.ones((x.shape[0],1)), x]

y=df.iloc[:,-2:-1]

model = LogisticRegression()
model.fit(X,y.values.ravel())


# x= df.iloc[:, 1:-1]
# y= df.iloc[:,-1:]
# admitted= df[df['Chance of Admit '] > 0.5]
# not_admitted = df[df['Chance of Admit '] < 0.5]
# plt.scatter(admitted.iloc[:,1:2],admitted.iloc[:,2:3])
# plt.scatter(not_admitted.iloc[:,1:2],not_admitted.iloc[:,2:3])
# plt.show()
#
# print(admitted.iloc[:,1:2])
import numpy as np
import pandas as pd 
import matplotlib.pylab as plt 

df = pd.read_csv("drawndata1.csv")
df.head(3)

X = df[['x', 'y']].values
y = df['z'] == "a"

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

#from sklearn.preprocessing import StandardScaler
#X_new = StandardScaler().fit_transform()

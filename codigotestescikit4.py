from ast import Starred
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline 

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X= np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

mod = KNeighborsRegressor().fit(X,y)

#pipeline object
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor())
])
pipe.fit(X,y)
pred = pipe.predict(X)


#pred 
plt.scatter(pred, y)
plt.show()

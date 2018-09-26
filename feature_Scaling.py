def featureScaling(arr):
    s_arr = sorted(arr)
    min = s_arr[0]
    max = s_arr[-1]
    if min == max:
        return arr
    else:
        for i in range(len(s_arr)):
            s_arr[i] = (s_arr[i] - min)/(max-min)
        return s_arr

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print(featureScaling(data))

# We can also use sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy as np 

weights = np.asmatrix(data, float).transpose()
rescaled_weight = MinMaxScaler().fit_transform(weights)
print(rescaled_weight)
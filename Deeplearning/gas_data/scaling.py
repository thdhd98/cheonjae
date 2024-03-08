from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DataScaler:
    def __init__(self, data):
        self.data = data

    def scale_and_sequence(self, window_size=90):
        dfx = self.data.copy()
        for col in dfx.columns:
            scaler = MinMaxScaler()
            dfx[col] = scaler.fit_transform(dfx[[col]])

        dfy = dfx[['공급량(톤)']]
        dfx = dfx.drop('공급량(톤)', axis=1)

        x = dfx.values
        y = dfy.values

        return self.create_dataset(x, y, window_size)

    def create_dataset(self, X, y, window_size):
        data_x, data_y = [], []
        for i in range(len(X) - window_size):
            data_x.append(X[i:(i + window_size)])
            data_y.append(y[i + window_size])
        return np.array(data_x), np.array(data_y)

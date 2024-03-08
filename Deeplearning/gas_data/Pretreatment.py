import pandas as pd

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def encode_and_clean(self):
        # 원-핫 인코딩
        data_encoded = pd.get_dummies(self.data, columns=['구분'])
        # 결측치 처리
        data_encoded.fillna(data_encoded.mean(), inplace=True)
        
        # 이상치 제거
        Q1 = data_encoded['공급량(톤)'].quantile(0.25)
        Q3 = data_encoded['공급량(톤)'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data_cleaned = data_encoded[(data_encoded['공급량(톤)'] >= lower_bound) & (data_encoded['공급량(톤)'] <= upper_bound)]
        
        return data_cleaned

from gas_data.load import DataLoader
from gas_data.Pretreatment import DataPreprocessor
from gas_data.scaling import DataScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
import matplotlib.pyplot as plt

def main():
    loader = DataLoader('./gas/한국가스공사_시간별 공급량_20181231.csv')
    data = loader.load_data()
    
    preprocessor = DataPreprocessor(data)
    cleaned_data = preprocessor.encode_and_clean()
    
    scaler = DataScaler(cleaned_data)
    data_x, data_y = scaler.scale_and_sequence(window_size=90)
    
    train_size = int(len(data_y) * 0.8)
    train_x, train_y = data_x[:train_size], data_y[:train_size]
    test_x, test_y = data_x[train_size:], data_y[train_size:]

    model = Sequential([
        GRU(32, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True),
        Dropout(0.5),
        GRU(32, return_sequences=False),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    history = model.fit(train_x, train_y, epochs=50, batch_size=128, validation_split=0.1, verbose=2)

    test_loss = model.evaluate(test_x, test_y, verbose=2)
    print('Test loss:', test_loss)

    pred_y = model.predict(test_x)

    plt.figure()
    plt.plot(test_y, color='red', label='real target y')
    plt.plot(pred_y, color='blue', label='predict y')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

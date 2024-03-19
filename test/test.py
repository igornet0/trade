import tensorflow as tf
import numpy as np

# Генерация случайного датасета для примера
hour_prices = np.concatenate((np.random.rand(10), np.array([0]*40)))
five_min_prices = np.random.rand(50)
four_hour_prices = np.concatenate((np.random.rand(5), np.array([0]* 45)))

# Подготовка данных
X = np.array([[hour_prices, five_min_prices, four_hour_prices]])

y = np.random.rand(1)

# Создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3, 50)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X, y, epochs=10)

# Сохранение модели
model.save('gas_price_prediction_model.keras')

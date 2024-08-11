import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU
from tensorflow.keras.regularizers import l2



# КОНСТАНТЫ
input_length = 2000
vocab_size = 10000
embedding_dim = 64  # Уменьшили размерность эмбеддинга
model_path = "./models_tmp/best_model.keras"
reviews_path = "Reviews.json"
batch_size = 128
epochs = 100000
learning_rate = 0.0001  # Уменьшили learning rate
patience = 250  # Уменьшили терпение


def print_green(text):
    print("\033[32m{}\033[0m".format(text))


# Функция предобработки текста
def preprocess_text(text, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)
    return padded_sequences[0]

# Загрузка и/или создание модели
if os.path.exists(model_path):
    print_green(f"Model loaded from {model_path}")
    model = tf.keras.models.load_model(model_path)
else:
    print_green("Model created")
    # СОЗДАНИЕ МОДЕЛИ
    model = tf.keras.models.Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        GRU(64, return_sequences=False, kernel_regularizer=l2(0.01)),  # Используем GRU вместо LSTM
        Dropout(0.7),  # Увеличили Dropout
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # L2-регуляризация
        Dropout(0.7),  # Ещё один Dropout слой
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

# Загрузка данных
file = pd.read_json(reviews_path)

good_reviews = file['good_reviews']
neutral_reviews = file['neutral_reviews']
bad_reviews = file['bad_reviews']

x_train = []
y_train = []

# Создание токенизатора
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(good_reviews + neutral_reviews + bad_reviews)

for k in range(len(good_reviews)):
    x_train.append(preprocess_text(good_reviews[k], tokenizer, input_length))
    y_train.append([0, 0, 1])  # Хорошие отзывы
    x_train.append(preprocess_text(neutral_reviews[k], tokenizer, input_length))
    y_train.append([0, 1, 0])  # Нейтральные отзывы
    x_train.append(preprocess_text(bad_reviews[k], tokenizer, input_length))
    y_train.append([1, 0, 0])  # Плохие отзывы

x_train = np.array(x_train)
y_train = np.array(y_train)

# CALLBACKS
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

checkpoint_callback = ModelCheckpoint(
    filepath=model_path,
    monitor='val_loss',
    save_best_only=True
)

# ОБУЧЕНИЕ
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    callbacks=[early_stopping, checkpoint_callback]
)

# СОХРАНЕНИЕ МОДЕЛИ
print(f"Лучшая модель сохранена в {model_path}")

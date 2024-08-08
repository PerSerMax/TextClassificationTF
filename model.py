import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Embedding, LSTM, Dense

# КОНСТАНТЫ
input_length = 2000
vocab_size = 10000  # Размер словаря для Embedding слоя
embedding_dim = 128  # Размерность векторов эмбеддинга
model_path = "model.keras"
batch_size = 128
epochs = 100000
learning_rate = 0.001
patience = 100

def preprocess_text(text, rev_len):
    text_bytes = text.encode('utf-8')
    tmp = np.zeros(rev_len)
    for i in range(min(len(text_bytes), rev_len)):
        tmp[i] = text_bytes[i]
    return tmp

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    # СОЗДАНИЕ МОДЕЛИ
    model = tf.keras.models.Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # Выходной слой с 3 нейронами
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

# СОЗДАНИЕ НАБОРА ДАННЫХ
file = pd.read_json("Top500FilmsReview.json")

good_reviews = []
neutral_reviews = []
bad_reviews = []

for reviews in file['film_reviews']:
    for review in reviews:
        if review is None:
            continue
        opinion = review["review_opinion"]
        title = review['title']
        text = review['text'].replace('\n', '').replace('\r', '')
        formatted_review = title + text
        if opinion == 'good':
            good_reviews.append(formatted_review)
        elif opinion == 'neutral':
            neutral_reviews.append(formatted_review)
        else:
            bad_reviews.append(formatted_review)

align_num = min(len(bad_reviews), len(good_reviews), len(neutral_reviews))

x_train = []
y_train = []

for k in range(align_num):
    x_train.append(preprocess_text(good_reviews[k], input_length))
    y_train.append([0, 0, 1])  # Хорошие отзывы
    x_train.append(preprocess_text(neutral_reviews[k], input_length))
    y_train.append([0, 1, 0])  # Нейтральные отзывы
    x_train.append(preprocess_text(bad_reviews[k], input_length))
    y_train.append([1, 0, 0])  # Плохие отзывы

x_train = np.array(x_train)
y_train = np.array(y_train)

# Проверка форм данных
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# CALLBACKS
early_stopping_val_loss = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

checkpoint_callback = ModelCheckpoint(
    filepath='./models_tmp/model_epoch_{epoch:02d}.keras',
    save_freq=(x_train.shape[0] // batch_size),  # Сохраняет модель каждые 30 эпох
    save_best_only=False
)

# ОБУЧЕНИЕ
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    callbacks=[early_stopping_val_loss, checkpoint_callback]
)

# СОХРАНЕНИЕ МОДЕЛИ
model.save(model_path)
print(f"Модель сохранена в {model_path}")

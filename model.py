import os
import pandas as pd
import numpy as np
import tensorflow as tf

# КОНСТАНТЫ
rev_len = 300
model_path = "model.keras"

def preprocess_text(text, rev_len):
    text_bytes = text.encode('utf-8')
    tmp = np.zeros(rev_len)
    for i in range(min(len(text_bytes), rev_len)):
        tmp[i] = text_bytes[i]
    return tmp

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Модель загружена из файла.")
else:
    # СОЗДАНИЕ МОДЕЛИ
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(rev_len, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(rev_len // 2, activation="relu"),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(rev_len // 4, activation="relu"),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(rev_len // 8, activation="relu"),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(rev_len // 16, activation="relu"),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
        loss=tf.keras.losses.BinaryCrossentropy(),
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
        x_train.append(preprocess_text(good_reviews[k], rev_len))
        y_train.append(1)
        x_train.append(preprocess_text(neutral_reviews[k], rev_len))
        y_train.append(0.5)
        x_train.append(preprocess_text(bad_reviews[k], rev_len))
        y_train.append(0)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Проверка форм данных
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # CALLBACKS
    early_stopping_val_loss = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3000,
        restore_best_weights=True
    )

    early_stopping_loss = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3000,
        restore_best_weights=True
    )

    # ОБУЧЕНИЕ
    history = model.fit(
        x_train,
        y_train,
        batch_size=8,
        epochs=100000,
        validation_split=0.2,
        callbacks=[early_stopping_val_loss, early_stopping_loss]
    )

    # СОХРАНЕНИЕ МОДЕЛИ
    model.save(model_path)
    print(f"Модель сохранена в {model_path}")

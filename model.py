import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.regularizers import l2
import random
import pymorphy2

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()

# Функция аугментации текста (без синонимов)
def augment_text(text, aug_prob=0.1):
    words = text.split()
    new_words = words.copy()

    # Случайное удаление слова
    if random.uniform(0, 1) < aug_prob:
        if len(new_words) > 1:
            del new_words[random.randint(0, len(new_words) - 1)]

    # Перестановка слов
    if random.uniform(0, 1) < aug_prob:
        random.shuffle(new_words)

    return ' '.join(new_words)

# КОНСТАНТЫ
input_length = 5000
vocab_size = 10000
embedding_dim = 128
model_path = "./models_tmp/best_model.keras"
reviews_path = "Reviews.json"
batch_size = 128
epochs = 100000
learning_rate = 0.0001
patience = 250

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
        GRU(32, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.7),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.7),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', 'mean_absolute_error']
    )

# Загрузка данных
file = pd.read_json(reviews_path)

good_reviews = file['good_reviews']
neutral_reviews = file['neutral_reviews']
bad_reviews = file['bad_reviews']

x_train = []
y_train = []
augmentation_factor = 2  # Количество аугментированных версий для каждого отзыва

# Создание токенизатора
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(good_reviews + neutral_reviews + bad_reviews)

for k in range(len(good_reviews)):
    # Преобразование оригинального текста в последовательности
    x_train.append(preprocess_text(good_reviews[k], tokenizer, input_length))
    y_train.append([0, 0, 1])  # Хорошие отзывы
    x_train.append(preprocess_text(neutral_reviews[k], tokenizer, input_length))
    y_train.append([0, 1, 0])  # Нейтральные отзывы
    x_train.append(preprocess_text(bad_reviews[k], tokenizer, input_length))
    y_train.append([1, 0, 0])  # Плохие отзывы
    
    # Создание аугментированных текстов и добавление их в тренировочный набор
    for _ in range(augmentation_factor):
        augmented_good = augment_text(good_reviews[k])
        augmented_neutral = augment_text(neutral_reviews[k])
        augmented_bad = augment_text(bad_reviews[k])

        x_train.append(preprocess_text(augmented_good, tokenizer, input_length))
        y_train.append([0, 0, 1])  # Хорошие отзывы
        x_train.append(preprocess_text(augmented_neutral, tokenizer, input_length))
        y_train.append([0, 1, 0])  # Нейтральные отзывы
        x_train.append(preprocess_text(augmented_bad, tokenizer, input_length))
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
    monitor='val_accuracy',
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

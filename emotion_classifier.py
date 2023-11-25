import string

import keras.layers
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix

stp_words = stopwords.words('english')
tokenizer = Tokenizer()

text_data =pd.read_excel(r"C:\Users\param\OneDrive\Desktop\Book1.xlsx")

#text_data = data.drop(axis=1, columns=['tweet_id'])

punctuations_list = string.punctuation


def remove_stopwords(text):
    imp_words = []

    # Storing the important words
    for word in str(text).split():
        word = word.lower()

        if word not in stp_words:
            imp_words.append(word)

    output = " ".join(imp_words)

    return output


# text_data['content'] = text_data['content'].apply(lambda text: remove_stopwords(text))
# Downsampling to balance the dataset


X = text_data['content']

Y = text_data['sentiment']

encoder = LabelEncoder()

Y = encoder.fit_transform(Y)

hot_labels = keras.utils.to_categorical(Y)

trainx, testx, trainy, testy = train_test_split(X, hot_labels, test_size=0.25)
# Tokenize the text data
tokenizer.fit_on_texts(trainx )

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(trainx)
test_sequences = tokenizer.texts_to_sequences(testx)

# Pad sequences to have the same length
max_len = 30  # maximum sequence length
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')

test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

# print(train_sequences)


# Build the model
model = Sequential()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len))
# model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(6, activation='softmax'))

# Print the model summary
print(model.summary())
model.save(r'model_emotion_classifier.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_sequences, y=trainy, epochs=15, verbose=1, batch_size=128,
                    callbacks=tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2, verbose=1, mode='max',
                                                               restore_best_weights=True))

hist = pd.DataFrame(model.history.history)
hist.plot()
plt.show()
predy = model.predict(test_sequences)


# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences, testy)
print('Test Loss :', test_loss)
print('Test Accuracy :', test_accuracy)


labels = set(encoder.inverse_transform(Y))
label_nums=set(Y)
print(labels,label_nums)


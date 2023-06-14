import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
from datetime import datetime


nltk.download('punkt')
nltk.download('wordnet')

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(text)]

training_data = [
    {'pattern': 'Hi', 'intent': 'greeting'},
    {'pattern': 'How are you?', 'intent': 'greeting'},
    {'pattern': 'What are you doing?', 'intent': 'question'},
    {'pattern': 'Who are you?', 'intent': 'question'},
    {'pattern': 'What are you?', 'intent': 'questions'},
    {'pattern': 'What is the current time?', 'intent': 'current_time'},
    {'pattern': 'What is the weather today?', 'intent': 'weather'},
    {'pattern': 'Goodbye', 'intent': 'goodbye'}
]

words = []
for data in training_data:
    words.extend(preprocess(data['pattern']))
words = sorted(list(set(words)))

intents = sorted(list(set(data['intent'] for data in training_data)))

X_train = []
y_train = []

for data in training_data:
    pattern_words = preprocess(data['pattern'])
    X_train.append([1 if word in pattern_words else 0 for word in words])
    y_train.append(intents.index(data['intent']))

X_train = np.array(X_train)
y_train = np.array(y_train)

model = Sequential()
model.add(Dense(128, input_shape=(len(words),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(intents), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

def predict_intent(text):
    input_data = np.array([preprocess(text)])
    X = np.zeros((len(input_data), len(words)))
    for i, sentence in enumerate(input_data):
        for word in sentence:
            if word in words:
                X[i, words.index(word)] = 1

    predictions = model.predict(X)
    predicted_intent = intents[np.argmax(predictions)]
    return predicted_intent

def generate_response(intent):
    if intent == 'greeting':
        return 'Hello!'
    elif intent == 'question':
        return 'I am a chatbot.My name is Demon'
    elif intent == 'questions':
        return 'I am a chatbot, with name Demon. I was created with the help of deep learning with codes and libraries.'
    elif intent == 'weather':
        return 'Sorry, I cannot provide weather information at the moment.'
    elif intent == 'current_time':
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        return f"The current time is {current_time}."
    elif intent == 'goodbye':
        return 'Goodbye!'
    else:
        return 'Sorry, I do not understand.'

while True:
    user_input = input('User: ')
    intent = predict_intent(user_input)
    response = generate_response(intent)
    print('Chatbot:', response)


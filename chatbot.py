import random
import pickle
import json
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
##tf.keras.models

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', encoding='utf-8').read())

# carregar arquivos
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

# limpar a frase
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# converter a frase em um 'saco' de palavras= lista de 0 e 1 que indicam se a palavra está lá ou não
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# função de previsão
def predict_class(sentece):
    bow = bag_of_words(sentece)
    res = model.predict(np.array([bow]))[0] # prevê os resultados com base no 'saco' de palavras
    ERROR_THRESHOLD = 0.25  # considera 25% de incertezas contanto que não esteja muito fora do padrão
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] # enumera para ter o index e a classe
    # removendo incertezas

    # ordena os resultados em ordem drecrescente
    results.sort(key=lambda x: x[1], reverse=True) # reverse = True ordem decrescente
    # cria a lista com classes e probabilidades
    return_list = []
    for r in results:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intents']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('START, o Bot está rodando!')

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
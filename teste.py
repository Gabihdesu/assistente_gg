#MODELO DE REDE NEURAL

import random
import json
import pickle
import numpy as np

##interpreta uma palavra mesmo escrita de diferentes formas
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()
# interpreta uma palavra mesmo escrita de diferentes formas

intents = json.loads(open('intents.json', encoding='utf-8').read())

words = []
classes = []
documentos = []
ignore_letters = ['?', '!', '.', ',']

# para cada parão dentro de intent
for intent in intents['intents']:
    for patterns in intent['patterns']:
        word_list = nltk.word_tokenize(patterns)  # tokanize separa cada frase em palavras
        words.extend(word_list)  # extende a lista
        documentos.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# interprete a palavra mesmo escrita de diferentes formaspara cada palavra na lista 'palavras' se não estiver na
# lista de ignorados
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))   # set elimina duplicatas e coloca em ordem

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))  # salvar em arquivos
pickle.dump(classes, open('classes.pkl', 'wb'))  # salvar em arquivos

#MACHINE LEARNING

#representar palavras como valores númericos como ocorrencias individuais 0 e 1
#tanto para sa palavras quanto para as classes
training = []
output_empty = [0] * len(classes)

for document in documentos:
    #entrada de dados
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        #se a palavra aparece na lista adiciona 1 na lista bag se não adiciona 0

    #lista saida de dados
    output_row = list(output_empty)
    #definir o index da saida de dados como 1
    output_row[classes.index(document[1])] = 1
    #adicionar tod os dados do documento na lista de treino
    training.append([bag, output_row])

random.shuffle(training)  # embaralhar a lista de treino
training = np.array(training)

train_x = list(training[:, 0])  # no eixo x tudo da dimensão 0
train_y = list(training[:, 1])  # no eixo y tudo da dimensão 1


#Modelo sequencial simples
model = Sequential()

#camada de input. Camada densa com 128 neuronios e um formato de input que depende do tamanho dos dados de treino para X
#activation especifica a função de ativação para retificar
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
#prevenir overfitting
model.add(Dense(64, activation='relu'))
#outra camada com 64 neuronios
model.add(Dropout(0.5))

#camada densa para as labels.
# softmax é a função que resume ou escala os resultados na camada de saida(output) assim todos somam um.
#para termos uma porcentagem de quão provável é ter aquele resultado(output) na label
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# lr= taxa de aprendizagem, declínio, impulso/velocidade
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#compilar modelo, otimizador, precisão

#A matriz np vai ser os dados de treino do eixo x e y. epochs = alimentar o mesmo dado 200 vezes na rede neural
# em um tamanho de lote (batch size)= 5. verbose = 1 para ter uma quantidade média de informação
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')


import json

import torch
import torch.nn as nn
import torch.optim as optim


import nltk
#nltk.download('wordnet')
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

import stanza
import warnings;   warnings.filterwarnings("ignore")

lemmatizer = WordNetLemmatizer()

# Inicializar el stemmer
#stemmer = PorterStemmer()
stemmer = SnowballStemmer("spanish")

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def lemma(word):
    return lemmatizer.lemmatize(word.lower())

def get_spanish_lemmas(word_list):
    stanza.download('es', package='ancora', processors='tokenize,mwt,pos,lemma', verbose=False)
    stNLP = stanza.Pipeline(processors='tokenize,mwt,pos,lemma', lang='es', use_gpu=True)
    docs = [stNLP(word) for word in word_list]
    # Extract lemmas
    lemmas = [word.lemma for doc in docs for sent in doc.sentences for word in sent.words]

    return lemmas

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

#Datos de entrenamiento
#Preprocessing
#organizacion de oraciones, palabras

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words_orig = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words_orig.extend(w)
        xy.append((w, tag))


#Excluir simbolos
ignore_words = ['?', '!', '¿', '.', ',']

# stemming y minusculas de cada palabra
all_words = [stem(w) for w in all_words_orig if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


# # lematizacion y minusculas de cada palabra
# all_words1 = [lemma(w) for w in all_words_orig if w not in ignore_words]
# all_words1 = sorted(set(all_words1))

# result = get_spanish_lemmas(all_words1)
# all_words1 = sorted(set(result))
# result_p = list(filter(lambda item: item not in ignore_words, all_words1))


#Transformacion de palabras a numeros
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


if __name__ == '__main__':
    print("Original:")
    print(all_words_orig)

    print("Stemming NLTK:", '   ', len(all_words), 'palabras')
    print(sorted(all_words))

    # print("Lematizacion NLTK:", '   ', len(all_words1), 'palabras')
    # print(sorted(all_words1))

    # print('Lematizacion Stanza:', '   ', len(result_p), 'palabras')
    # print(sorted(result_p))

    print('Oraciones para entrenamiento:', y_train.shape[0])
    print('Tags:', tags)
    print('Label', y_train)

    print(X_train[10])


# Definir el modelo
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

model = NeuralNet(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = model(torch.from_numpy(X_train).float())
    loss = criterion(outputs, torch.from_numpy(y_train).long())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#    if (epoch + 1) % 100 == 0:
#        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Guardar el modelo
torch.save({
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'all_words': all_words,
    'tags': tags,
    'model_state': model.state_dict()
}, 'modelo.pth')
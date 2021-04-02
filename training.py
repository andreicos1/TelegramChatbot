import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = list('?!.,;')

# Access the subdictionaries in the json intents file
for intent in intents['intents']:
    # For each of the pattern, apply nltk processing
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #split into individual words
        words.extend(word_list) #append to total list
        # Also append the word list and it's name to the document
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize all words that are not among the letters that should be ignored
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) #drop duplicates

# Save the words and classes into a pickle file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Using bag of words
training = []
output_empty = [0] * len(classes) #init list with 0 for each class

for document in documents:
    # For each combination create an empty bag of words
    bag = []
    word_patterns = document[0] #get the word patterns (idx 1 is the class name)
    # Lowercase words and lemmatize them
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        # For all words in the documents, use 1 if it's in this particular document, else 0
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = output_empty.copy() #create a copy of the empty output
    output_row[classes.index(document[1])] = 1 # set the row corresponding to this class as 1
    training.append([bag, output_row]) #append the bag (X) and output (y) to the training list

# Shuffle the training list and turn into a numpy array
random.shuffle(training)
training = np.array(training)

# Separate x and y
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create the Neural Network
def create_model():
    ''' Creates the Neural Network Sequential model and returns is '''
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5)) #to prevent overfitting drop 50% of data randomly
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax')) #in softmax results add up to 1
    return model

model = create_model()

# Define the optimizer
sgd = SGD(lr=.01, decay=1e-6, momentum=.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics='accuracy')

# Fit the model and save it
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5", hist)
print("Training Complete! File saved as 'chatbot_model.h5' ")

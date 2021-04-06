import random
import json
import pickle
import numpy as np


from sklearn.naive_bayes import MultinomialNB
from joblib import dump
import nltk
from nltk.stem import WordNetLemmatizer

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

for document in documents:
    # For each combination create an empty bag of words
    bag = []
    word_patterns = document[0] #get the word patterns (idx 1 is the class name)
    # Lowercase words and lemmatize them
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        # For all words in the documents, use 1 if it's in this particular document, else 0
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = classes.index(document[1]) #get the index of the class
    training.append([bag, output_row]) #append the bag (X) and output (y) to the training list

# Shuffle the training list and turn into a numpy array
random.shuffle(training)
training = np.array(training)

# Separate x and y
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create a Naive Bayes model
model = MultinomialNB()
# Fit the model and save it
model.fit(train_x, train_y)
dump(model, "chatbot_model.joblib")

print("Training Complete! File saved as 'chatbot_model.joblib' ")


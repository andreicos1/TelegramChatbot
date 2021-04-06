import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from joblib import load

lemmatizer = WordNetLemmatizer() #init lemmatizer
intents = json.loads(open('intents.json').read()) #open the intents file
# Open the words and classes files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load("chatbot_model.joblib") # load the trained model

def clean_up_sentence(sentence):
    '''Clean up the user input before it can be identified by the model'''
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    ''' Get the bag of words for the inputted text '''
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    # Loop through words in this sentence and all words in the intents file
    for word_this_sentence in sentence_words:
        for i, word in enumerate(words):
            if word_this_sentence==word:
                # Set the idx of this word to 1 if it's in this sentence
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    ''' Predict the class of the user input '''
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0] #predict the class
    return res

def get_response(intent, intents_json):
    ''' Using the predicted class, output a random response
     Takes the prediction output and the json dictionary as input
     '''
    possible_replies = intents_json['intents'][intent]['responses']
    return random.choice(possible_replies)

def test_bot():
    ''' This function is used to test the bot '''
    while True:
        # get message input
        message = input("")
        if message.lower()=='exit': #simple way to exit the program
            return "Test ended."
        intent = predict_class(message) #get the intent
        res = get_response(intent, intents) #get the response
        print(res) #output it to the console

if __name__ == '__main__':
    # Test the bot when the program is run directly from this file
    test_bot()
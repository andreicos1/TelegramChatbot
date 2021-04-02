import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer() #init lemmatizer
intents = json.loads(open('intents.json').read()) #open the intents file
# Open the words and classes files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model("chatbot_model.model") # load the trained model

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
    # Get all classes with p >25%
    ERROR_THRESHOLD = .25
    results =[[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort the results by highest probability first
    results.sort(key=lambda a: a[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    ''' Using the predicted, class, output a response
     Takes the prediction output (A WHAT?) and the json dictionary as input
     '''
    tag = intents_list[0]['intent'] # get the top predicted intent
    list_of_intents = intents_json['intents'] #get the intents dictionary
    for i in list_of_intents:
        # Find the predicted tag
        if i['tag'] == tag:
            # Pick a random response
            result = random.choice(i['responses'])
            return result

def test_bot():
    ''' This function is used to test the bot '''
    while True:
        # get message input
        message = input("")
        if message.lower()=='exit': #simple way to exit the program
            return "Test ended."
        ints = predict_class(message) #get the intent
        res = get_response(ints, intents) #get the response
        print(res) #output it to the console

if __name__ == '__main__':
    # Test the bot when the program is run directly from this file
    test_bot()
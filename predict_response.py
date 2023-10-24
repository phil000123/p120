import nltk
nltk.download("punkt")
import json
import pickle
import numpy as np
import random
ignore_words = ['?', '!',',','.', "'s", "'m"]

import tensorflow 
from data_preprocessing import get_stem_words
model = tensorflow.keras.models.load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def preProcessUserInput(user_input):
    inputWordToken1 = nltk.word_tokenize(user_input)
    inputWordToken2 = get_stem_words(inputWordToken1, ignore_words)
    inputWordToken2 = sorted(list(set(inputWordToken2)))
    bag = []
    bagOfWord = []
    for word in words:
        if word in inputWordToken2:
            bagOfWord.append(1)
        else:
            bagOfWord.append(0)
    bag.append(bagOfWord)
    return np.array(bag)
def botClassPrediction(user_input):
    inp = preProcessUserInput(user_input)
    prediction = model.predict(inp)
    predicted_classLabel = np.argmax(prediction[0])
    return predicted_classLabel
def bot_response(user_input):
    predictedClassLabel = botClassPrediction(user_input)
    predicted_class = classes[predictedClassLabel]
    for intent in intents["intents"]:
        if intent["tag"] == predicted_class:
            bot_response = random.choice(intent["responses"])
            return bot_response
print("hi im stellar how can i help you")
while True:
    user_input = input("Type your message here")
    print(user_input)
    response = bot_response(user_input)
    print(response)
    
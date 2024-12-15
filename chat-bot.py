# importing the required modules. 
import random 
import json 
import pickle 
import numpy as np 
import nltk 
nltk.download('wordnet')
nltk.download('punkt_tab')

from keras.models import Sequential 
from nltk.stem import WordNetLemmatizer 
from keras.layers import Dense, Activation, Dropout 
from keras.optimizers import SGD 


lemmatizer = WordNetLemmatizer() 

# reading the json.intense file 
intents =  {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "Good morning"], "responses": ["Hello!", "Hi there!", "Greetings!"]},
        {"tag": "goodbye", "patterns": ["Bye", "Goodbye", "See you", "Take care"], "responses": ["Goodbye!", "See you later!", "Take care!"]},
        {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot"], "responses": ["You're welcome!", "Glad to help!", "Anytime!"]},
        {"tag": "bye", "patterns": ["bye", "see you", "later"], "responses": ["Goodbye! See you later!"]},
    ]
}

# creating empty lists to store data 
words = [] 
classes = [] 
documents = [] 
ignore_letters = ["?", "!", ".", ","] 
for intent in intents['intents']: 
	for pattern in intent['patterns']: 
		# separating words from patterns 
		word_list = nltk.word_tokenize(pattern) 
		words.extend(word_list) # and adding them to words list 
		
		# associating patterns with respective tags 
		documents.append(((word_list), intent['tag'])) 

		# appending the tags to the class list 
		if intent['tag'] not in classes: 
			classes.append(intent['tag']) 

# storing the root words or lemma 
words = [lemmatizer.lemmatize(word) 
		for word in words if word not in ignore_letters] 
words = sorted(set(words)) 

# saving the words and classes list to binary files 
pickle.dump(words, open('words.pkl', 'wb')) 
pickle.dump(classes, open('classes.pkl', 'wb')) 
# we need numerical values of the 
# words because a neural network 
# needs numerical values to work with 
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    pattern, tag = document  # Unpack the document tuple
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in pattern]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # making a copy of the output_empty
    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1  # Use the tag from the document tuple
    training.append([bag, output_row])

# splitting the data 
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

# creating a Sequential machine learning model 
model = Sequential() 
model.add(Dense(128, input_shape=(len(train_x[0]), ), 
				activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(len(train_y[0]), 
				activation='softmax')) 

# compiling the model 
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)   
model.compile(loss='categorical_crossentropy', 
			optimizer=sgd, metrics=['accuracy']) 
hist = model.fit(np.array(train_x), np.array(train_y), 
				epochs=200, batch_size=5, verbose=1) 

# saving the model 
model.save("chatbotmodel.keras", hist) 

# print statement to show the 
# successful training of the Chatbot model 
print("Yay!") 


import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# loading the files we made previously
intents = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "Good morning"], "responses": ["Hello!", "Hi there!", "Greetings!"]},
        {"tag": "goodbye", "patterns": ["Bye", "Goodbye", "See you", "Take care"], "responses": ["Goodbye!", "See you later!", "Take care!"]},
        {"tag": "thanks", "patterns": ["Thank you", "Thanks", "Thanks a lot"], "responses": ["You're welcome!", "Glad to help!", "Anytime!"]},
        {"tag": "bye", "patterns": ["bye", "see you", "later"], "responses": ["Goodbye! See you later!"]},
    ]
}
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.keras')
def clean_up_sentences(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word)
					for word in sentence_words]
	return sentence_words
def bagw(sentence):

	# separate out words from the input sentence
	sentence_words = clean_up_sentences(sentence)
	bag = [0]*len(words)
	for w in sentence_words:
		for i, word in enumerate(words):

			# check whether the word
			# is present in the input as well
			if word == w:

				# as the list of words
				# created earlier.
				bag[i] = 1

	# return a numpy array
	return np.array(bag)
def predict_class(sentence):
	bow = bagw(sentence)
	res = model.predict(np.array([bow]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i, r] for i, r in enumerate(res)
			if r > ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent': classes[r[0]],
							'probability': str(r[1])})
		return return_list
def get_response(intents_list, intents_json):
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']
	result = ""
	for i in list_of_intents:
		if i['tag'] == tag:

			# prints a random response
			result = random.choice(i['responses'])
			break
	return result

print("Chatbot is up!")
while True:
	message = input("")
	ints = predict_class(message)
	res = get_response(ints, intents)
	print(res)

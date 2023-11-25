import tensorflow as tf
import keras
import keras.layers
import pandas as pd
from wordcloud import wordcloud
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

stp_words=stopwords.words('english')
tokenizer=Tokenizer()

data = pd.read_csv(r"C:\Users\param\OneDrive\Desktop\spam_ham_dataset.csv")

text_data= data.drop(axis=1,columns=['label', 'Unnamed: 0'])


punctuations_list = string.punctuation


text_data['text']=text_data['text'].str.replace('Subject ', '')

def remove_stopwords(text):
	imp_words = []

	# Storing the important words
	for word in str(text).split():
		word = word.lower()

		if word not in stp_words:
			imp_words.append(word)

	output = " ".join(imp_words)

	return output


text_data['text'] = text_data['text'].apply(lambda text: remove_stopwords(text))
# Downsampling to balance the dataset
ham_msg = text_data[data.label_num == 0]
spam_msg = text_data[data.label_num == 1]
ham_msg = ham_msg.sample(n=len(spam_msg),
						random_state=42)

# Plotting the counts of down sampled dataset
text_data = ham_msg._append(spam_msg)
X=text_data['text'].str.replace('Subject :', '')

Y=text_data['label_num']


trainx,testx,trainy,testy=train_test_split(X,Y,test_size=0.25,random_state=12)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainx,)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(trainx)
test_sequences = tokenizer.texts_to_sequences(testx)

# Pad sequences to have the same length
max_len = 100 # maximum sequence length
train_sequences = pad_sequences(train_sequences,maxlen=max_len,padding='post',truncating='post')


test_sequences = pad_sequences(test_sequences,maxlen=max_len,padding='post',truncating='post')

#print(train_sequences)


# Build the model
model=Sequential()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,output_dim=32,input_length=max_len))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Print the model summary
print(model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(x=train_sequences,validation_data=(test_sequences, testy),y=trainy,epochs=25,verbose=1,
				  callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=6,verbose=1,mode='max',restore_best_weights=True))

hist=pd.DataFrame(model.history.history)
hist.plot()
plt.show()
predy=model.predict(test_sequences)


for i in range(len(predy)):
	predy[i]=int(round(predy[i][0]))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences, testy)
print('Test Loss :',test_loss)
print('Test Accuracy :',test_accuracy)

print('/n',confusion_matrix(testy,predy))
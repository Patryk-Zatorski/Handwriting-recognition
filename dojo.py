from tensorflow import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten

(train_images, train_label), (test_images, test_labels) = mnist.load_data()

train_images=train_images/255
test_images=test_images/255

model=Sequential()
model.add(Flatten(input_shape=(28,28)))           
model.add(Dense(256, activation='sigmoid'))       
model.add(Dense(128, activation='sigmoid'))       
model.add(Dense(64, activation='sigmoid'))        
model.add(Dense(10, activation='softmax'))   

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

history=model.fit(train_images, train_label, validation_data=(test_images, test_labels), epochs=10)

plt.plot(history.history['acc'], label='accuracy')            
plt.plot(history.history['val_acc'], label='val_accuracy')
plt.title('acc')
plt.legend()

plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

"""
To load 

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
"""
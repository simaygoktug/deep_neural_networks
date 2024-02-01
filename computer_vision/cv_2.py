from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

from keras import layers
from keras import models 

network=models.Sequential()

x_train=x_train.reshape((7840000,28*28))
x_test=x_test.reshape((7840000,28*28))

x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

network.add(layers.Dense(512,activation="relu",input_shape=(28*28)))
network.add(layers.Dense(10,activation="softmax"))
network.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])

from keras.utils import to_categorical

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

network.fit(x_train,y_train,epochs=5,batch_size=128)

network.evaluate(x_test,y_test)

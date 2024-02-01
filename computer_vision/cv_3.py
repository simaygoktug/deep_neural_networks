import tensorflow as tf
import numpy as np

mnist=tf.keras.datasets.mnist

(X_train,y_train),(X_test,y_test)=mnist.load_data()

#Fotoğraflar aslında 255 tane sayıdan oluştuğu için modelin daha iyi performans vermesi için veri setini standartlaştırıyoruz.

X_train=X_train/255.0
X_test=X_test/255.0

model=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                  tf.keras.layers.Dense(128,activation="relu"),
                                  tf.keras.layers.Dropout(0.2),
                                  tf.keras.layers.Dense(10)
                                  ])

predictions=model(X_train[:1]).np()
tf.nn.softmax(predictions).np()

y_true=[1,2]
y_pred=[[0.05,0.95,0],[0.1,0.8,0.1]]
scce=tf.keras.losses.SparseCategoricalCrossentropy()
scce(y_true,y_pred).np()

loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1],predictions).np()

model.compile(optimizer="adam",
              loss=loss_fn,
              metrics=["accuracy"])

model.fit(X_train,y_train,epochs=5)

model.evaluate(X_test,y_test,veerbos=2)

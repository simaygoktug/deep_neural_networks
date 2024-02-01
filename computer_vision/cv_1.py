#CNN (Convolutional Neural Network):

#CNN'ler özellikle görsel verilerin işlenmesi için tasarlanmış sinir ağı türüdür. 
#Temel amacı, görüntü verilerindeki özellikleri çıkarmak ve bu özellikleri sınıflandırma, nesne tespiti veya segmentasyon gibi görevlerde kullanmaktır. 
#CNN'ler, özellikle evrişim ve havuzlama katmanlarından oluşur.

#Evrişim (Convolution): Görüntülerdeki özellikleri tespit etmek için kullanılan bir işlem. 
#Çekirdek (kernel) adı verilen küçük bir matris, görüntü üzerinde kaydırılarak evrişim işlemi gerçekleştirilir.
#Havuzlama (Pooling): Öznitelik haritasının boyutunu küçültmek ve hesaplama yükünü azaltmak için kullanılır. 
#Genellikle maksimum havuzlama veya ortalama havuzlama kullanılır.

###################################################################################################

#LSTM (Long Short-Term Memory):

#LSTM, özellikle sıralı verilerin işlenmesi için tasarlanmış bir tür rekürrent sinir ağıdır. 
#Sıralı veriler, zamansal veya dizisel bağlantıları içeren verilerdir. 
#LSTM'lerin temel amacı, bu tür verilerdeki bağlantıları anlamak ve gelecekteki değerleri tahmin etmektir.

#Hafıza Hücreleri (Memory Cells): LSTM'lerin temel bileşenleri hafıza hücreleridir. Bu hücreler, önceki zaman adımlarından gelen bilgileri saklayarak sıralı verilerdeki uzun vadeli bağlantıları yakalamayı sağlar.
#Giriş, Çıkış ve Unutma Kapıları: LSTM'ler, giriş verilerinin hafıza hücrelerine ne kadar ekleneceğini, hangi bilgilerin unutulacağını ve hafıza hücrelerinden çıkışın nasıl hesaplanacağını belirlemek için giriş, çıkış ve unutma kapıları gibi mekanizmaları kullanır.

#Farklar:

#CNN'ler özellikle görsel verilerin işlenmesi için kullanılırken, LSTM'ler sıralı verilerin işlenmesi için kullanılır.
#CNN'ler evrişim ve havuzlama katmanlarından oluşurken, LSTM'ler hafıza hücreleri ve giriş/çıkış mekanizmalarından oluşur.
#CNN'ler özellikle görüntü sınıflandırma, nesne tespiti gibi görevlerde kullanılırken, LSTM'ler zaman serisi tahmini, metin üretimi gibi sıralı veri görevlerinde kullanılır.


###################################################################################################
# Loading Fashion MNIST dataset
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

class_names = ["T-shirt / top", "Trouser", "Pullover", "Dress",
        "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

import matplotlib.pyplot as plt
plt.figure()
plt.imshow (X_train[1])
plt.colorbar()
plt.grid(False)
plt.show()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.Sequential ([
   tf.keras.layers.Flatten (input_shape = (28, 28), name = "Input"),
   tf.keras.layers.Dense (128, activation = 'relu', name = "Hidden"),
   tf.keras.layers.Dense (10, name = "Output")
])

hidden = model.layers[1]
print(hidden.name)

weights, biases = hidden.get_weights()
print(weights)
print(biases)

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits = True),
              optimizer = 'adam',
              metrics = ['accuracy'])

history = model.fit (X_train, y_train, epochs = 10, validation_split = 0.1)

import pandas as pd
pd.DataFrame (history.history).plot (figsize = (8, 5))
plt.grid(True)
plt.show()

test_loss, test_acc = model.evaluate (X_test, y_test, verbose = 2)
print ('\ nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential ([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(X_test)

predictions[0]

import numpy as np
np.argmax(predictions[0])

y_test[0]

###################################################################################################
#Bilgisayar Görüşü Devam#

from sklearn.datasets import load_sample_images

images=load_sample_images()["images"]
images[0].shape

#Veri Ön İşleme

images=tf.keras.layers.CenterCrop(height=80,width=120)(images)
images=tf.keras.layers.Rescaling(scale=1/255)(images)

#Convolution

conv_layer=tf.keras.layers.Conv2D(filters=32, kernel_size=7)
#filter=Öznitelik haritası 
fmaps=conv_layer(images)

conv_layer=tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding="same")
#padding ayarlanmazsa varsayılan yani valid olarak atanır. Bu da çıktı öznitelik haritasının giriş görüntü boyutlarından daha küçük olacağı anlamına gelir. 
#Bu durumda, çıktı öznitelik haritasının boyutları giriş görüntü boyutlarıyla aynı olacaktır. "same" padding, çekirdek kaydırılırken görüntünün kenarlarına sıfır değerler eklenerek boyut kaybının engellenmesini sağlar.
#Hangi yöntemin kullanılması gerektiği projenizin gereksinimlerine bağlıdır. 
#Eğer çıktı boyutları önemliyse ve boyut kaybını engellemek isteniyorsa, "same" padding kullanılabilir. 
#Eğer boyut kaybı önemli değilse ve çıktı öznitelik haritasının boyutları küçülebilirse, "valid" padding tercih edilebilir.
fmaps=conv_layer(images) 

kernels, biases=conv_layer.get_weights()
#kernel=Katman tarafından üretilen matrisin ağırlığını ölçen parametredir. 7 demek 7x7'lik bir çekirdek anlamına gelir.
#bias=Katman tarafından üretilen bias vektörüdür.

#Convolutional Katman
#Küçük bir grup sayı matrisinin ya da kernelin input görseline uygulanıp filtre edilerek özellik haritasının çıkarılması. (öznitelik elde etme)

#Pooling Katmanı
#CNN katmanının diğer bir building katmanı olarak görülebilir. Bu şekilde özellik haritasının boyutu küçültülerek sadece en ama en önemli bilgilerin elde edilmesi sağlanır.

max_pool=tf.keras.layers.MaxPool2D(pool_size=2)
output=max_pool(images)

global_avg_pool=tf.keras.layers.GlobalAvgPool2D()
global_avg_pool(images)

###################################################################################################
#Uygulama

import os

base_dir="/kaggle/input/rockpaperscissors/rps-cv-images"
paper_dir=os.path.join(base_dir,"paper")
rock_dir=os.path.join(base_dir,"rock")
scissors_dir=os.path.join(base_dir,"scissors")

import random

random_İmage=random.sample(os.listdir(paper_dir),1)
img=tf.keras.utils.load_img(f"{paper_dir}/{random_İmage[0]}")

img=tf.keras.utils.img_to_array(img)

#Veri Ön İşleme

train_ds=tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    image_size=(180,180),
    batch_size=32,
    seed=42 
    )

#for image_batch, labels_batch in train_ds:
#    print(image_batch.shape)
#    print(labels_batch.shape)
#    break 

class_names=train_ds.class_names

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

for images, labels in train_ds.take(1):
    for i in range (9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

val_ds=tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    image_size=(180,180),
    batch_size=32,
    seed=42 
    )

#Performans Arttırma

train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

#.cache() = Veri kümesinin bellekte veya diskte önbelleğe alınmasını sağlar. Bu, verilerin daha hızlı bir şekilde yüklenmesini sağlar. Özellikle veri kümesi küçük değilse veya her epoch'ta aynı verileri tekrar tekrar kullanıyorsanız faydalı olabilir.
#.prefetch(buffer_size=tf.data.AUTOTUNE) = Bu işlev, TensorFlow'un otomatik olarak uygun bir önbellek boyutunu seçmesini sağlar. Veri yükleme ve işleme süreçlerini asenkron hale getirir, yani bir işlem veriyi alırken diğer işlem veriyi önbelleğe alabilir. Bu, veri yükleme süreçlerinin daha verimli ve kesintisiz olmasını sağlar.

#Data Augmentation

data_augmentation=tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical",seed=42),
    tf.keras.layers.RandomRotation(0.1, seed=42)
])

#Model Kurma

model=tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(180,180,3)),
    data_augmentation,
    tf.keras.layers.Conv2D(128,3,padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128,3,padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Conv2D(128,3,padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128,3,padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dense(3)        
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

#Model Eğitimi

history=model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

#Doğruluk Değerlendirme

acc=history.history["history"]
val_acc=history.history["val_accuracy"]

loss=history.history["loss"]
val_loss=history.history["val_loss"]

img_array=tf.keras.utils.img_to_array(img)
img_array=tf.expand_dims(img_array,0)

predictions=model.predict(img_array)

import numpy as np

score=tf.nn.softmax(predictions[0])
class_names[np.argmax(score)]

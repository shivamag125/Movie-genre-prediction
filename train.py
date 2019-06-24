import numpy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

model= Sequential()
#Convolution first 
model.add(Conv2D(32, (3,3), input_shape= (64,64,3), strides=(2,2), padding='valid'))
model.add(Activation('relu'))
#depth wise 1
model.add(SeparableConv2D(32, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

#depth wise 2
model.add(SeparableConv2D(32, kernel_size=(3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))

#depth wise 3
model.add(SeparableConv2D(32, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

#depth wise 4 
model.add(SeparableConv2D(32, kernel_size=(3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))

#depth wise 5
model.add(SeparableConv2D(32, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))

#depth wise 6
#model.add(SeparableConv2D(512, kernel_size=(3,3), strides=(2,2), padding='valid'))
#model.add(Activation('relu'))

#depth wise *5
#model.add(SeparableConv2D(512, kernel_size=(3,3), strides=(1,1), padding='same'))
#model.add(Activation('relu'))

#model.add(SeparableConv2D(512, kernel_size=(3,3), strides=(1,1), padding='same'))
#model.add(Activation('relu'))

#model.add(SeparableConv2D(512, kernel_size=(3,3), strides=(1,1), padding='same'))
#model.add(Activation('relu'))

#model.add(SeparableConv2D(512, kernel_size=(3,3), strides=(1,1), padding='same'))
#model.add(Activation('relu'))

#model.add(SeparableConv2D(512, kernel_size=(3,3), strides=(1,1), padding='same'))
#model.add(Activation('relu'))

#depth wise 7
#model.add(SeparableConv2D(1024, kernel_size=(3,3), strides=(2,2), padding='valid'))
#model.add(Activation('relu'))

#depth wise 8
#model.add(SeparableConv2D(1024, kernel_size=(3,3), strides=(2,2), padding='same'))
#model.add(Activation('relu'))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.summary()
plot_model(model, to_file='mob_net_short_graph.png')
batch_size=16

train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)


train_gen=train_datagen.flow_from_directory(directory="/home/shivam/Documents/genre_pred/data_r",target_size=(64,64),color_mode='rgb',batch_size=32,class_mode='categorical')
valid_gen=test_datagen.flow_from_directory(directory="/home/shivam/Documents/genre_pred/data_r_test",target_size=(64,64),color_mode='rgb',batch_size=32,class_mode='categorical')

#model.fit_generator(train_gen,steps_per_epoch=50, epochs=100,validation_data=valid_gen,validation_steps=25)
#model.save_weights('/home/shivam/Documents/genre_pred/third_try.h5')

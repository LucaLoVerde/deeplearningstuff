import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow_core.contrib import util

batch_size = 50
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)
# data augmentation
data_generator = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1,
                                    featurewise_center=True,
                                    featurewise_std_normalization=True, horizontal_flip=True)
data_generator.fit(X_train)
# standardize test set
for i in range(len(X_test)):
    X_test[i] = data_generator.standardize(X_test[i])

# net arch
mdl = Sequential()
mdl.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
mdl.add(BatchNormalization())
mdl.add(Activation('elu'))
mdl.add(Conv2D(32, (3, 3), padding='same'))
mdl.add(BatchNormalization())
mdl.add(Activation('elu'))
mdl.add(MaxPooling2D(pool_size=(2, 2)))
mdl.add(Dropout(0.2))

mdl.add(Conv2D(64, (3, 3), padding='same'))
mdl.add(BatchNormalization())
mdl.add(Activation('elu'))

mdl.add(Conv2D(64, (3, 3), padding='same'))
mdl.add(BatchNormalization())
mdl.add(Activation('elu'))
mdl.add(MaxPooling2D(pool_size=(2, 2)))
mdl.add(Dropout(0.2))

mdl.add(Conv2D(128, (3, 3), padding='same'))
mdl.add(BatchNormalization())
mdl.add(Activation('elu'))
mdl.add(Conv2D(128, (3, 3), padding='same'))
mdl.add(BatchNormalization())
mdl.add(Activation('elu'))
mdl.add(MaxPooling2D(pool_size=(2, 2)))
mdl.add(Dropout(0.5))
mdl.add(Flatten())
mdl.add(Dense(10, activation='softmax'))
# optimizer
mdl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train
mdl.fit_generator(
    generator=data_generator.flow(x=X_train, y=Y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=10,
    validation_data=(X_test, Y_test),
    workers=4,
    use_multiprocessing=True
)
# Windows support for multiprocessing/multithreading is apparently pathetic, so this will probably fail. I must have
# spent a whole day trying to install different version of everything. I really think it's time for a proper linux box.

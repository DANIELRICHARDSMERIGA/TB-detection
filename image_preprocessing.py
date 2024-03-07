import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import random
from shutil import copyfile
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten

'''
# Define image size and batch size
IMG_SIZE = 224
BATCH_SIZE = 32

# Define data generators for train, validation and test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'archive/chest_xray/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'archive/chest_xray/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'archive/chest_xray/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Define the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
model.save("Model.h5","label.txt")
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
#print('Test accuracy:', test_acc)
np.save('saved_accuracy',test_acc)'''


extract_path = 'C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Image_dataset/'

n_Tuberculosis = len(os.listdir(extract_path+'Tuberculosis'))
n_normal = len(os.listdir(extract_path+'Normal'))

print("no. of TB images: {} and no. of normal images: {}".format(n_Tuberculosis,n_normal))

'''try:
    #os.mkdir('C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/train')
    #os.mkdir('C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/test')
    #os.mkdir('C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/val')

    #os.mkdir('C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/train/Tuberculosis')
    #os.mkdir('C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/train/Normal')

    #os.mkdir('C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/test/Tuberculosis')
    #os.mkdir('C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/test/Normal')

    #os.mkdir('C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/val/Tuberculosis')
    #os.mkdir('C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/val/Normal')
except OSError:
    print('file')'''


def split_data(SOURCE,TRAINING,VALIDATION,TEST):
    all_file = []

    for file_name in os.listdir(SOURCE):
        file_path = SOURCE+file_name

        if os.path.getsize(file_path):
            all_file.append(file_name)
        else:
            print("{} is zero length. So ignore".format(file_name))
    
    n_file  = len(all_file)
    split_point = round(n_file * 0.8)
    split_point_1 = round(n_file *  0.9)

    shuffled = random.sample(all_file,n_file)

    train_set = all_file[:split_point]
    val_set = all_file[split_point:split_point_1]
    test_set = all_file[split_point_1:]

    for file_name in train_set:
        copyfile(SOURCE + file_name,TRAINING + file_name)
    
    for file_name in test_set:
        copyfile(SOURCE+file_name,TEST+file_name)
    
    for file_name in val_set:
        copyfile(SOURCE+file_name , VALIDATION+file_name)


Tuberculosis_source = "C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Image_dataset/Tuberculosis/"
Normal_source = "C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Image_dataset/Normal/"

train_Tuberculosis = "C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/train/Tuberculosis/"
train_Normal = "C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/train/Normal/"

val_Tuberclosis = "C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/val/Tuberculosis/"
val_Normal = "C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/val/Normal/"

test_Tuberclosis = "C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/test/Tuberculosis/"
test_Normal = "C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/test/Normal/"

split_data(Tuberculosis_source,train_Tuberculosis,val_Tuberclosis,test_Tuberclosis)
split_data(Normal_source,train_Normal,val_Normal,test_Normal)


path_ = "C:/Users/richa/Downloads/archive (1)/archive/TB_Chest_Radiography_Database/Split_data/"


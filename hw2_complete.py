### Add lines to import modules as needed
import tensorflow as tf
import numpy as np
from keras import Input, layers
## 
def build_model1():
  model = tf.keras.Sequential([
    Input(shape = input_size),
    layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2)),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2)),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2)),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size = (4,4), strides = (4,4)),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation = 'relu')
  ])
  return model

def build_model2():
  model = tf.keras.Sequential([
    Input(shape = input_size),
    layers.SeparableConv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2)),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.SeparableConv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size = (4,4), strides = (4,4)),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(10, activation = 'relu')
  ])
  return model

def build_model3():
  input = tf.keras.Input(shape = input_size)
  Conv1 = layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2))
  x = Conv1(input)
  x = layers.Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2))(x)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2))(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.25)(x)
  x = layers.MaxPooling2D(pool_size = (4,4), strides = (4,4))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation = 'relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.25)(x)
  output = layers.Dense(10, activation = 'relu')(x)
  
  model = tf.keras.Model(inputs = input, outputs = output, name = 'model3')
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = tf.keras.Sequential([
    Input(shape = input_size),
    layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same',strides = (2,2)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.MaxPooling2D(pool_size = (4,4), strides = (4,4)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(10, activation = 'relu')
  ])
  return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  #20% validation set
  val_samples = int(len(train_images)*0.2)
  val_id = np.random.choice(np.arange(len(train_images)), size = val_samples, replace = False)
  train_id = np.setdiff1d(np.arange(len(train_images)), val_id)
  
  #set train and val images
  val_im = train_images[val_id, :,:,:]
  train_im = train_images[train_id, :,:,:]
  #set train and val labels
  val_labels = train_labels[val_id].squeeze()
  train_labels = train_labels[train_id].squeeze()
  #designate input size for model
  input_size = train_im.shape[1:]
  
  ########################################
  ## Build and train model 1
  model1 = build_model1()
  # compile and train model 1.
  model1.compile(optimizer = 'adam',
               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
               metrics = ['accuracy']
               )
  model1.summary()
  
  #model1_out = model1.fit(train_im,train_labels,validation_data = (val_im, val_labels), epochs = 50)
  
  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  # compile and train model 1.
  model2.compile(optimizer = 'adam',
               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
               metrics = ['accuracy']
               )
  model2.summary()
  
  #model2_out = model2.fit(train_im,train_labels,validation_data = (val_im, val_labels), epochs = 30)
  
  
  ### Repeat for model 3 and your best sub-50k params model
  model3 = build_model3()
  # compile and train model 1.
  model3.compile(optimizer = 'adam',
               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
               metrics = ['accuracy']
               )
  model3.summary()
  
  #model3_out = model3.fit(train_im,train_labels,validation_data = (val_im, val_labels), epochs = 30)
  
  
  #model 50k
  model50k = build_model50k()
  # compile and train model 1.
  model50k.compile(optimizer = 'adam',
               loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
               metrics = ['accuracy']
               )
  model50k.summary()
  
  model50k_out = model50k.fit(train_im,train_labels,validation_data = (val_im, val_labels), epochs = 30)
  
  model50k.save('best_model.h5')
  
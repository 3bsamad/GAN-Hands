# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:40:11 2020

@author: ctj-oe
"""



## TRAINING THE WHOLE REAL HANDS DATASET ON DCGAN ##

#Essential libraries 
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

#Path to the images needed for training and testing
#Main directory, which contains subfolder "faces" containing the faces dataset

data_dir = "F:\Omar\Organized RealHands GANeratedDataset\colorBG"


# Generation resolution - Must be square 
# Training data is also scaled to this.
GENERATE_RES = 2 # Generation resolution factor 
# (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3

# Preview image 
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
DATA_PATH = data_dir
EPOCHS = 10000
BATCH_SIZE = 128
BUFFER_SIZE = 40000

print(f"Will generate {GENERATE_SQUARE}px square images.")






import pathlib
#Data loading and preprocessing
# Image set has 210k+ images.  Can take some time 
# for initial preprocessing.
# Because of this time needed, save a Numpy preprocessed file.
# Note, that file is large enough to cause problems for 
# some verisons of Pickle,
# so Numpy binary files are used.
# ~~ This is the first method to load the dataset, using arrays. Other method uses
# keras: tf.keras.preprocessing.image_dataset_from_directory

data_dir = pathlib.Path(data_dir)




#first method to create the ds
train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode=None,
    image_size=(GENERATE_SQUARE,GENERATE_SQUARE),
    batch_size=BATCH_SIZE,
    shuffle=False
    )

for image_batch in train_ds:
  print(image_batch.shape)
  break



#second method
training_binary_path = os.path.join(DATA_PATH,
        f'training_data_colorhands_all_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')

print(f"Looking for file: {training_binary_path}")

if not os.path.isfile(training_binary_path):
  start = time.time()
  print("Loading training images...")

  training_data = []
  
  #paths to the training data
  #this is a workaround to read files as array from multiple subdirectories
    
  path1 = os.path.join(DATA_PATH,'color1')
  path2 = os.path.join(DATA_PATH,'color1static')
  path3 = os.path.join(DATA_PATH,'color2')
  path4 = os.path.join(DATA_PATH,'color3')
  path5 = os.path.join(DATA_PATH,'color4_1')
  path6 = os.path.join(DATA_PATH,'color4_2')
  path7 = os.path.join(DATA_PATH,'color5_1')
  path8 = os.path.join(DATA_PATH,'color5_2')
  path9 = os.path.join(DATA_PATH,'color6_1')
  path10 = os.path.join(DATA_PATH,'color6_2')
  path11= os.path.join(DATA_PATH,'color6_3')
  path12 = os.path.join(DATA_PATH,'color7')
  path13 = os.path.join(DATA_PATH,'color7static')

    
    
  """"multiple loops to read the data. 13 loops for 13 directories"""
  
  for filename in tqdm(os.listdir(path1)):
      path = os.path.join(path1,filename)
      image = Image.open(path).resize((GENERATE_SQUARE,
            GENERATE_SQUARE),Image.ANTIALIAS)
      training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path2)):
          path = os.path.join(path2,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path3)):
          path = os.path.join(path3,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path4)):
          path = os.path.join(path4,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path5)):
          path = os.path.join(path5,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path6)):
          path = os.path.join(path6,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path7)):
          path = os.path.join(path7,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path8)):
          path = os.path.join(path8,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path9)):
          path = os.path.join(path9,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path10)):
          path = os.path.join(path10,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path11)):
          path = os.path.join(path11,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))

  for filename in tqdm(os.listdir(path12)):
          path = os.path.join(path12,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image)) 

  for filename in tqdm(os.listdir(path13)):
          path = os.path.join(path13,filename)
          image = Image.open(path).resize((GENERATE_SQUARE,
                GENERATE_SQUARE),Image.ANTIALIAS)
          training_data.append(np.asarray(image))           
      
      
  training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,
            GENERATE_SQUARE,IMAGE_CHANNELS))
  training_data = training_data.astype(np.float32)
  training_data = training_data / 127.5 - 1.





  print("Saving training image binary...")
  np.save(training_binary_path,training_data)
  elapsed = time.time()-start
  print (f'Image preprocess time: {hms_string(elapsed)}')
else:
  print("Loading previous training pickle...")
  training_data = np.load(training_binary_path)



train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)




from tensorflow.keras import layers

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x: (normalization_layer(x)))
image_batch = next(iter(normalized_ds))
first_image = image_batch[8]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))



"""  °°° *** ~~~ Building the model ~~~ *** °°°  """


#generator
def build_generator(seed_size, channels):
    model = Sequential()

    model.add(Dense(4*4*256,activation="relu",input_dim=seed_size))
    model.add(Reshape((4,4,256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    #model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    #model.add(LeakyReLU(alpha=0.2))
   
    # Output resolution, additional upsampling
    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    #model.add(LeakyReLU(alpha=0.2))

    if GENERATE_RES>1:
      model.add(UpSampling2D(size=(GENERATE_RES,GENERATE_RES)))
      model.add(Conv2D(128,kernel_size=3,padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(Activation("relu"))
      #model.add(LeakyReLU(alpha=0.2))

    # Final CNN layer
    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    model.add(Activation("tanh"))

    model.summary()
    return model

#discriminator
def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, 
                     padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.5))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.5))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    return model

generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)

noise = tf.random.normal([1, SEED_SIZE])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])


image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)

discriminator = build_discriminator(image_shape)
decision = discriminator(generated_image)
print (decision)



""" show images during training """

def save_images(cnt,noise):
  image_array = np.full(( 
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3), 
      255, dtype=np.uint8)
  
  generated_images = generator.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255
        image_count += 1

          
  output_path = os.path.join(DATA_PATH,'output')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  filename = os.path.join(output_path,f"train-{cnt}.png")
  im = Image.fromarray(image_array)
  im.save(filename)





""" Loss functions """

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

""" Both the generator and discriminator use Adam and the same learning rate and momentum. 
This does not need to be the case. If you use a GENERATE_RES greater than 3 you may need to
tune these learning rates, as well as other training and hyperparameters. """
#Optimizers
generator_optimizer = tf.keras.optimizers.Adam(0.0002,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0002,0.5)





def plot_history(g, d):
	# plot loss
	plt.plot(d, label='Discriminator Loss')
	plt.plot(g, label='Generator Loss')
	plt.legend()
    
	# save plot to file
	plt.savefig('losses')
	plt.close()





""" TRAINING """


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(seed, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  return gen_loss,disc_loss


gloss=[]
dloss=[]

def train(dataset, epochs):
  fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
  start = time.time()

  for epoch in range(epochs):
    epoch_start = time.time()

    gen_loss_list = []
    disc_loss_list = []

    for image_batch in dataset:
      t = train_step(image_batch)
      gen_loss_list.append(t[0])
      disc_loss_list.append(t[1])

    g_loss = sum(gen_loss_list) / len(gen_loss_list)
    d_loss = sum(disc_loss_list) / len(disc_loss_list)

    
    gloss.append(f'{g_loss}')
    dloss.append(f'{d_loss}')
    
    epoch_elapsed = time.time()-epoch_start
    print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},'\
           ' {hms_string(epoch_elapsed)}')
    save_images(epoch,fixed_seed)

  elapsed = time.time()-start
  print (f'Training time: {hms_string(elapsed)}')
  
  
  
  
train(train_dataset, EPOCHS)

generator.save(os.path.join(DATA_PATH,"hands_generator.h5"))

plot_history(gloss,dloss)

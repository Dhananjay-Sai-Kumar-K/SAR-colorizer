import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout

# Define paths to data
grayscale_images_path =  r"C:\Users\dhana\Downloads\SAR Orig\v_2\urban\s1"
colorful_images_path =  r"C:\Users\dhana\Downloads\SAR Orig\v_2\urban\s2"

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Function to load and preprocess images
def load_images(image_path, img_height, img_width, is_color):
    images = []
    for filename in os.listdir(image_path): #returns a list cont files avl in the image_path
        #data collection
        img = cv2.imread(os.path.join(image_path, filename), cv2.IMREAD_COLOR if is_color else cv2.IMREAD_GRAYSCALE)
        #imread - reads an image, cv2.IMREAD_COLOR - if colorimage, cv2.IMREAD_GRAYSCALE - grey
        if img is not None:
            #1st Preprocessing
            img = cv2.resize(img, (img_height, img_width))
            #resizes img to 256X256 - widely used size
            if not is_color: 
                # for grey img(SAR Orig img) it adds another channel(like R,G,B Channel) becoz U-Net model needs it
                img = np.expand_dims(img, axis=-1)  # Adds channel dimension for grayscale images as U-net pre-trained on M's of Colour images
            images.append(img)
    return np.array(images) 
    #Returns the list of images as a NumPy array, which is often a convenient format for working with images in Python.

# Load and preprocess data
X = load_images(grayscale_images_path, IMG_HEIGHT, IMG_WIDTH, is_color=False)
y = load_images(colorful_images_path, IMG_HEIGHT, IMG_WIDTH, is_color=True)

# Normalize images
X = X / 255.0
y = y / 255.0

def unet(input_shape):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)  # Added layer
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)  # Added layer
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # Added layer
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)  # Added layer
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottleneck
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)  # Added layer
    
    # Decoder
    up6 = concatenate([UpSampling2D((2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)  # Added layer

    up7 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)  # Added layer

    up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)  # Added layer

    up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)  # Added layer

    # Output layer
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    # Model definition
    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# Initialize the model
input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
model = unet(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Use the data generator for training
train_datagen = datagen.flow(X_train, y_train, batch_size=32)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Initialize the Adam optimizer
adam_optimizer = Adam(learning_rate=0.0009)

# Compile the model with the selected optimizer
model.compile(optimizer=adam_optimizer,  # use the initialized optimizer
              loss='mean_squared_error',  # corrected syntax
              metrics=['accuracy'])

# Make sure to set `verbose=1` to show the progress bar
history = model.fit(
    train_datagen, 
    epochs=60, 
    steps_per_epoch=len(X_train) // 32,  # use integer division here
    validation_data=(X_test, y_test),
    verbose=1  # Ensure progress bar is shown
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_input_image(image_path, img_height, img_width):
    """
    Load and preprocess a single SAR grayscale image for prediction.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to the required dimensions
    img = cv2.resize(img, (img_width, img_height))
    
    # Normalize the image
    img = img / 255.0
    
    # Expand dimensions to match input shape (1, height, width, channels)
    img = np.expand_dims(img, axis=(0, -1))
    
    return img

def predict_and_display(model, image_path, img_height, img_width):
    """
    Predict the color output from a SAR grayscale input image and display the result.
    """
    # Preprocess the input image
    input_image = preprocess_input_image(image_path, img_height, img_width)
    
    # Predict the output color image using the trained model
    predicted_image = model.predict(input_image)
    
    # Remove the batch dimension
    predicted_image = np.squeeze(predicted_image, axis=0)
    
    # Ensure the output is in the range [0, 1]
    predicted_image = np.clip(predicted_image, 0, 1)
    
    # Display the input and output images side-by-side
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title('Input SAR Grayscale Image')
    plt.imshow(np.squeeze(input_image), cmap='gray')  # Remove the channel dimension for displaying grayscale image
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Predicted Color Output Image')
    plt.imshow(predicted_image)
    plt.axis('off')
    
    plt.show()

# Example usage
image_path = r"C:\Users\dhana\Downloads\SAR V3\SAR_Gmini\ROIs1970_fall_s1_145_p5.png"
predict_and_display(model, image_path, IMG_HEIGHT, IMG_WIDTH)


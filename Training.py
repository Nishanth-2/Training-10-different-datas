# Mount Google Drive to access your dataset
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Define the path to your dataset in Google Drive
dataset_dir = '/content/drive/My Drive/path_to_your_folder/ANIMALS-10'  # Replace with your Google Drive folder path

# Create ImageDataGenerators for training and validation with a split
datagen = ImageDataGenerator(
    rescale=1.0/255,               # Rescale pixel values from [0, 255] to [0, 1]
    validation_split=0.2,          # Reserve 20% of the data for validation
)

# Training data generator
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),         # Resize all images to 150x150 pixels
    batch_size=32,                  # Process 32 images at a time
    class_mode='categorical',       # Perform one-hot encoding for 10 classes
    subset='training'               # Use 80% of the data for training
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),         # Resize all images to 150x150 pixels
    batch_size=32,                  # Process 32 images at a time
    class_mode='categorical',       # Perform one-hot encoding for 10 classes
    subset='validation'             # Use 20% of the data for validation
)

# Building a simple neural network model
model = Sequential()

# Flatten the image data (convert 150x150x3 into a 1D vector of length 67500)
model.add(Flatten(input_shape=(150, 150, 3)))

# First Dense layer with 512 neurons
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Regularization to prevent overfitting

# Second Dense layer with 256 neurons
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Output layer with 10 neurons for 10 categories
model.add(Dense(10, activation='softmax'))  # Output probabilities for each of the 10 categories

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,                                                               # Train for 50 epochs
    validation_data=validation_generator,                                    # Use the validation data for validation
    validation_steps=validation_generator.samples // validation_generator.batch_size  # Number of steps in validation
)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Save the trained model
model.save('/content/drive/My Drive/animal_classifier_mlp.h5')  # Save the model in Google Drive

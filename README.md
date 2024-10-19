ANIMALS-10 Image Classification with Neural Networks
This project implements a simple neural network model to classify images from the ANIMALS-10 dataset. The dataset contains images of 10 different categories of animals, and this neural network is trained to distinguish between them.

Project Overview
The project uses a fully connected neural network (MLP) to classify images into 10 categories. It loads the images from Google Drive, preprocesses them using ImageDataGenerator, trains the model, and saves it for future use.

Table of Contents

Requirements
Dataset
Installation and Setup
Training the Model
Testing the Model
Results
Future Improvements


Requirements

The following Python libraries are required to run the project:
TensorFlow
Keras
NumPy
Google Colab (for Google Drive access)
PyDrive (optional, for accessing Google Drive)

You can install the required packages using pip:
pip install tensorflow numpy

Dataset
The dataset used for this project is the ANIMALS-10 dataset, which contains images of 10 different animal categories. The dataset should be organized into subfolders, each representing a different class.
ANIMALS-10/
    ├── cat/
    ├── dog/
    ├── elephant/
    ├── ... (other animal categories)


Installation and Setup
1. Google Drive Setup
This project uses Google Drive for storing the dataset and saving the model. To use this setup:

Upload the dataset to your Google Drive.
Mount the Google Drive in Google Colab.
2. Cloning the Project
Clone this repository or download the project files.

3. Google Colab Setup
Open the project in Google Colab.
Mount your Google Drive with the following code
from google.colab import drive
drive.mount('/content/drive')

4. Accessing the Dataset from Google Drive
After mounting, set the path to the dataset folder in your Google Drive:
dataset_dir = '/content/drive/My Drive/path_to_your_folder/ANIMALS-10'


Training the Model
To train the model, you need to load the dataset from Google Drive and preprocess it. The training code is as follows:
# Load and preprocess data using ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,  # 20% validation split
)

# Load the dataset for training and validation
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build, compile, and train the model
model = Sequential()
model.add(Flatten(input_shape=(150, 150, 3)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_generator, epochs=50, validation_data=validation_generator)

Saving the Model
After training, the model can be saved to Google Drive for future use:
model.save('/content/drive/My Drive/animal_classifier_mlp.h5')


Testing the Model
You can test the saved model by loading it and using new test images. Here’s how to do it:

# Load the saved model
model = tf.keras.models.load_model('/content/drive/My Drive/animal_classifier_mlp.h5')

# Define the path to your test image
test_image_path = '/content/drive/My Drive/path_to_your_test_image/test_image.jpg'

# Preprocess the image and make a prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

test_image = preprocess_image(test_image_path)
predictions = model.predict(test_image)

# Print the predicted class
predicted_class = np.argmax(predictions, axis=1)[0]
print(f"Predicted class: {predicted_class}")

Testing on Multiple Images
If you want to test the model on multiple images, you can use the ImageDataGenerator to load a test folder and make predictions on multiple test images at once.

Results
After training, the model can achieve a reasonable level of accuracy in classifying the 10 animal categories. The results will vary depending on the size of the dataset and the number of epochs trained.

You can evaluate the model’s performance on the validation set using:
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy}")



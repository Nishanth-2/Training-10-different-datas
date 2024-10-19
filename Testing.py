import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('/content/drive/My Drive/animal_classifier_mlp.h5')

# Define the path to your test image
test_image_path = '/content/drive/My Drive/path_to_your_test_image/test_image.jpg'  # Replace with the path to your test image

# Load and preprocess the test image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))  # Load the image and resize it to 150x150
    img = img_to_array(img)  # Convert the image to an array
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img = img / 255.0  # Normalize the image (same as in training)
    return img

# Preprocess the test image
test_image = preprocess_image(test_image_path)

# Predict the class probabilities
predictions = model.predict(test_image)

# Get the class labels
class_indices = train_generator.class_indices  # Assuming you used the same ImageDataGenerator for training
class_labels = {v: k for k, v in class_indices.items()}  # Invert the class_indices to get class labels

# Get the predicted class
predicted_class = np.argmax(predictions, axis=1)[0]

# Print the predicted class
print(f"Predicted class: {class_labels[predicted_class]}")

# Optionally, print the predicted probabilities for each class
print("Class probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"{class_labels[i]}: {prob:.4f}")

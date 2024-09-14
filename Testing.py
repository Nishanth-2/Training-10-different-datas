from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
img_path = '/content/CAT.jpeg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale the image just like during training

# Predict the category
predictions = model.predict(img_array)

# Decode the prediction
predicted_class_index = np.argmax(predictions, axis=1)
class_labels = list(train_generator.class_indices.keys())  # Get class labels from the training generator
predicted_class_label = class_labels[predicted_class_index[0]]

print(f"Predicted class: {predicted_class_label}")

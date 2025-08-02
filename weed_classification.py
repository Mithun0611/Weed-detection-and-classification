import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# def load_images(data_path, img_size=(64, 64)):
#     images = []
#     labels = []
#     class_names = os.listdir(data_path)
#     for index, cls in enumerate(class_names):
#         cls_folder = os.path.join(data_path, cls)
#         for img_name in os.listdir(cls_folder):
#             img_path = os.path.join(cls_folder, img_name)
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, img_size)
#             images.append(img)
#             labels.append(index)
#     return np.array(images), np.array(labels), class_names

def load_images(data_path, img_size=(64, 64)):
    images = []
    labels = []
    class_names = os.listdir(data_path)
    for index, cls in enumerate(class_names):
        cls_folder = os.path.join(data_path, cls)
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(index)
    return np.array(images), np.array(labels), class_names

data_path = "datasets"  
x, y, class_names = load_images(data_path)
x = x.astype('float32') / 255.0
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.save("weed_classifier_model.h5")


loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()


y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true_labels, y_pred_labels)
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def predict_new_image(img_path, threshold=0.6):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img)[0] 
    class_index = np.argmax(pred)
    confidence = pred[class_index]
    
    if confidence >= threshold:
        print(f"Predicted Class: {class_names[class_index]} with confidence {confidence*100:.2f}%")
    else:
        print(f"Prediction confidence too low ({confidence*100:.2f}%) → Class: Unknown")


test_folder = "test_images"



for file in os.listdir(test_folder):
    path = os.path.join(test_folder, file)
    print(f"\nTesting image: {file}")
    predict_new_image(path, threshold=0.6)



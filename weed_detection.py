import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import cv2

# Training
train_dir = 'E:/Industrial project/weed detection'
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(train_dir, target_size=(128, 128), class_mode='binary', subset='training', shuffle=True)
val_data = datagen.flow_from_directory(train_dir, target_size=(128, 128), class_mode='binary', subset='validation', shuffle=False)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=5, validation_data=val_data)

# Confusion Matrix
val_data.reset()
y_true = val_data.classes
y_pred_prob = model.predict(val_data, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
cm = confusion_matrix(y_true, y_pred)
class_names = list(val_data.class_indices.keys())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix on Validation Data')
plt.show()


# Test image clear prediction with corrected rectangle
test_image_path = r'weed detection\weed_images\agri_0_122.jpeg'
try:
    img_pil = load_img(test_image_path, target_size=(128, 128))
    img_arr = img_to_array(img_pil) / 255.0
    img_expanded = np.expand_dims(img_arr, axis=0)
    pred = model.predict(img_expanded, verbose=0)[0][0]
    
    # Load original image for display (not resized to 128x128, but displayed in 512x512)
    img_cv = cv2.cvtColor(np.array(load_img(test_image_path)), cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (512, 512))  # Resize for better visibility

    if pred > 0.5:
        print(f"The test image is predicted as: WEED with {pred*100:.2f}%")

        # Draw a centered rectangle around the image
        top_left = (50, 50)
        bottom_right = (462, 462)  # 512 - 50 = 462
        cv2.rectangle(img_cv, top_left, bottom_right, (0, 0, 255), 3)
        
        # Put 'Weed' text at top-left corner of the rectangle
        text_position = (top_left[0] + 10, top_left[1] + 40)
        cv2.putText(img_cv, 'Weed', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Show image
        cv2.imshow('Weed Detection', img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print(f"The test image is predicted as: CROP with {(1-pred)*100:.2f}%")

except FileNotFoundError:
    print(f"Test image not found. Check the path.")

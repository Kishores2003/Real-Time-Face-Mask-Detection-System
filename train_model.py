import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from mask_detector_model import create_model

img_width, img_height = 224, 224
batch_size = 32
epochs = 20

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'data/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

model = create_model()
from tensorflow.keras.metrics import Precision, Recall

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy', Precision(), Recall()])

checkpoint = ModelCheckpoint('mask_detector_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint, early_stop]
)

model.save('mask_detector_model.keras')

train_acc = history.history['accuracy'][-1]
train_precision = history.history['precision'][-1]
train_recall = history.history['recall'][-1]
val_acc = history.history['val_accuracy'][-1]
val_precision = history.history['val_precision'][-1]
val_recall = history.history['val_recall'][-1]

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Training Precision: {train_precision:.2f}")
print(f"Training Recall: {train_recall:.2f}")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Validation Precision: {val_precision:.2f}")
print(f"Validation Recall: {val_recall:.2f}")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/test',  
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

y_pred_test = model.predict(test_generator)
y_pred_test = (y_pred_test > 0.5).astype("int32")

y_true_test = test_generator.classes

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

test_accuracy = accuracy_score(y_true_test, y_pred_test)
test_precision = precision_score(y_true_test, y_pred_test)
test_recall = recall_score(y_true_test, y_pred_test)
test_f1 = f1_score(y_true_test, y_pred_test)


print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Precision: {test_precision:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test F1 Score: {test_f1:.2f}")

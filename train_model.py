import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

dataset_dir = '/Users/hayubin/Downloads/Project dataset/train/'
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir, labels='inferred', label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH), interpolation='nearest',
    batch_size=BATCH_SIZE, shuffle=True, seed=123,
    validation_split=0.2, subset='training'
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir, labels='inferred', label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH), interpolation='nearest',
    batch_size=BATCH_SIZE, shuffle=False, seed=123,
    validation_split=0.2, subset='validation'
)

class_names = train_dataset.class_names
num_classes = len(class_names)

rescale_layer = tf.keras.layers.Rescaling(1./255)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
])

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
    rescale_layer,
    data_augmentation,
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint_filepath = 'best_road_object_model_val_acc.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

epochs = 50

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    callbacks=[early_stopping_callback, model_checkpoint_callback]
)

if history and history.history:
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    if acc:
        epochs_ran = range(len(acc))
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_ran, acc, 'bo-', label='Training Accuracy')
        if val_acc:
             plt.plot(epochs_ran, val_acc, 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy (128x128)')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_ran, loss, 'bo-', label='Training Loss')
        if val_loss:
            plt.plot(epochs_ran, val_loss, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss (128x128)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        pass
else:
    pass

if os.path.exists(checkpoint_filepath):
    best_model = tf.keras.models.load_model(checkpoint_filepath)

    val_loss, val_accuracy = best_model.evaluate(validation_dataset, verbose=1)

    y_pred_probabilities = best_model.predict(validation_dataset)
    y_pred_classes = np.argmax(y_pred_probabilities, axis=1)
    y_true_classes = []
    for images, labels in validation_dataset:
        y_true_classes.extend(np.argmax(labels.numpy(), axis=1))
    y_true_classes = np.array(y_true_classes)

    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Best Model by val_accuracy)')
    plt.show()

    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0))
else:
    pass
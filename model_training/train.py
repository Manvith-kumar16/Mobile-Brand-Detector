import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import os

DATASET_PATH = r'C:\Users\manvi\OneDrive\mobile-detector-project\data\train'
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
EPOCHS = 100 
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset path '{DATASET_PATH}' not found.")
    print("Please update the DATASET_PATH variable to the correct location.")
else:
    print(f"Dataset found at: {DATASET_PATH}")
    print("Starting data loading and model training...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,     
        width_shift_range=0.3,  
        height_shift_range=0.3, 
        shear_range=0.3,        
        zoom_range=0.3,         
        horizontal_flip=True,   
        fill_mode='nearest',    
        validation_split=0.2    
    )
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = train_generator.num_classes

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x) 
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)


    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model with the EarlyStopping callback
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping] # Add the callback here
    )
    
    # --- Save the Trained Model and Class Labels ---
    model.save('mobile_detector_model.h5')
    print("Model saved as mobile_detector_model.h5")

    with open('class_labels.txt', 'w') as f:
        for cls in train_generator.class_indices.keys():
            f.write(f'{cls}\n')
    print("Class labels saved to class_labels.txt")

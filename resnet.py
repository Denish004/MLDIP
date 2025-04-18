# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision import models, transforms
# # from PIL import Image
# # import pandas as pd
# # import os

# # # Custom Dataset
# # class MultiLabelEuroSAT(Dataset):
# #     def __init__(self, img_dir, label_csv, transform=None):
# #         self.img_dir = img_dir
# #         self.transform = transform
# #         self.df = pd.read_csv(label_csv)
# #         self.num_classes = 10

# #     def __len__(self):
# #         return len(self.df)

# #     def __getitem__(self, idx):
# #         img_path = os.path.join(self.img_dir, self.df.iloc[idx]["image"])
# #         image = Image.open(img_path).convert("RGB")
# #         if self.transform:
# #             image = self.transform(image)

# #         label_indices = eval(self.df.iloc[idx]["labels"])
# #         labels = torch.zeros(self.num_classes)
# #         labels[label_indices] = 1.0
# #         return image, labels

# # # Transforms
# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                          std=[0.229, 0.224, 0.225])
# # ])

# # # Dataset and DataLoader
# # dataset = MultiLabelEuroSAT("multi_label_data", "labels.csv", transform)
# # loader = DataLoader(dataset, batch_size=16, shuffle=True)

# # # Model
# # model = models.resnet18(pretrained=True)
# # model.fc = nn.Sequential(
# #     nn.Linear(model.fc.in_features, 10),
# #     nn.Sigmoid()  # for multi-label
# # )
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = model.to(device)

# # # Training setup
# # criterion = nn.BCELoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # # Training loop
# # for epoch in range(5):
# #     model.train()
# #     total_loss = 0.0
# #     correct = 0
# #     total = 0

# #     for images, labels in loader:
# #         images, labels = images.to(device), labels.to(device)

# #         optimizer.zero_grad()
# #         outputs = model(images)
# #         loss = criterion(outputs, labels)
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()
# #         preds = outputs > 0.5
# #         correct += (preds == labels.bool()).all(dim=1).sum().item()
# #         total += labels.size(0)

# #     acc = correct / total * 100
# #     print(f"✅ Epoch {epoch+1}/5 - Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# # # Save model
# # torch.save(model.state_dict(), "resnet_eurosat_multilabel1.pth")
# # print("💾 Model saved as resnet_eurosat_multilabel.pth")






# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.data import Dataset, DataLoader
# # # from torchvision import models, transforms
# # # from PIL import Image
# # # import pandas as pd
# # # import os

# # # # ---------------------
# # # # Custom Dataset
# # # # ---------------------
# # # class MultiLabelEuroSAT(Dataset):
# # #     def __init__(self, img_dir, label_csv, transform=None):
# # #         self.img_dir = img_dir  # Dataset directory: "multi_label_data"
# # #         self.transform = transform
# # #         self.df = pd.read_csv(label_csv)  # CSV file: "labels.csv"
# # #         self.num_classes = 10
# # #         print(f"[INFO] Loaded {len(self.df)} entries from {label_csv}")

# # #     def __len__(self):
# # #         return len(self.df)

# # #     def __getitem__(self, idx):
# # #         img_path = os.path.join(self.img_dir, self.df.iloc[idx]["image"])
# # #         image = Image.open(img_path).convert("RGB")
# # #         if self.transform:
# # #             image = self.transform(image)

# # #         label_indices = eval(self.df.iloc[idx]["labels"])  # e.g., "[0, 3, 7]"
# # #         labels = torch.zeros(self.num_classes)
# # #         labels[label_indices] = 1.0
# # #         return image, labels

# # # # ---------------------
# # # # Transforms, Dataset, and DataLoader
# # # # ---------------------
# # # transform = transforms.Compose([
# # #     transforms.Resize((224, 224)),
# # #     transforms.ToTensor(),
# # #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# # #                          std=[0.229, 0.224, 0.225])
# # # ])

# # # dataset = MultiLabelEuroSAT("multi_label_data", "labels.csv", transform)
# # # loader = DataLoader(dataset, batch_size=16, shuffle=True)
# # # print(f"[INFO] Dataset loaded: {len(dataset)} samples, DataLoader created.")

# # # # ---------------------
# # # # Model Setup
# # # # ---------------------
# # # # Using ResNet18 pretrained on ImageNet
# # # model = models.resnet18(pretrained=True)
# # # model.fc = nn.Sequential(
# # #     nn.Linear(model.fc.in_features, 10),
# # #     nn.Sigmoid()  # For multi-label output
# # # )
# # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # model = model.to(device)
# # # print(f"[INFO] Model loaded on device: {device}")
# # # print(model)  # Print model architecture for verification

# # # # ---------------------
# # # # Training Setup
# # # # ---------------------
# # # criterion = nn.BCELoss()
# # # optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # # # ---------------------
# # # # Training Loop
# # # # ---------------------
# # # num_epochs = 5
# # # for epoch in range(num_epochs):
# # #     model.train()
# # #     total_loss = 0.0
# # #     correct = 0
# # #     total = 0

# # #     for images, labels in loader:
# # #         images, labels = images.to(device), labels.to(device)

# # #         optimizer.zero_grad()
# # #         outputs = model(images)
# # #         loss = criterion(outputs, labels)
# # #         loss.backward()
# # #         optimizer.step()

# # #         total_loss += loss.item()
# # #         preds = outputs > 0.5  # Using 0.5 as threshold for each class
# # #         correct += (preds == labels.bool()).all(dim=1).sum().item()
# # #         total += labels.size(0)

# # #     acc = correct / total * 100
# # #     print(f"✅ Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# # # # ---------------------
# # # # Save the Model
# # # # ---------------------
# # # torch.save(model.state_dict(), "resnet_eurosat_multilabel2.pth")
# # # print("💾 Model saved as resnet_eurosat_multilabel1.pth")
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.initializers import glorot_uniform, random_uniform
# import matplotlib.pyplot as plt

# # Check for GPU availability
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     print(f"Found {len(physical_devices)} GPU(s):")
#     for device in physical_devices:
#         print(device)
#     # Set memory growth to prevent TensorFlow from consuming all GPU memory
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
#     print("Using GPU.")
# else:
#     print("No GPU found. Using CPU.")

# # Optionally, set the device to use the first available GPU
# if len(physical_devices) > 0:
#     try:
#         tf.config.set_visible_devices(physical_devices[0], 'GPU')  # Use the first GPU
#         print("Using GPU: ", physical_devices[0])
#     except RuntimeError as e:
#         print(f"Error setting GPU device: {e}")
# else:
#     print("No GPU found, running on CPU.")

# # Dataset parameters
# dataset_url = '2750'
# batch_size = 32
# img_height = 64
# img_width = 64
# validation_split = 0.2
# rescale = 1.0 / 255

# # Data loading and preprocessing
# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     validation_split=validation_split,
#     rescale=rescale
# )

# train_dataset = datagen.flow_from_directory(
#     directory=dataset_url,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     subset='training',
#     shuffle=True,
#     class_mode='categorical'
# )

# test_dataset = datagen.flow_from_directory(
#     directory=dataset_url,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     subset='validation',
#     shuffle=True,
#     class_mode='categorical'
# )

# # Display sample images
# class_names = list(train_dataset.class_indices.keys())
# sample_images, sample_labels = next(train_dataset)
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(sample_images[i])
#     plt.title(class_names[sample_labels[i].argmax()])
#     plt.axis("off")
# plt.show()

# # Identity block
# def identity_block(X, f, filters, training=True, initializer=random_uniform):
#     F1, F2, F3 = filters
#     X_shortcut = X
#     X = Conv2D(F1, (1, 1), kernel_initializer=initializer(seed=0))(X)
#     X = BatchNormalization(axis=3)(X, training=training)
#     X = Activation('relu')(X)

#     X = Conv2D(F2, (f, f), padding='same', kernel_initializer=initializer(seed=0))(X)
#     X = BatchNormalization(axis=3)(X, training=training)
#     X = Activation('relu')(X)

#     X = Conv2D(F3, (1, 1), kernel_initializer=initializer(seed=0))(X)
#     X = BatchNormalization(axis=3)(X, training=training)

#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
#     return X

# # Convolutional block
# def convolutional_block(X, f, filters, s=2, training=True, initializer=glorot_uniform):
#     F1, F2, F3 = filters
#     X_shortcut = X

#     X = Conv2D(F1, (1, 1), strides=(s, s), kernel_initializer=initializer(seed=0))(X)
#     X = BatchNormalization(axis=3)(X, training=training)
#     X = Activation('relu')(X)

#     X = Conv2D(F2, (f, f), padding='same', kernel_initializer=initializer(seed=0))(X)
#     X = BatchNormalization(axis=3)(X, training=training)
#     X = Activation('relu')(X)

#     X = Conv2D(F3, (1, 1), kernel_initializer=initializer(seed=0))(X)
#     X = BatchNormalization(axis=3)(X, training=training)

#     X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), kernel_initializer=initializer(seed=0))(X_shortcut)
#     X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)

#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
#     return X

# # ResNet50 model definition
# def ResNet50(input_shape=(64, 64, 3), classes=10):
#     X_input = Input(input_shape)
#     X = ZeroPadding2D((3, 3))(X_input)

#     X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3)(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((3, 3), strides=(2, 2))(X)

#     X = convolutional_block(X, 3, [64, 64, 256], s=1)
#     X = identity_block(X, 3, [64, 64, 256])
#     X = identity_block(X, 3, [64, 64, 256])

#     X = convolutional_block(X, 3, [128, 128, 512], s=2)
#     X = identity_block(X, 3, [128, 128, 512])
#     X = identity_block(X, 3, [128, 128, 512])
#     X = identity_block(X, 3, [128, 128, 512])

#     X = convolutional_block(X, 3, [256, 256, 1024], s=2)
#     X = identity_block(X, 3, [256, 256, 1024])
#     X = identity_block(X, 3, [256, 256, 1024])
#     X = identity_block(X, 3, [256, 256, 1024])
#     X = identity_block(X, 3, [256, 256, 1024])
#     X = identity_block(X, 3, [256, 256, 1024])

#     X = convolutional_block(X, 3, [512, 512, 2048], s=2)
#     X = identity_block(X, 3, [512, 512, 2048])
#     X = identity_block(X, 3, [512, 512, 2048])

#     X = AveragePooling2D(pool_size=(2, 2))(X)
#     X = Flatten()(X)
#     X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

#     model = Model(inputs=X_input, outputs=X)
#     return model

# # Build and compile model
# model = ResNet50(input_shape=(64, 64, 3), classes=len(class_names))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Specify device context for GPU usage
# with tf.device('/GPU:0'):  # Use the first GPU
#     history = model.fit(train_dataset, validation_data=test_dataset, epochs=20)

# # Save model
# model.save("resnet50_model.h5")
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform, random_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Found {len(physical_devices)} GPU(s):")
    for device in physical_devices:
        print(device)
    # Set memory growth to prevent TensorFlow from consuming all GPU memory
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Using GPU.")
else:
    print("No GPU found. Using CPU.")

# Dataset parameters
dataset_path = '2750'  # Your dataset path
img_dir = os.path.join('multi_label_data')  # Assuming images are in an 'images' subfolder
labels_csv = os.path.join( 'labels.csv')  # Path to your multi-label CSV
batch_size = 32
img_height = 64
img_width = 64
validation_split = 0.2

# Custom multi-label dataset generator
class MultiLabelDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_dir, labels_df, batch_size=32, img_size=(64, 64), 
                 augment=False, shuffle=True, subset=None, validation_split=0.0):
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        
        # Process dataframe
        self.labels_df = labels_df
        self.num_classes = len([col for col in labels_df.columns if col != 'image'])
        self.class_names = [col for col in labels_df.columns if col != 'image']
        
        # Handle train/validation split
        if subset and validation_split > 0:
            # Get indices for train or validation
            n = len(labels_df)
            indices = np.arange(n)
            np.random.shuffle(indices)
            split_idx = int(n * (1 - validation_split))
            
            if subset == 'training':
                self.indices = indices[:split_idx]
            else:  # validation
                self.indices = indices[split_idx:]
        else:
            self.indices = np.arange(len(labels_df))
            
        self.augment = augment
        self.on_epoch_end()
        
        print(f"[INFO] Created {'training' if subset=='training' else 'validation'} generator with {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_indices), self.img_size[0], self.img_size[1], 3))
        batch_y = np.zeros((len(batch_indices), self.num_classes))
        
        for i, idx in enumerate(batch_indices):
            # Load image
            img_path = os.path.join(self.img_dir, self.labels_df.iloc[idx]['image'])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Apply augmentation if needed
            if self.augment:
                img_array = self._augment_image(img_array)
            
            # Normalize
            img_array = img_array / 255.0
            batch_x[i] = img_array
            
            # Get labels (all columns except 'image')
            labels = self.labels_df.iloc[idx].drop('image').values
            batch_y[i] = labels
            
        return batch_x, batch_y
    
    def _augment_image(self, img):
        # Simple augmentation
        if np.random.random() > 0.5:
            img = tf.image.flip_left_right(img).numpy()
        if np.random.random() > 0.5:
            img = tf.image.flip_up_down(img).numpy()
        return img
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Load labels CSV
try:
    # Try to load the CSV assuming it has columns: 'image', 'class1', 'class2', etc.
    # Where class columns contain 0 or 1 indicating absence/presence of that class
    labels_df = pd.read_csv(labels_csv)
    print(f"Loaded {len(labels_df)} entries from {labels_csv}")
except FileNotFoundError:
    # If the file doesn't exist, show a message about expected format
    print(f"Error: Could not find {labels_csv}")
    print("Please ensure you have a CSV file with the following format:")
    print("image,forest,pasture,residential,water,...)  # Headers")
    print("img1.jpg,1,0,1,0,...)  # 1 means class is present, 0 means absent")
    print("img2.jpg,0,1,0,1,...)") 
    raise
    
# Check if the CSV has the expected format
if 'image' not in labels_df.columns:
    print("Error: CSV must have an 'image' column with image filenames")
    raise ValueError("Invalid CSV format")

# Create train and validation generators
train_generator = MultiLabelDataGenerator(
    img_dir=img_dir,
    labels_df=labels_df,
    batch_size=batch_size,
    img_size=(img_height, img_width),
    augment=True,
    shuffle=True,
    subset='training',
    validation_split=validation_split
)

val_generator = MultiLabelDataGenerator(
    img_dir=img_dir,
    labels_df=labels_df,
    batch_size=batch_size,
    img_size=(img_height, img_width),
    augment=False,
    shuffle=False,
    subset='validation',
    validation_split=validation_split
)

# Get class names and number of classes
class_names = train_generator.class_names
num_classes = train_generator.num_classes
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Display sample images with multi-labels
sample_images, sample_labels = train_generator[0]
plt.figure(figsize=(12, 12))
for i in range(min(9, len(sample_images))):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i])
    
    # Create label text showing all present classes
    present_classes = [class_names[j] for j in range(num_classes) if sample_labels[i][j] > 0.5]
    label_text = ", ".join(present_classes)
    plt.title(label_text, fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.show()

# Identity block
def identity_block(X, f, filters, training=True, initializer=random_uniform):
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1, 1), kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f, f), padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

# Convolutional block
def convolutional_block(X, f, filters, s=2, training=True, initializer=glorot_uniform):
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides=(s, s), kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f, f), padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)

    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X

# ResNet50 model definition - modified for multi-label classification
def ResNet50_MultiLabel(input_shape=(64, 64, 3), classes=10):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, 3, [64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, 3, [128, 128, 512], s=2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    X = convolutional_block(X, 3, [256, 256, 1024], s=2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, 3, [512, 512, 2048], s=2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = AveragePooling2D(pool_size=(2, 2))(X)
    X = Flatten()(X)
    X = Dropout(0.5)(X)  # Add dropout to prevent overfitting
    
    # Output layer for multi-label classification with sigmoid activation
    # Each neuron outputs a probability for each class independently
    X = Dense(classes, activation='sigmoid', kernel_initializer=glorot_uniform(seed=0), name='fc' + str(classes))(X)

    model = Model(inputs=X_input, outputs=X)
    return model

# Build model with dynamic class count from dataset
model = ResNet50_MultiLabel(input_shape=(img_height, img_width, 3), classes=num_classes)

# Compile model with appropriate optimizer and loss function for multi-label classification
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Binary cross-entropy for multi-label
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]  # AUC is useful for multi-label
)

# Print model summary
model.summary()

# Define callbacks
checkpoint = ModelCheckpoint(
    'resnet50_multilabel_best.h5',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=10,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train the model
with tf.device('/GPU:0' if len(physical_devices) > 0 else '/CPU:0'):
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.title('AUC')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# Save model
model.save("resnet50_multilabel.h5")
print("Model saved as resnet50_multilabel.h5")

# Evaluate model and show classification report
from sklearn.metrics import classification_report, multilabel_confusion_matrix

# Get predictions on validation set
all_labels = []
all_preds = []

for i in range(len(val_generator)):
    x, y = val_generator[i]
    preds = model.predict(x)
    all_labels.extend(y)
    all_preds.extend(preds)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Convert probabilities to binary predictions
threshold = 0.5
binary_preds = (all_preds > threshold).astype(int)

# Print classification report
print("\nClassification Report (threshold=0.5):")
print(classification_report(all_labels, binary_preds, target_names=class_names))

# Function to predict on new images
def predict_image(model, image_path, class_names):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)[0]
    
    # Get classes with probability > threshold
    threshold = 0.5
    detected_classes = [(class_names[i], predictions[i]) for i in range(len(class_names)) if predictions[i] > threshold]
    
    # Sort by probability in descending order
    detected_classes.sort(key=lambda x: x[1], reverse=True)
    
    return img, detected_classes

# Example usage (uncomment to use):
"""
test_image_path = "path/to/test_image.jpg"
img, detected = predict_image(model, test_image_path, class_names)

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title("Detected classes: " + ", ".join([f"{cls} ({prob:.2f})" for cls, prob in detected]))
plt.axis("off")
plt.show()
"""
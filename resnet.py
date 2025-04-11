# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms
# from PIL import Image
# import pandas as pd
# import os

# # Custom Dataset
# class MultiLabelEuroSAT(Dataset):
#     def __init__(self, img_dir, label_csv, transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         self.df = pd.read_csv(label_csv)
#         self.num_classes = 10

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.df.iloc[idx]["image"])
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)

#         label_indices = eval(self.df.iloc[idx]["labels"])
#         labels = torch.zeros(self.num_classes)
#         labels[label_indices] = 1.0
#         return image, labels

# # Transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # Dataset and DataLoader
# dataset = MultiLabelEuroSAT("multi_label_data", "labels.csv", transform)
# loader = DataLoader(dataset, batch_size=16, shuffle=True)

# # Model
# model = models.resnet18(pretrained=True)
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 10),
#     nn.Sigmoid()  # for multi-label
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Training setup
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Training loop
# for epoch in range(5):
#     model.train()
#     total_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in loader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         preds = outputs > 0.5
#         correct += (preds == labels.bool()).all(dim=1).sum().item()
#         total += labels.size(0)

#     acc = correct / total * 100
#     print(f"✅ Epoch {epoch+1}/5 - Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# # Save model
# torch.save(model.state_dict(), "resnet_eurosat_multilabel1.pth")
# print("💾 Model saved as resnet_eurosat_multilabel.pth")






# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import Dataset, DataLoader
# # from torchvision import models, transforms
# # from PIL import Image
# # import pandas as pd
# # import os

# # # ---------------------
# # # Custom Dataset
# # # ---------------------
# # class MultiLabelEuroSAT(Dataset):
# #     def __init__(self, img_dir, label_csv, transform=None):
# #         self.img_dir = img_dir  # Dataset directory: "multi_label_data"
# #         self.transform = transform
# #         self.df = pd.read_csv(label_csv)  # CSV file: "labels.csv"
# #         self.num_classes = 10
# #         print(f"[INFO] Loaded {len(self.df)} entries from {label_csv}")

# #     def __len__(self):
# #         return len(self.df)

# #     def __getitem__(self, idx):
# #         img_path = os.path.join(self.img_dir, self.df.iloc[idx]["image"])
# #         image = Image.open(img_path).convert("RGB")
# #         if self.transform:
# #             image = self.transform(image)

# #         label_indices = eval(self.df.iloc[idx]["labels"])  # e.g., "[0, 3, 7]"
# #         labels = torch.zeros(self.num_classes)
# #         labels[label_indices] = 1.0
# #         return image, labels

# # # ---------------------
# # # Transforms, Dataset, and DataLoader
# # # ---------------------
# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                          std=[0.229, 0.224, 0.225])
# # ])

# # dataset = MultiLabelEuroSAT("multi_label_data", "labels.csv", transform)
# # loader = DataLoader(dataset, batch_size=16, shuffle=True)
# # print(f"[INFO] Dataset loaded: {len(dataset)} samples, DataLoader created.")

# # # ---------------------
# # # Model Setup
# # # ---------------------
# # # Using ResNet18 pretrained on ImageNet
# # model = models.resnet18(pretrained=True)
# # model.fc = nn.Sequential(
# #     nn.Linear(model.fc.in_features, 10),
# #     nn.Sigmoid()  # For multi-label output
# # )
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = model.to(device)
# # print(f"[INFO] Model loaded on device: {device}")
# # print(model)  # Print model architecture for verification

# # # ---------------------
# # # Training Setup
# # # ---------------------
# # criterion = nn.BCELoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # # ---------------------
# # # Training Loop
# # # ---------------------
# # num_epochs = 5
# # for epoch in range(num_epochs):
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
# #         preds = outputs > 0.5  # Using 0.5 as threshold for each class
# #         correct += (preds == labels.bool()).all(dim=1).sum().item()
# #         total += labels.size(0)

# #     acc = correct / total * 100
# #     print(f"✅ Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

# # # ---------------------
# # # Save the Model
# # # ---------------------
# # torch.save(model.state_dict(), "resnet_eurosat_multilabel2.pth")
# # print("💾 Model saved as resnet_eurosat_multilabel1.pth")
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform, random_uniform
import matplotlib.pyplot as plt

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

# Optionally, set the device to use the first available GPU
if len(physical_devices) > 0:
    try:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')  # Use the first GPU
        print("Using GPU: ", physical_devices[0])
    except RuntimeError as e:
        print(f"Error setting GPU device: {e}")
else:
    print("No GPU found, running on CPU.")

# Dataset parameters
dataset_url = '2750'
batch_size = 32
img_height = 64
img_width = 64
validation_split = 0.2
rescale = 1.0 / 255

# Data loading and preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=validation_split,
    rescale=rescale
)

train_dataset = datagen.flow_from_directory(
    directory=dataset_url,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training',
    shuffle=True,
    class_mode='categorical'
)

test_dataset = datagen.flow_from_directory(
    directory=dataset_url,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation',
    shuffle=True,
    class_mode='categorical'
)

# Display sample images
class_names = list(train_dataset.class_indices.keys())
sample_images, sample_labels = next(train_dataset)
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i])
    plt.title(class_names[sample_labels[i].argmax()])
    plt.axis("off")
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

# ResNet50 model definition
def ResNet50(input_shape=(64, 64, 3), classes=10):
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
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X)
    return model

# Build and compile model
model = ResNet50(input_shape=(64, 64, 3), classes=len(class_names))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Specify device context for GPU usage
with tf.device('/GPU:0'):  # Use the first GPU
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=20)

# Save model
model.save("resnet50_model.h5")

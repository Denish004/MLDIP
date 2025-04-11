import os
import random
import shutil
from PIL import Image
from torchvision import transforms
import pandas as pd

# Define input and output paths
input_root = "2750"  # Original EuroSAT dataset
output_dir = "multi_label_data"
os.makedirs(output_dir, exist_ok=True)

# Define classes and mapping
classes = sorted(os.listdir(input_root))
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

# Define image transform (resize to 224x224)
resize = transforms.Resize((224, 224))

# Generate synthetic mixed-label images
rows = []
num_samples = 500

for i in range(num_samples):
    selected_classes = random.sample(classes, k=random.randint(2, 4))  # select 2-4 classes
    images = []

    for cls in selected_classes:
        cls_path = os.path.join(input_root, cls)
        img_name = random.choice(os.listdir(cls_path))
        img_path = os.path.join(cls_path, img_name)
        img = Image.open(img_path).convert("RGB")
        images.append(resize(img))

    # Create a 2x2 grid image
    width, height = images[0].size
    new_img = Image.new("RGB", (width * 2, height * 2))
    for idx, img in enumerate(images[:4]):
        x = (idx % 2) * width
        y = (idx // 2) * height
        new_img.paste(img, (x, y))

    # Save the mixed image
    out_img_name = f"mixed_{i:03d}.jpg"
    out_img_path = os.path.join(output_dir, out_img_name)
    new_img.save(out_img_path)

    # Store labels
    label_indices = [class_to_idx[cls] for cls in selected_classes]
    rows.append([out_img_name, label_indices])

# Save labels to CSV
df = pd.DataFrame(rows, columns=["image", "labels"])
csv_path = "labels.csv"
df.to_csv(csv_path, index=False)

output_dir, csv_path, len(df)  # Return info for confirmation

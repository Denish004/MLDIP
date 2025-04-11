import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import platform

# ===========================================
# Modern GPU Configuration
# ===========================================
def setup_device():
    """Configure and return the best available device (GPU or CPU) with detailed diagnostics"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🔍 Found {gpu_count} GPU(s):")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # in GB
            print(f"  GPU {i}: {gpu_name} (Compute {gpu_capability[0]}.{gpu_capability[1]}, {total_memory:.2f} GB)")
        
        # Set the device to the first GPU
        device = torch.device("cuda:0")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Print CUDA version and other GPU info
        print(f"📊 CUDA Version: {torch.version.cuda}")
        print(f"📊 PyTorch CUDA: {torch.version.cuda}")
        print(f"📊 cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        print("📊 cuDNN Benchmark: Enabled")
        
        return device
    else:
        # Try MPS (Metal Performance Shaders) for Mac with M1/M2 chips
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✅ No CUDA GPU found, but MPS is available. Using Apple Silicon GPU.")
            return device
        
        print("⚠️ No GPU found. Using CPU.")
        return torch.device("cpu")

# ===========================================
# Dataset Configuration with Modern Transforms
# ===========================================
class DatasetManager:
    def __init__(self, dataset_path, img_size=224, batch_size=32, val_split=0.2, num_workers=None):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        
        # Auto-configure optimal number of workers based on CPU cores
        if num_workers is None:
            if platform.system() == 'Windows':
                self.num_workers = 0  # Default to 0 for Windows to avoid issues
            else:
                self.num_workers = min(os.cpu_count(), 8)  # Use up to 8 workers on non-Windows
        else:
            self.num_workers = num_workers
            
        print(f"🔄 DataLoader workers: {self.num_workers}")
        
        # Modern data transforms with better augmentation
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # Set up datasets
        self._setup_datasets()
    
    def _setup_datasets(self):
        """Initialize and split datasets"""
        print(f"📂 Loading dataset from: {self.dataset_path}")
        
        try:
            # Full dataset with training transforms
            full_dataset = ImageFolder(root=self.dataset_path, transform=self.data_transforms['train'])
            
            # Get dataset size and calculate split
            dataset_size = len(full_dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(self.val_split * dataset_size))
            
            # Shuffle indices for random split
            np.random.seed(42)
            np.random.shuffle(indices)
            
            # Split indices
            train_indices, val_indices = indices[split:], indices[:split]
            
            # Create datasets with appropriate transforms
            self.train_dataset = ImageFolder(root=self.dataset_path, transform=self.data_transforms['train'])
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices)
            
            self.val_dataset = ImageFolder(root=self.dataset_path, transform=self.data_transforms['val'])
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices)
            
            # Save class names
            self.classes = full_dataset.classes
            self.num_classes = len(self.classes)
            
            print(f"✅ Dataset loaded: {len(self.train_dataset)} training samples, {len(self.val_dataset)} validation samples")
            print(f"📊 Classes: {self.classes}")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def get_dataloaders(self, device):
        """Create and return data loaders optimized for the device"""
        # Configure DataLoader parameters
        loader_args = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': device.type != 'cpu',  # Pin memory only for GPU
        }
        
        # Create DataLoaders
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            **loader_args
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            **loader_args
        )
        
        return train_loader, val_loader

# ===========================================
# Modern ResNet50 with GPU Optimizations
# ===========================================
def create_model(num_classes, device, use_pretrained=False):
    """Create and configure a ResNet50 model with the latest optimizations"""
    if use_pretrained:
        print("🔄 Loading pretrained ResNet50 model")
        # Use the latest pretrained model from torchvision
        model = models.resnet50(weights='IMAGENET1K_V2')
        # Replace final layer to match our number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        print("🔄 Creating custom ResNet50 model (without pretraining)")
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Use DataParallel if multiple GPUs are available
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    return model

# ===========================================
# Modern Training with Mixed Precision
# ===========================================
def train_model(model, train_loader, val_loader, device, num_classes, num_epochs=20, 
                learning_rate=0.001, save_dir='models'):
    """Train model with mixed precision, early stopping, and other modern optimizations"""
    # Create directory for saving models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup loss, optimizer and schedulers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate schedulers - using both step and plateau-based reduction
    # OneCycleLR is good for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, 
        steps_per_epoch=len(train_loader), 
        epochs=num_epochs
    )
    
    # Initialize the GradScaler for mixed precision training (faster on modern GPUs)
    scaler = GradScaler() if device.type == 'cuda' else None
    use_mixed_precision = device.type == 'cuda' and scaler is not None
    
    if use_mixed_precision:
        print("🚀 Using mixed precision training")
    
    # Track best model and metrics
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 5  # Early stopping patience
    
    # Initialize training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    total_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 30)
        
        # ----------------
        # Training Phase
        # ----------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if available
            if use_mixed_precision:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward and optimize with scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Update LR if using OneCycleLR
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Print batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {train_loss/(batch_idx+1):.4f} | "
                      f"Acc: {100.*train_correct/train_total:.2f}% | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # ----------------
        # Validation Phase
        # ----------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass (no mixed precision needed for validation)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {100.*train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {100.*val_acc:.2f}%")
        print(f"Time: {epoch_time:.1f}s")
        
        # Check if this is the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            save_path = os.path.join(save_dir, 'resnet50_best.pth')
            
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
                
            print(f"✅ New best model saved! Val Acc: {100.*val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{max_patience}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    final_path = os.path.join(save_dir, 'resnet50_final.pth')
    
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), final_path)
    else:
        torch.save(model.state_dict(), final_path)
    
    print(f"💾 Final model saved to {final_path}")
    
    # Calculate total training time
    total_time = time.time() - total_start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"🏁 Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"🏆 Best validation accuracy: {100.*best_val_acc:.2f}%")
    
    return model, history

# ===========================================
# Plot Training Results
# ===========================================
def plot_training_history(history, save_path='results'):
    """Plot training history with modern styling"""
    os.makedirs(save_path, exist_ok=True)
    
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot accuracy
    axs[0].plot(history['train_acc'], 'o-', label='Training')
    axs[0].plot(history['val_acc'], 'o-', label='Validation')
    axs[0].set_title('Model Accuracy', fontsize=16)
    axs[0].set_ylabel('Accuracy', fontsize=14)
    axs[0].set_xlabel('Epoch', fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].grid(True)
    
    # Add best validation accuracy as text
    best_val_idx = np.argmax(history['val_acc'])
    best_val_acc = history['val_acc'][best_val_idx]
    axs[0].annotate(f'Best: {best_val_acc:.4f}',
                  xy=(best_val_idx, best_val_acc),
                  xytext=(best_val_idx, best_val_acc - 0.1),
                  arrowprops=dict(facecolor='black', shrink=0.05),
                  fontsize=12)
    
    # Plot loss
    axs[1].plot(history['train_loss'], 'o-', label='Training')
    axs[1].plot(history['val_loss'], 'o-', label='Validation')
    axs[1].set_title('Model Loss', fontsize=16)
    axs[1].set_ylabel('Loss', fontsize=14)
    axs[1].set_xlabel('Epoch', fontsize=14)
    axs[1].legend(fontsize=12)
    axs[1].grid(True)
    
    # Add learning rate as text
    plt.figtext(0.5, 0.01, f"Initial LR: {history['lr'][0]:.6f}, Final LR: {history['lr'][-1]:.6f}",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300)
    print(f"📊 Training history saved to {os.path.join(save_path, 'training_history.png')}")
    plt.close()

# ===========================================
# Main Function
# ===========================================
def main():
    # Configuration
    dataset_path = '2750'  # Change to your dataset path
    img_size = 224  # Modern models typically use 224x224 or higher
    batch_size = 32  # Adjust based on your GPU memory
    num_epochs = 20  # Adjust as needed
    
    # 1. Setup device (GPU/CPU)
    device = setup_device()
    
    try:
        # 2. Setup data with modern transforms
        data_manager = DatasetManager(
            dataset_path=dataset_path,
            img_size=img_size,
            batch_size=batch_size,
            val_split=0.2
        )
        
        # 3. Get data loaders
        train_loader, val_loader = data_manager.get_dataloaders(device)
        
        # 4. Create model
        model = create_model(
            num_classes=data_manager.num_classes,
            device=device,
            use_pretrained=True  # Set to False to train from scratch
        )
        
        # 5. Train model with modern optimizations
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_classes=data_manager.num_classes,
            num_epochs=num_epochs,
            learning_rate=0.001,
            save_dir='models'
        )
        
        # 6. Plot and save training history
        plot_training_history(history, save_path='results')
        
        # 7. Print summary
        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETE! 🎉")
        print("="*50)
        print(f"📊 Dataset: {dataset_path}")
        print(f"📊 Classes: {data_manager.classes}")
        print(f"📊 Training samples: {len(data_manager.train_dataset)}")
        print(f"📊 Validation samples: {len(data_manager.val_dataset)}")
        print(f"💾 Models saved in: models/")
        print(f"📈 Results saved in: results/")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️ TROUBLESHOOTING TIPS:")
        print("1. Verify dataset path exists and contains image folders")
        print("2. Check GPU memory if you get CUDA out of memory errors")
        print("3. Reduce batch size if necessary")
        print("4. Ensure dataset follows the correct structure:")
        print("   dataset_path/")
        print("   ├── class1/")
        print("   │   ├── img1.jpg")
        print("   │   └── img2.jpg")
        print("   └── class2/")
        print("       ├── img3.jpg")
        print("       └── img4.jpg")

if __name__ == "__main__":
    # Required for Windows compatibility
    import multiprocessing
    if platform.system() == "Windows":
        multiprocessing.freeze_support()
    
    # Run the main function
    main()
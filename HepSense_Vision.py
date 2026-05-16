

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.autograd import Function
from torchvision import datasets, transforms, models
from PIL import Image


try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("[WARN] pytorch-grad-cam not installed. Grad-CAM XAI will be disabled.")
    print("       Install with: pip install grad-cam")

DATA_DIR        = "./Dataset"
IMG_SIZE        = 224
BATCH_SIZE      = 32
NUM_EPOCHS      = 25
LR              = 1e-4
NUM_CLASSES     = 5          # F0, F1, F2, F3, F4
NUM_DOMAINS     = 2          # source (train) vs target (val) -- pseudo-domains
FOCAL_GAMMA     = 2.0
MODEL_SAVE_PATH = "hepsense_vision_dann_v2.pth"
CLASS_NAMES     = ["F0", "F1", "F2", "F3", "F4"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transforms -- training with augmentation, validation without
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_dataloaders(data_dir: str, val_split: float = 0.2):
    """
    Build train/val DataLoaders from an ImageFolder directory.
    Uses WeightedRandomSampler on the training split to counteract
    the heavy F0/F4 imbalance (2114/1698 vs 793–861 for F1-F3).
    """
    full_dataset = datasets.ImageFolder(data_dir)
    total       = len(full_dataset)
    val_size    = int(total * val_split)
    train_size  = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply transforms via wrapper datasets
    train_dataset = _TransformSubset(train_dataset, train_transforms)
    val_dataset   = _TransformSubset(val_dataset, val_transforms)

    # --- Weighted Random Sampler for training ---
    train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    weights_per_class = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [weights_per_class[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"[DATA] Train: {train_size} | Val: {val_size}")
    print(f"[DATA] Class distribution (train): {dict(sorted(class_counts.items()))}")
    return train_loader, val_loader, full_dataset.classes


class _TransformSubset(torch.utils.data.Dataset):
    """Thin wrapper to apply transforms to a random_split Subset."""

    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
        # Expose underlying indices for the sampler
        self.indices   = subset.indices

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class _GradientReversalFunction(Function):
    """Flips the gradient sign during backpropagation."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps the GRL function with a schedulable lambda parameter."""

    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambda_val)


class HepSenseDANN(nn.Module):
    """
    Domain-Adversarial Neural Network for fibrosis staging.

    Components:
        - feature_extractor : EfficientNet-B0 (pretrained, final FC removed)
        - stage_classifier  : F0–F4 classification head
        - domain_classifier : Binary domain head behind a GRL

    The GRL forces the feature extractor to produce representations that are
    INVARIANT to the imaging device / acquisition protocol, directly fixing
    the cross-domain generalization failure reported in Joo et al. 2023.
    """

    def __init__(self, num_classes: int = 5, num_domains: int = 2):
        super().__init__()

        # --- Backbone: EfficientNet-B0 ---
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        feat_dim = backbone.classifier[1].in_features   # 1280
        backbone.classifier = nn.Identity()              # strip original head
        self.feature_extractor = backbone

        # --- Stage classifier (the clinical output) ---
        self.stage_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        # --- Domain classifier (adversarial head) ---
        self.grl = GradientReversalLayer(lambda_val=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_domains),
        )

    def forward(self, x, alpha: float = 1.0):
        """
        Args:
            x:     input images  [B, 3, 224, 224]
            alpha: GRL lambda schedule value (ramps 0->1 during training)
        Returns:
            stage_logits:  [B, num_classes]
            domain_logits: [B, num_domains]
        """
        self.grl.lambda_val = alpha
        features = self.feature_extractor(x)          # [B, 1280]
        stage_logits  = self.stage_classifier(features)
        domain_logits = self.domain_classifier(self.grl(features))
        return stage_logits, domain_logits




class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for multi-class classification.

    FL(p_t) = -alpha_t (1 - p_t)^gamma  log(p_t)

    This down-weights the loss for well-classified examples (majority F0/F4)
    and focuses training on the hard-to-classify minority stages (F1-F3).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha      # per-class weight tensor
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss




def _grl_lambda_schedule(epoch: int, total_epochs: int) -> float:
    """Gradual ramp from 0 -> 1 over training (DANN paper schedule)."""
    p = epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def train_model(data_dir: str = DATA_DIR, num_epochs: int = NUM_EPOCHS):
    """Full training procedure for the HepSense DANN vision model."""

    print("=" * 60)
    print("  HepSense DANN Vision -- Training Pipeline")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {num_epochs}")
    print(f"  Batch  : {BATCH_SIZE}")
    print(f"  LR     : {LR}")
    print("=" * 60)

    train_loader, val_loader, class_names = build_dataloaders(data_dir)

    class_counts = torch.tensor([2114, 861, 793, 857, 1698], dtype=torch.float32)
    alpha_weights = (1.0 / class_counts)
    alpha_weights = alpha_weights / alpha_weights.sum() * NUM_CLASSES  # normalize
    alpha_weights = alpha_weights.to(DEVICE)

    # --- Model, Loss, Optimizer ---
    model = HepSenseDANN(num_classes=NUM_CLASSES, num_domains=NUM_DOMAINS).to(DEVICE)
    stage_criterion  = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha_weights)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        alpha = _grl_lambda_schedule(epoch, num_epochs)

        # ---------- TRAIN ----------
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Pseudo-domain labels: all training images = domain 0
            domain_labels = torch.zeros(images.size(0), dtype=torch.long, device=DEVICE)

            stage_logits, domain_logits = model(images, alpha=alpha)

            loss_stage  = stage_criterion(stage_logits, labels)
            loss_domain = domain_criterion(domain_logits, domain_labels)
            loss = loss_stage + 0.3 * loss_domain  # weighted sum

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (stage_logits.argmax(1) == labels).sum().item()
            total   += images.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ---------- VALIDATE ----------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                stage_logits, _ = model(images, alpha=alpha)
                val_correct += (stage_logits.argmax(1) == labels).sum().item()
                val_total   += images.size(0)
        val_acc = val_correct / val_total

        scheduler.step()

        print(
            f"Epoch [{epoch+1:02d}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Acc: {val_acc:.4f}  alpha(GRL): {alpha:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Restore best and save
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[SAVED] Best model (Val Acc={best_val_acc:.4f}) -> {MODEL_SAVE_PATH}")
    return model



def load_trained_model(path: str = MODEL_SAVE_PATH) -> HepSenseDANN:
    """Load a trained HepSenseDANN from disk."""
    model = HepSenseDANN(num_classes=NUM_CLASSES, num_domains=NUM_DOMAINS)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print(f"[MODEL] Loaded from {path} on {DEVICE}")
    return model


def _preprocess_image(image_path: str) -> tuple:
    """
    Load and preprocess a single image for inference.
    Returns: (input_tensor [1,3,224,224], rgb_numpy [H,W,3] 0-1 float)
    """
    img = Image.open(image_path).convert("RGB")
    rgb_np = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    transform = val_transforms
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    return input_tensor, rgb_np.astype(np.float32)


def generate_gradcam(model: HepSenseDANN, input_tensor: torch.Tensor,
                     rgb_image: np.ndarray, target_class: int) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the predicted fibrosis stage.

    Args:
        model:        Trained HepSenseDANN
        input_tensor: Preprocessed image tensor [1,3,224,224]
        rgb_image:    Original image as float32 numpy [H,W,3] in [0,1]
        target_class: Predicted class index

    Returns:
        cam_overlay:  RGB numpy array [H,W,3] with heatmap overlay
    """
    if not GRADCAM_AVAILABLE:
        print("[WARN] Grad-CAM unavailable. Returning original image.")
        return (rgb_image * 255).astype(np.uint8)

    # Wrap the model so GradCAM only sees stage_classifier output
    class _StageWrapper(nn.Module):
        def __init__(self, dann_model):
            super().__init__()
            self.model = dann_model
        def forward(self, x):
            stage_logits, _ = self.model(x, alpha=0.0)
            return stage_logits

    wrapper = _StageWrapper(model).to(DEVICE).eval()

    # Target the last convolutional block of EfficientNet-B0
    target_layers = [model.feature_extractor.features[-1]]

    cam = GradCAM(model=wrapper, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    cam_overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return cam_overlay


def predict_ultrasound(image_path: str, model: HepSenseDANN) -> dict:
    """
    Run full inference on a single ultrasound image.

    Args:
        image_path: Path to the ultrasound image file.
        model:      Trained HepSenseDANN instance.

    Returns:
        dict with keys:
            - "predicted_stage"   : str   ("F0" .. "F4")
            - "confidence"        : float (softmax probability)
            - "probabilities"     : dict  {stage: prob}
            - "gradcam_overlay"   : np.ndarray [H,W,3] uint8
    """
    input_tensor, rgb_np = _preprocess_image(image_path)

    with torch.no_grad():
        stage_logits, _ = model(input_tensor, alpha=0.0)
        probs = torch.softmax(stage_logits, dim=1).cpu().numpy()[0]

    predicted_idx = int(np.argmax(probs))
    predicted_stage = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])

    # Grad-CAM visualization
    cam_overlay = generate_gradcam(model, input_tensor, rgb_np, predicted_idx)

    return {
        "predicted_stage":  predicted_stage,
        "confidence":       confidence,
        "probabilities":    {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        "gradcam_overlay":  cam_overlay,
    }



if __name__ == "__main__":
    trained_model = train_model()

    # Quick sanity check on the first image in F0
    test_img_dir = os.path.join(DATA_DIR, "F0")
    test_images  = os.listdir(test_img_dir)
    if test_images:
        test_path = os.path.join(test_img_dir, test_images[0])
        result = predict_ultrasound(test_path, trained_model)
        print(f"\n[TEST] {test_path}")
        print(f"  Predicted : {result['predicted_stage']} ({result['confidence']:.2%})")
        print(f"  All probs : {result['probabilities']}")

        # Save Grad-CAM visualization
        if result["gradcam_overlay"] is not None:
            plt.figure(figsize=(6, 6))
            plt.imshow(result["gradcam_overlay"])
            plt.title(f"Grad-CAM: {result['predicted_stage']} ({result['confidence']:.1%})")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("gradcam_sample.png", dpi=150)
            plt.close()
            print("  Grad-CAM saved -> gradcam_sample.png")
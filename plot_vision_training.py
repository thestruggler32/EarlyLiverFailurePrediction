import matplotlib.pyplot as plt
import re

# Data extracted from logs
logs = """Epoch [01/35]  Train Loss: 0.8423  Train Acc: 0.4058  Val Acc: 0.5182  alpha(GRL): 0.000
Epoch [02/35]  Train Loss: 0.6474  Train Acc: 0.5064  Val Acc: 0.5475  alpha(GRL): 0.142
Epoch [03/35]  Train Loss: 0.5522  Train Acc: 0.5671  Val Acc: 0.6353  alpha(GRL): 0.278
Epoch [04/35]  Train Loss: 0.4744  Train Acc: 0.6217  Val Acc: 0.7009  alpha(GRL): 0.404
Epoch [05/35]  Train Loss: 0.3804  Train Acc: 0.6871  Val Acc: 0.7437  alpha(GRL): 0.516
Epoch [06/35]  Train Loss: 0.3346  Train Acc: 0.7233  Val Acc: 0.6582  alpha(GRL): 0.613
Epoch [07/35]  Train Loss: 0.2993  Train Acc: 0.7438  Val Acc: 0.8188  alpha(GRL): 0.695
Epoch [08/35]  Train Loss: 0.2463  Train Acc: 0.7940  Val Acc: 0.8236  alpha(GRL): 0.762
Epoch [09/35]  Train Loss: 0.1920  Train Acc: 0.8377  Val Acc: 0.8703  alpha(GRL): 0.815
Epoch [10/35]  Train Loss: 0.1619  Train Acc: 0.8537  Val Acc: 0.8861  alpha(GRL): 0.858
Epoch [11/35]  Train Loss: 0.1456  Train Acc: 0.8626  Val Acc: 0.8758  alpha(GRL): 0.891
Epoch [12/35]  Train Loss: 0.1164  Train Acc: 0.8927  Val Acc: 0.9248  alpha(GRL): 0.917
Epoch [13/35]  Train Loss: 0.0881  Train Acc: 0.9118  Val Acc: 0.9367  alpha(GRL): 0.937
Epoch [14/35]  Train Loss: 0.0930  Train Acc: 0.9114  Val Acc: 0.9090  alpha(GRL): 0.952
Epoch [15/35]  Train Loss: 0.0804  Train Acc: 0.9144  Val Acc: 0.9407  alpha(GRL): 0.964
Epoch [16/35]  Train Loss: 0.0753  Train Acc: 0.9255  Val Acc: 0.9375  alpha(GRL): 0.973
Epoch [17/35]  Train Loss: 0.0670  Train Acc: 0.9332  Val Acc: 0.9470  alpha(GRL): 0.980
Epoch [18/35]  Train Loss: 0.0480  Train Acc: 0.9500  Val Acc: 0.9407  alpha(GRL): 0.985
Epoch [19/35]  Train Loss: 0.0664  Train Acc: 0.9334  Val Acc: 0.9589  alpha(GRL): 0.988
Epoch [20/35]  Train Loss: 0.0425  Train Acc: 0.9559  Val Acc: 0.9644  alpha(GRL): 0.991
Epoch [21/35]  Train Loss: 0.0325  Train Acc: 0.9658  Val Acc: 0.9834  alpha(GRL): 0.993
Epoch [22/35]  Train Loss: 0.0315  Train Acc: 0.9694  Val Acc: 0.9747  alpha(GRL): 0.995
Epoch [23/35]  Train Loss: 0.0231  Train Acc: 0.9719  Val Acc: 0.9786  alpha(GRL): 0.996
Epoch [24/35]  Train Loss: 0.0258  Train Acc: 0.9703  Val Acc: 0.9794  alpha(GRL): 0.997
Epoch [25/35]  Train Loss: 0.0458  Train Acc: 0.9541  Val Acc: 0.9747  alpha(GRL): 0.998
Epoch [26/35]  Train Loss: 0.0269  Train Acc: 0.9694  Val Acc: 0.9810  alpha(GRL): 0.998
Epoch [27/35]  Train Loss: 0.0212  Train Acc: 0.9741  Val Acc: 0.9834  alpha(GRL): 0.999
Epoch [28/35]  Train Loss: 0.0216  Train Acc: 0.9719  Val Acc: 0.9834  alpha(GRL): 0.999
Epoch [29/35]  Train Loss: 0.0158  Train Acc: 0.9810  Val Acc: 0.9818  alpha(GRL): 0.999
Epoch [30/35]  Train Loss: 0.0168  Train Acc: 0.9781  Val Acc: 0.9834  alpha(GRL): 0.999
Epoch [31/35]  Train Loss: 0.0139  Train Acc: 0.9800  Val Acc: 0.9842  alpha(GRL): 1.000
Epoch [32/35]  Train Loss: 0.0127  Train Acc: 0.9842  Val Acc: 0.9866  alpha(GRL): 1.000
Epoch [33/35]  Train Loss: 0.0152  Train Acc: 0.9783  Val Acc: 0.9802  alpha(GRL): 1.000
Epoch [34/35]  Train Loss: 0.0123  Train Acc: 0.9818  Val Acc: 0.9858  alpha(GRL): 1.000
Epoch [35/35]  Train Loss: 0.0145  Train Acc: 0.9814  Val Acc: 0.9834  alpha(GRL): 1.000"""

epochs = []
train_loss = []
train_acc = []
val_acc = []

for line in logs.strip().split('\n'):
    m = re.search(r'Epoch \[(\d+)/35\]\s+Train Loss: ([\d.]+)\s+Train Acc: ([\d.]+)\s+Val Acc: ([\d.]+)', line)
    if m:
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        train_acc.append(float(m.group(3)))
        val_acc.append(float(m.group(4)))

# Set a dark theme for a premium clinical feel
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0f172a')
fig.suptitle('HepSense Vision Pipeline (DenseNet121 + DANN)', fontsize=20, fontweight='bold', color='#e2e8f0', y=1.05)

# Plot 1: Accuracy
ax1.set_facecolor('#1e293b')
ax1.plot(epochs, train_acc, color='#38bdf8', linewidth=2.5, marker='o', label='Train Accuracy')
ax1.plot(epochs, val_acc, color='#34d399', linewidth=2.5, marker='s', label='Validation Accuracy')
ax1.set_title('Model Accuracy vs Epochs', color='#f8fafc', fontsize=16)
ax1.set_xlabel('Epoch', color='#94a3b8', fontsize=12)
ax1.set_ylabel('Accuracy', color='#94a3b8', fontsize=12)
ax1.grid(True, color='#334155', linestyle='--', alpha=0.7)
ax1.legend(loc='lower right', frameon=True, facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')
ax1.tick_params(colors='#94a3b8')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#334155')
ax1.spines['left'].set_color('#334155')

# Plot 2: Loss
ax2.set_facecolor('#1e293b')
ax2.plot(epochs, train_loss, color='#f43f5e', linewidth=2.5, marker='^', label='Train Loss')
ax2.set_title('Cross Entropy Loss', color='#f8fafc', fontsize=16)
ax2.set_xlabel('Epoch', color='#94a3b8', fontsize=12)
ax2.set_ylabel('Loss', color='#94a3b8', fontsize=12)
ax2.grid(True, color='#334155', linestyle='--', alpha=0.7)
ax2.legend(loc='upper right', frameon=True, facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')
ax2.tick_params(colors='#94a3b8')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#334155')
ax2.spines['left'].set_color('#334155')

plt.tight_layout()
plt.savefig('vision_training_curves.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved vision_training_curves.png")

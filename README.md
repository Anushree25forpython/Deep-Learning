🧠 Project: CIFAR-10 Image Classification using CNN (PyTorch)

📌 Overview
- This project implements a Convolutional Neural Network (CNN) in PyTorch to classify images from the CIFAR-10 dataset. The model aims to accurately predict the category of images into one of 10 classes, using a deep learning pipeline with training, evaluation, and optimization components.

---

🎯 Objective
- Build and train a CNN model on the CIFAR-10 training set and evaluate its accuracy on the test set. The goal is to maximize model performance, especially accuracy, while understanding the training dynamics.

---

📁 Dataset: CIFAR-10
- Total Images: 60,000 (32x32 color images)
  - Training Set: 50,000 images
  - Test Set: 10,000 images
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Format: Image files (not tensors by default)
- Accessed using: torchvision.datasets.CIFAR10()
- Transformed using: transforms.Compose with normalization and tensor conversion

---

🔄 Data Loading Pipeline

- Used PyTorch’s Dataset and DataLoader abstractions:
  - Converts images to tensors
  - Enables batching and shuffling
- Separate data loaders for training and testing sets
- Batches and normalizes the data to speed up convergence

---

🧱 Model Architecture

➤ Backbone: 5 Custom Blocks (Nth_Block)
- Each block includes:
  - A linear layer producing a weight vector a
  - 3 convolutional layers followed by ReLU activations
  - Weighted sum of conv outputs using parts of a: a1 * conv1 + a2 * conv2 + a3 * conv3
  - BatchNorm and MaxPooling applied after each block 
➤ Classifier (MLP)
- Fully Connected Layers with:
    - ReLU activations
    - Dropout regularization
  - Ends with a Linear layer producing logits for 10 classes

---

⚙️ Training Configuration

- Loss Function: nn.CrossEntropyLoss (combines Softmax + NLL)
- Optimizer: torch.optim.Adam
   - Learning Rate: 0.001
   - Weight Decay: 0.0001
   - Epochs: 30

---

🚀 Training Procedure
  
  Implemented via trainf() function:
   - Trains over multiple epochs on GPU (cuda) or CPU
   - Tracks:
       - Training Loss
       - Training Accuracy
       - Validation Accuracy
  - Visualization using a custom Animator:
       - Loss curve
       - Accuracy curves (train & validation)
       - Displayed live during training
  - Execution time tracked using a Timer utility

---

📊 Results

Metric	Score
 - Training Accuracy	~88.9%
 - Validation Accuracy	~83.5%
 - Final Training Loss	~0.319

   - Training & validation accuracy improved steadily without overfitting
   - Model generalizes well to unseen test data

---

🧠 Key Learnings & Optimization Strategies

- Backbone-style CNNs with learnable combination of conv layers enhance representation
- Adaptive average pooling helped unify feature dimensions for MLP
- Normalization & dropout played key roles in improving generalization
- Further improvements possible via:
          - Advanced architectures (ResNet, EfficientNet)
          - Data augmentation
          - Learning rate scheduling

---

🛠️ Tools & Libraries

`PyTorch`
`torchvision`
`NumPy`
`Matplotlib`
`Google Colab` (or Jupyter Notebook)

# ReducedMNIST-Neural-Network-Evaluation-in-Python

# ReducedMNIST Classification Project

## Overview

This project builds and evaluates feedforward and convolutional neural networks on the **ReducedMNIST** dataset (1,000 training and 200 testing images per digit). Three main tasks are covered:

1. **MLP (FFNN)** with 1, 3, and 5 hidden layers using flattened 28×28 inputs.
2. **CNN** trained directly on 28×28 grayscale images.
3. **Hyperparameter Variations**: at least two architectural or training changes, with performance analysis.

Finally, results (accuracy, train time, test time) are compared against those from Assignment 1.

---

## Dataset

* **Training**: 10,000 total images (1,000/class)
* **Testing**: 2,000 total images (200/class)
* Grayscale, normalized to \[0, 1]

## Dependencies

```bash
Python 3.8+
pip install torch torchvision numpy
```

## 1. Multilayer Perceptron (MLP)

### Architecture Variants

* **1 hidden layer**: `[784 → H → 10]`
* **3 hidden layers**: `[784 → H → H → H → 10]`
* **5 hidden layers**: `[784 → H ×5 → 10]`

> *H is a chosen hidden dimension (e.g. 256).*

### Training

* Loss: CrossEntropy
* Optimizer: Adam (lr=0.001)
* Epochs: 10
* Batch size: 64

```bash
python train_mlp.py --layers 1 --hidden_dim 256
python train_mlp.py --layers 3 --hidden_dim 256
python train_mlp.py --layers 5 --hidden_dim 256
```

## 2. Convolutional Neural Networks (CNN)

### Base Model (LeNet‑Style)

1. Conv(1→6, 5×5) → ReLU → Pool(2×2)
2. Conv(6→16, 5×5) → ReLU → Pool(2×2)
3. FC(16×4×4→120) → ReLU
4. FC(120→84) → ReLU
5. FC(84→10)

```bash
python train_cnn.py --variant base
```

## 3. Hyperparameter Variations

Implemented variants (see `train_cnn.py`):

* **Var1**: Increased filters (16→32) in Conv layers.
* **Var3**: Added dropout after conv & FC layers.
* **Var4**: Added extra conv block + LeakyReLU + dropout.
* **Var5**: Added BatchNorm after conv layers.

Run:

```bash
python train_cnn.py --variant var1
python train_cnn.py --variant var3
python train_cnn.py --variant var4
python train_cnn.py --variant var5
```

## Results Comparison

	Baseline model	            Accuracy	  Training time	  Testing time

  Batch Size = 32,   Epochs = 5	  96.7 %	   25.28 sec	     0.77 sec
	Batch Size = 32,   Epochs = 10	 97.25 %   53.16 sec	     0.76 sec
	Batch Size = 16,   Epochs = 10	 97.8 %	   62 sec	         0.94 sec
	Variations (batch size = 16, epochs = 10)	Accuracy	Training time	Testing time
	Variation 1:	Increased Number of Filters	98.6 %	68 sec	1.04 sec
	Variation 2:	Addition of a Third Convolutional Layer	97.45 %	62.02 sec	0.89 sec
	Variation 3:	Incorporation of Dropout	97.7 %	59.4 sec	0.95 sec
	Variation 4:	Addition of Batch Normalization	97.9%	65.08 sec	0.96 sec


## Customization

* Adjust hidden dimensions (`--hidden_dim`), learning rate, batch size.
* Swap activation functions (ReLU, LeakyReLU, Sigmoid).
* Modify dropout rates or normalization layers.


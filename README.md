# HowDeepLearningWorks

> A from-scratch Deep Learning implementation in C# and .NET 8 that exposes the mathematics behind learning instead of hiding it behind a ready-made ML framework.

## Overview

`HowDeepLearningWorks` is an educational neural-network implementation built from the mathematical foundations upward.

The project focuses on one question:

**What actually happens to the weights when a neural network learns?**

Instead of starting with a high-level API such as `Fit()` or `Predict()`, the implementation builds the underlying mechanics directly in C#:

- Linear algebra
- Activation functions and derivatives
- Forward propagation
- Loss calculation
- Backpropagation
- Gradient calculation
- Gradient descent
- Training
- Test-set evaluation
- Prediction and accuracy

The project is intentionally small. Its purpose is to make the relationship between mathematics, algorithms, and software implementation easy to follow.

## Network Architecture

The current demonstration network is:

```text
Input: 4 features
        │
        ▼
   Dense 4 → 8
     ReLU
        │
        ▼
   Dense 8 → 8
     ReLU
        │
        ▼
   Dense 8 → 6
     ReLU
        │
        ▼
   Dense 6 → 4
     ReLU
        │
        ▼
   Dense 4 → 1
    Sigmoid
        │
        ▼
   Binary prediction
```

## Learning Pipeline

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Binary Cross Entropy
  ↓
Backpropagation
  ↓
Gradients (dW, db, dx)
  ↓
Gradient Descent
  ↓
Updated Weights / Biases
  ↓
Next Training Step
```

For a dense layer:

```text
z  = W x + b
a  = activation(z)
```

During backpropagation:

```text
dW = dz · xᵀ
db = dz
dx = Wᵀ · dz
```

Parameter updates use gradient descent:

```text
W ← W - η dW
b ← b - η db
```

## Verification

The project validates the implementation at several levels.

### Mathematical Operations

Vector and matrix operations are tested, including:

- Addition and subtraction
- Scalar multiplication
- Dot product
- Matrix × vector
- Matrix × matrix
- Transpose

### Activation Functions

Implemented and tested:

- ReLU
- Sigmoid
- Tanh

Both function values and derivatives are covered.

### Backpropagation

The dense layer and multi-layer network calculate:

```text
dW
db
dx
```

and propagate gradients backward through the network.

### Numerical Gradient Checking

Analytical gradients are compared with numerical gradients using finite differences.

The current verification checks **172 weights** across the five-layer network.

### Training

The training loop is verified by measuring the loss before and after training.

### Train / Test Evaluation

The demonstration uses separate training and test samples and reports predictions and classification accuracy on unseen test samples.

The current demonstration reaches **100% accuracy on its small deterministic test set**. This result is only intended to verify the implementation pipeline; it is not a claim of real-world model performance.

## Why Build It From Scratch?

High-level ML libraries are useful because they hide implementation details and make production systems easier to build.

This project has a different purpose: expose those details.

The implementation deliberately does **not** depend on:

- ML.NET
- TensorFlow.NET
- TorchSharp
- Accord.NET
- Other ready-made Deep Learning frameworks

The mathematics is implemented directly in C#.

## Project Structure

```text
HowDeepLearningWorks/
├── src/
│   └── HowDeepLearningWorks/
│       ├── Mathematics/
│       │   ├── Vector.cs
│       │   └── Matrix.cs
│       ├── ActivationFunctions/
│       │   ├── IActivationFunction.cs
│       │   ├── ReLU.cs
│       │   ├── Sigmoid.cs
│       │   └── Tanh.cs
│       ├── LossFunctions/
│       │   └── BinaryCrossEntropy.cs
│       └── NeuralNetworks/
│           ├── DenseLayer.cs
│           └── NeuralNetwork.cs
│
├── examples/
│   └── HowDeepLearningWorks.Console/
├── tests/
│   └── HowDeepLearningWorks.Tests/
├── docs/
│   └── architecture/
│       └── adr/
└── .github/
    └── workflows/
```

## Running the Demonstration

From the repository root:

```powershell
dotnet restore examples/HowDeepLearningWorks.Console/HowDeepLearningWorks.Console.csproj

dotnet build examples/HowDeepLearningWorks.Console/HowDeepLearningWorks.Console.csproj --configuration Release

dotnet run --project examples/HowDeepLearningWorks.Console/HowDeepLearningWorks.Console.csproj --configuration Release
```

The console demonstration runs the mathematical and neural-network checks and then performs training and test-set evaluation.

## Design Principles

### Mathematics First

The implementation starts with vectors and matrices rather than a high-level neural-network abstraction.

### Explicit Learning Mechanics

Forward propagation, loss, gradients, backpropagation, and parameter updates remain visible in the code.

### Small Architecture

Abstractions are introduced only where they represent a real variation point, such as activation functions.

### Verification Before Expansion

Each major capability is tested before the project moves to the next layer of complexity.

## Current Status

The first educational implementation is complete for the intended scope:

- [x] Mathematical core
- [x] Activation functions
- [x] Dense layer
- [x] Forward propagation
- [x] Backpropagation
- [x] Numerical gradient checking
- [x] Gradient descent
- [x] Training loop
- [x] Train/test evaluation
- [x] Prediction
- [x] Accuracy calculation

The project intentionally stops here rather than turning into a general-purpose Deep Learning framework.

## What This Demonstrates

The value of this project is not the number of machine-learning APIs it contains.

It demonstrates the complete path from mathematical reasoning to executable software:

```text
Mathematics
    ↓
Algorithm
    ↓
Implementation
    ↓
Numerical Verification
    ↓
Training
    ↓
Evaluation
```

That connection is the core of `HowDeepLearningWorks`.

## License

MIT License

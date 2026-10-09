# HowDeepLearningWorks

[![CI](https://github.com/peymanpro/HowDeepLearningWorks/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/peymanpro/HowDeepLearningWorks/actions/workflows/ci.yml)
[![.NET](https://img.shields.io/badge/.NET-8.0-512BD4)](https://dotnet.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Understand how neural networks learn by implementing the mathematics in C#—without hiding the mechanics behind a machine-learning framework.**

**HowDeepLearningWorks** is a small, from-scratch educational implementation of a dense neural network. It makes the path from linear algebra to training explicit: forward propagation, loss, backpropagation, numerical gradient checking, parameter updates, and evaluation.

The goal is understanding and verification, not competing with production libraries.

## What is implemented?

- **Linear algebra:** vectors, matrices, vector dot products, matrix multiplication, and transpose.
- **Activation functions:** ReLU, Sigmoid, and Tanh, including derivatives.
- **Dense layers:** \(z = Wx + b\), optional element-wise activation, and gradients for weights, biases, and inputs.
- **Learning:** Binary Cross-Entropy, backpropagation, and gradient-descent parameter updates.
- **Verification:** automated xUnit tests, finite-difference gradient checks, and an executable training/evaluation demonstration.
- **Evaluation:** predictions and accuracy on a small held-out synthetic classification set.

The core library has no dependency on ML.NET, TensorFlow, PyTorch, TorchSharp, or another ready-made deep-learning framework.

## Network in the demonstration

The sample uses a sequential fully connected network for binary classification:

    4 input features
          |
          v
     Dense(4 -> 8) + ReLU
          |
          v
     Dense(8 -> 8) + ReLU
          |
          v
     Dense(8 -> 6) + ReLU
          |
          v
     Dense(6 -> 4) + ReLU
          |
          v
     Dense(4 -> 1) + Sigmoid
          |
          v
     Probability in [0, 1] -> binary class

The sample deliberately keeps the architecture small so the calculations remain traceable in a debugger.

## How learning works

For one input vector \(x \in \mathbb{R}^{n}\), a dense layer with \(m\) outputs computes:

\[
z = Wx + b, \qquad a = \phi(z)
\]

where \(W \in \mathbb{R}^{m \times n}\), \(b \in \mathbb{R}^{m}\), and \(\phi\) is an activation function.

Given the gradient arriving from the next operation, backpropagation applies the chain rule:

\[
\delta = \frac{\partial L}{\partial a} \odot \phi'(z)
\]

\[
\frac{\partial L}{\partial W} = \delta x^T, \qquad
\frac{\partial L}{\partial b} = \delta, \qquad
\frac{\partial L}{\partial x} = W^T\delta
\]

Gradient descent then updates each parameter:

\[
W \leftarrow W - \eta\frac{\partial L}{\partial W}, \qquad
b \leftarrow b - \eta\frac{\partial L}{\partial b}
\]

The implementation exposes these calculations directly in **DenseLayer**. See [Mathematical Foundations](docs/MATHEMATICAL-FOUNDATIONS.md) for dimensional analysis, the loss derivative, and finite-difference gradient checking.

## Verification

The test suite covers:

- vector and matrix operations, including incompatible dimensions;
- activation-function values and derivatives;
- Binary Cross-Entropy values, derivatives, and invalid input handling;
- dense-layer forward propagation and weight, bias, and input gradients;
- network layer-shape validation and backward-pass preconditions;
- analytical weight **and bias** gradients compared with central finite differences;
- a gradient-descent step that reduces the loss for a simple example.

The console demonstration also performs an end-to-end training run and evaluates a held-out synthetic test set. The test set is intentionally small and simple: **100% accuracy on it is only a pipeline sanity check, not evidence of generalization or real-world model quality.**

GitHub Actions restores and builds the solution, runs the automated test project, and executes the console demonstration.

## Run locally

Requires the [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).

From the repository root:

    dotnet restore HowDeepLearningWorks.sln
    dotnet build HowDeepLearningWorks.sln --configuration Release --no-restore
    dotnet test tests/HowDeepLearningWorks.Tests/HowDeepLearningWorks.Tests.csproj --configuration Release --no-build --no-restore
    dotnet run --project examples/HowDeepLearningWorks.Console/HowDeepLearningWorks.Console.csproj --configuration Release --no-build

The console output reports the mathematical checks, training-loss change, predictions, and test accuracy. A failed check exits with an exception and a non-zero process result.

## Repository layout

    .
    ├── src/HowDeepLearningWorks/
    │   ├── Mathematics/          # Vector and Matrix
    │   ├── ActivationFunctions/  # ReLU, Sigmoid, Tanh
    │   ├── LossFunctions/        # Binary Cross-Entropy
    │   └── NeuralNetworks/       # DenseLayer and NeuralNetwork
    ├── tests/HowDeepLearningWorks.Tests/
    │   ├── MathematicsTests.cs
    │   └── NeuralNetworkLearningTests.cs
    ├── examples/HowDeepLearningWorks.Console/
    │   └── Program.cs            # Executable training and verification demo
    ├── docs/
    │   ├── MATHEMATICAL-FOUNDATIONS.md
    │   └── architecture/adr/
    └── .github/workflows/ci.yml

## Scope and limitations

This project is intentionally a **learning implementation**, not a general-purpose or production-ready neural-network framework.

- It currently supports sequential fully connected layers and one sample at a time.
- Its matrix operations use straightforward managed C# loops rather than optimized numerical kernels.
- The training demonstration uses gradient descent; optimizers such as Adam, batching, model serialization, GPU execution, and production deployment are outside the current scope.
- The synthetic dataset is designed to make the training pipeline easy to inspect, not to serve as a meaningful benchmark.

These constraints keep attention on the most important learning sequence:

    Mathematics -> Algorithm -> Implementation -> Numerical checks -> Training -> Evaluation

## License

MIT. See [LICENSE](LICENSE).

# HowDeepLearningWorks v1.0.0

## Overview

The first stable educational release of HowDeepLearningWorks: a from-scratch dense neural-network implementation in C# and .NET 8 that makes the mathematics of learning visible.

## Included

- Vector and matrix operations implemented in managed C#.
- ReLU, Sigmoid, and Tanh activation functions and derivatives.
- Dense-layer forward propagation and backpropagation for weight, bias, and input gradients.
- Binary Cross-Entropy loss and gradient-descent updates.
- A sequential five-layer binary-classification demonstration.
- Automated xUnit coverage for linear algebra, activation functions, loss calculations, network validation, gradient descent, and finite-difference gradient checks for weights and biases.
- A mathematical guide covering tensor shapes, the chain rule, Binary Cross-Entropy with Sigmoid, and numerical gradient checking.
- GitHub Actions CI that restores and builds the full solution, runs the xUnit suite, and executes the console demonstration.

## Verification

The release workflow is gated on a successful restore/build, xUnit run, and console demonstration. At the release candidate commit, GitHub Actions completed successfully:

- 16 automated tests passed; 0 failed and 0 skipped.
- The console gradient check passed for all 172 weights in the demonstration network.
- The end-to-end console training and test evaluation completed successfully.

The held-out dataset contains only four synthetic samples. Its 100% accuracy is a pipeline sanity check, not a benchmark or evidence of generalization.

## Known limitations

This is a learning project rather than a production ML framework. It processes one example at a time, uses straightforward managed C# matrix loops, and currently demonstrates gradient descent only. Mini-batches, Adam, GPU execution, serialization, and production deployment are outside this release's scope.

## Requirements and license

- .NET 8 SDK
- MIT License

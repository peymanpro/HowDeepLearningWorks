# HowDeepLearningWorks v1.0.1

## Summary

Documentation-only patch release following v1.0.0.

## Changes

- Use GitHub-supported dollar delimiters for inline and display mathematics in the README and Mathematical Foundations guide.
- Keep the derivations for dense-layer gradients, Binary Cross-Entropy with Sigmoid, gradient descent, and finite-difference gradient checking readable in GitHub Markdown.

No executable code or model behavior changed from v1.0.0.
The release tag points to the source snapshot containing these Markdown rendering corrections.

## Verification

The source-code test suite remains unchanged: 16 automated test cases cover the numerical operations, activation and loss functions, backpropagation, weight and bias gradients, and a gradient-descent update. CI also runs the console gradient check for all 172 weights in the demonstration network and executes the end-to-end synthetic training/evaluation example.

The synthetic hold-out set contains four samples. Its 100% accuracy is a pipeline sanity check, not a benchmark.

## Requirements and license

- .NET 8 SDK
- MIT License

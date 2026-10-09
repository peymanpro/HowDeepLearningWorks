# HowDeepLearningWorks v1.1.0

## Summary

This release improves reproducibility and training behavior while keeping the project focused on understanding neural networks from first principles.

## What's new

- **Seeded initialization:** call `network.InitializeWeights(seed: 42)` to reproduce the initial weights on the same .NET runtime.
- **He normal for ReLU:** samples weights from a zero-mean normal distribution with variance `2 / fanIn`, generated using the Box–Muller transform.
- **Xavier/Glorot uniform for other layers:** samples weights from `[-sqrt(6 / (fanIn + fanOut)), +sqrt(6 / (fanIn + fanOut))]`.
- **Deterministic console example:** the demonstration now reports the seed, hyperparameters, initial/final training loss, individual test predictions, and test-set accuracy without embedding a long collection of manual assertions.
- **Expanded automated tests:** coverage verifies seed reproducibility, He variance scale, Xavier bounds, initializer selection by activation, unchanged zero biases, and error handling.
- **Mathematical documentation:** the initialization distributions and rationale are documented alongside the existing backpropagation and gradient-checking derivations.

## Verification

The GitHub Actions workflow restores and builds the solution, runs the xUnit suite, and executes the console demo.

The release candidate passed **22 automated test cases** with no failures or skips. The console demonstration completed successfully with seed 42, reducing training loss from approximately 0.708178 to 0.000857. It classified all four synthetic held-out samples correctly; this is a small pipeline sanity check, not a benchmark or evidence of real-world generalization.

## Known limitations

This remains an educational implementation. Training uses one sample at a time and basic gradient descent. It does not provide batching, Adam, GPU execution, serialization, optimized numerical kernels, or production deployment facilities.

## Requirements and license

- .NET 8 SDK
- MIT License

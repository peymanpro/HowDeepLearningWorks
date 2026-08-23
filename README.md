@'
# HowDeepLearningWorks

A from-scratch Deep Learning implementation in C# and .NET.

## Goal

The purpose of this project is to understand and implement the core mechanics of a neural network without using ready-made Machine Learning or Deep Learning frameworks.

The first version focuses on one concrete network:

```text
Input
  |
  v
Hidden Layer 1
  |
  v
Hidden Layer 2
  |
  v
Hidden Layer 3
  |
  v
Hidden Layer 4
  |
  v
Output

The network will support:

Forward propagation
Activation functions
Loss calculation
Backpropagation
Gradient calculation
Weight and bias updates
Training
Test data evaluation
Accuracy calculation
Prediction
Inspection of learned weights
Network

The initial neural network topology is:

4 → 8 → 8 → 6 → 4 → 1

The hidden layers use ReLU and the output layer uses Sigmoid for binary classification.

Learning

The training process is intentionally explicit:

Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Weight / Bias Update
  ↓
Next Epoch

The weight update follows gradient descent:

W_new = W_old - learningRate × gradient

Project Structure
HowDeepLearningWorks/
│
├── src/
│   └── HowDeepLearningWorks/
│
├── examples/
│   └── HowDeepLearningWorks.Console/
│
├── tests/
│   └── HowDeepLearningWorks.Tests/
│
├── docs/
│   └── architecture/
│       └── adr/
│
└── .github/
    └── workflows/
Development Phases
Phase 0

Project foundation and architecture.

Phase 1

Mathematical core.

Phase 2

Neuron and dense layer.

Phase 3

Activation functions.

Phase 4

Forward propagation.

Phase 5

Loss and backpropagation.

Phase 6

Gradient descent and parameter updates.

Phase 7

Training engine.

Phase 8

Testing, evaluation and prediction.

Phase 9

Experiments and result visualization.

Phase 10

Documentation and GitHub release.

Constraints

The implementation does not use:

ML.NET
TensorFlow.NET
TorchSharp
Accord.NET
other ready-made Machine Learning / Deep Learning frameworks

The neural-network mathematics is implemented directly in C#.

Current Status

Phase 0 is being completed.

No neural-network implementation has been added yet.

License

MIT
'@ | Set-Content -Encoding UTF8 README.md


ADR:

```powershell id="ztfwgh"
@'
# ADR-001: Initial Project Architecture

## Status

Accepted

## Context

The purpose of HowDeepLearningWorks is to demonstrate how a neural network works internally rather than to build a general-purpose Deep Learning framework.

The first version therefore needs a small and understandable architecture.

## Decision

Use a small three-project structure:

```text
HowDeepLearningWorks
        ^
        |
HowDeepLearningWorks.Console

HowDeepLearningWorks.Tests
        |
        v
HowDeepLearningWorks

The Core project contains the neural-network implementation.

The Console project contains executable experiments.

The Test project contains correctness tests.

Initial Network

The first network will use four hidden layers:

4 → 8 → 8 → 6 → 4 → 1

Hidden layers use ReLU.

The output layer uses Sigmoid.

Learning Algorithm

The first implementation will use:

Forward propagation
Binary Cross Entropy
Backpropagation
Gradient descent

Only one optimization algorithm is planned for the initial version.

Additional optimizers are intentionally outside the first version.

Design Patterns

Patterns will only be used where they provide real value.

An activation-function abstraction is justified because the network must support different activation functions.

A strategy-style abstraction may be introduced for activation functions or future optimization algorithms when the implementation reaches the relevant phase.

No generic framework architecture will be introduced merely to increase abstraction.

Consequences

The project remains small enough to understand while still demonstrating the real mechanics of Deep Learning.

The architecture can be expanded later if the actual implementation requires it.
'@ | Set-Content -Encoding UTF8 docs/architecture/adr/ADR-001-initial-architecture.md


License:

```powershell id="go6gqy"
@'
MIT License

Copyright (c) 2026 Peyman Salimi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'@ | Set-Content -Encoding UTF8 LICENSE

CI:

@'
name: CI

on:
  push:
    branches:
      - main
      - master
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: 8.0.x

      - name: Restore Core
        run: dotnet restore src/HowDeepLearningWorks/HowDeepLearningWorks.csproj

      - name: Restore Console
        run: dotnet restore examples/HowDeepLearningWorks.Console/HowDeepLearningWorks.Console.csproj

      - name: Build Core
        run: dotnet build src/HowDeepLearningWorks/HowDeepLearningWorks.csproj --configuration Release --no-restore

      - name: Build Console
        run: dotnet build examples/HowDeepLearningWorks.Console/HowDeepLearningWorks.Console.csproj --configuration Release --no-restore
'@ | Set-Content -Encoding UTF8 .github/workflows/ci.yml
# ADR-001: Initial Project Architecture

## Status

Accepted

## Context

HowDeepLearningWorks is designed to demonstrate the internal mechanics of a neural network through a small and understandable implementation in C# and .NET.

The first version focuses on a single feed-forward neural network with four hidden layers, explicit forward propagation, backpropagation, gradient calculation, parameter updates, training, and evaluation.

## Decision

Use a deliberately small three-project structure:

```text
HowDeepLearningWorks
├── Core
├── Console
└── Tests
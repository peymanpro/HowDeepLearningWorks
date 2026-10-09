# Migrating to SharpNeuralNetwork

The current source tree uses the `SharpNeuralNetwork` library identity:

- Solution: `SharpNeuralNetwork.sln`
- Core project: `src/SharpNeuralNetwork/SharpNeuralNetwork.csproj`
- Core namespace root: `SharpNeuralNetwork`
- Console example: `examples/SharpNeuralNetwork.Console`
- Test project: `tests/SharpNeuralNetwork.Tests`

## Compatibility note

This rename changes namespace identifiers, assembly names, and project paths. Code or project files that reference the earlier `HowDeepLearningWorks.*` namespaces or old project paths must be updated to the new names. This is a source- and binary-breaking rename, so it should be released as a new major version rather than silently treated as a patch.

The mathematical operations, forward pass, backpropagation, loss calculation, initializers, and test intent are unchanged by the naming refactor. The existing v1.x release records are historical snapshots and remain associated with the code and identity under which they were published.

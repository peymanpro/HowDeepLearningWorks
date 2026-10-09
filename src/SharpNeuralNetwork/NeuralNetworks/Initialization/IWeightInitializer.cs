using SharpNeuralNetwork.Mathematics;

namespace SharpNeuralNetwork.NeuralNetworks.Initialization;

/// <summary>
/// Defines a strategy for initializing a dense layer's weight matrix.
/// </summary>
public interface IWeightInitializer
{
    /// <summary>
    /// Initializes the supplied matrix using the provided random number generator.
    /// </summary>
    /// <param name="weights">The matrix to initialize. Rows are output units and columns are input units.</param>
    /// <param name="random">The random number generator used to produce reproducible values.</param>
    void Initialize(Matrix weights, Random random);
}

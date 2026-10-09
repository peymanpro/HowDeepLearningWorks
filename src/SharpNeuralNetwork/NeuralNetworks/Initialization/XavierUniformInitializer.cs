using SharpNeuralNetwork.Mathematics;

namespace SharpNeuralNetwork.NeuralNetworks.Initialization;

/// <summary>
/// Initializes weights uniformly in [-limit, limit], where
/// limit = sqrt(6 / (fan-in + fan-out)).
/// This is the Xavier/Glorot uniform initialization.
/// </summary>
public sealed class XavierUniformInitializer : IWeightInitializer
{
    /// <inheritdoc />
    public void Initialize(Matrix weights, Random random)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(random);

        if (weights.Rows == 0 || weights.Columns == 0)
        {
            throw new ArgumentException(
                "The weight matrix must have at least one row and one column.",
                nameof(weights));
        }

        var limit = Math.Sqrt(6.0 / (weights.Columns + weights.Rows));

        for (var row = 0; row < weights.Rows; row++)
        {
            for (var column = 0; column < weights.Columns; column++)
            {
                weights[row, column] = ((2.0 * random.NextDouble()) - 1.0) * limit;
            }
        }
    }
}

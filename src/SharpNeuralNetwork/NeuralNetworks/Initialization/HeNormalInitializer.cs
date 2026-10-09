using SharpNeuralNetwork.Mathematics;

namespace SharpNeuralNetwork.NeuralNetworks.Initialization;

/// <summary>
/// Initializes weights from a zero-mean normal distribution with variance 2 / fan-in.
/// This initialization is commonly used with ReLU hidden layers.
/// </summary>
public sealed class HeNormalInitializer : IWeightInitializer
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

        var standardDeviation = Math.Sqrt(2.0 / weights.Columns);

        for (var row = 0; row < weights.Rows; row++)
        {
            for (var column = 0; column < weights.Columns; column++)
            {
                // Box-Muller transform: standard normal sample, scaled by sqrt(2 / fan-in).
                var u1 = 1.0 - random.NextDouble();
                var u2 = random.NextDouble();
                var standardNormal =
                    Math.Sqrt(-2.0 * Math.Log(u1)) *
                    Math.Cos(2.0 * Math.PI * u2);

                weights[row, column] = standardNormal * standardDeviation;
            }
        }
    }
}

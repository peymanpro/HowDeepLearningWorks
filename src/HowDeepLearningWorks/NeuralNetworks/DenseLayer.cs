using HowDeepLearningWorks.Mathematics;

namespace HowDeepLearningWorks.NeuralNetworks;

/// <summary>
/// Represents a fully connected neural network layer.
/// </summary>
public sealed class DenseLayer
{
    private readonly Matrix _weights;
    private readonly Vector _bias;

    /// <summary>
    /// Creates a fully connected layer.
    /// </summary>
    /// <param name="inputSize">Number of input neurons.</param>
    /// <param name="outputSize">Number of output neurons.</param>
    public DenseLayer(int inputSize, int outputSize)
    {
        if (inputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputSize));
        }

        if (outputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputSize));
        }

        _weights = new Matrix(new double[outputSize, inputSize]);
        _bias = new Vector(new double[outputSize]);
    }

    /// <summary>
    /// Gets the layer weights.
    /// </summary>
    public Matrix Weights => _weights;

    /// <summary>
    /// Gets the layer bias.
    /// </summary>
    public Vector Bias => _bias;

    /// <summary>
    /// Computes the linear forward pass:
    /// z = Wx + b
    /// </summary>
    public Vector Forward(Vector input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.Length != _weights.Columns)
        {
            throw new ArgumentException(
                $"Expected input size {_weights.Columns}, but received {input.Length}.",
                nameof(input));
        }

        return (_weights * input) + _bias;
    }
}
using HowDeepLearningWorks.Mathematics;

namespace HowDeepLearningWorks.NeuralNetworks;

/// <summary>
/// Represents a fully connected neural network layer.
/// </summary>
public sealed class DenseLayer
{
    private readonly Matrix _weights;
    private readonly Vector _bias;

    private Vector? _lastInput;

    /// <summary>
    /// Creates a fully connected layer.
    /// </summary>
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

        WeightGradients = new Matrix(new double[outputSize, inputSize]);
        BiasGradients = new Vector(new double[outputSize]);
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
    /// Gets the gradients of the weights.
    /// </summary>
    public Matrix WeightGradients { get; }

    /// <summary>
    /// Gets the gradients of the bias.
    /// </summary>
    public Vector BiasGradients { get; }

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

        var inputCopy = new double[input.Length];

        for (var i = 0; i < input.Length; i++)
        {
            inputCopy[i] = input[i];
        }

        _lastInput = new Vector(inputCopy);

        return (_weights * input) + _bias;
    }

    /// <summary>
    /// Computes gradients for the layer using the gradient
    /// received from the next layer.
    ///
    /// For z = Wx + b:
    /// dW = dz * x^T
    /// db = dz
    /// dx = W^T * dz
    /// </summary>
    public Vector Backward(Vector outputGradient)
    {
        ArgumentNullException.ThrowIfNull(outputGradient);

        if (_lastInput is null)
        {
            throw new InvalidOperationException(
                "Backward cannot be called before Forward.");
        }

        if (outputGradient.Length != _weights.Rows)
        {
            throw new ArgumentException(
                $"Expected output gradient size {_weights.Rows}, " +
                $"but received {outputGradient.Length}.",
                nameof(outputGradient));
        }

        for (var row = 0; row < _weights.Rows; row++)
        {
            BiasGradients[row] = outputGradient[row];

            for (var column = 0; column < _weights.Columns; column++)
            {
                WeightGradients[row, column] =
                    outputGradient[row] * _lastInput[column];
            }
        }

        var inputGradient = new double[_weights.Columns];

        for (var column = 0; column < _weights.Columns; column++)
        {
            var sum = 0.0;

            for (var row = 0; row < _weights.Rows; row++)
            {
                sum += _weights[row, column] * outputGradient[row];
            }

            inputGradient[column] = sum;
        }

        return new Vector(inputGradient);
    }
}

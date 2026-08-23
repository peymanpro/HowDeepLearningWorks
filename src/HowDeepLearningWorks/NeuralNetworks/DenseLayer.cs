using HowDeepLearningWorks.ActivationFunctions;
using HowDeepLearningWorks.Mathematics;

namespace HowDeepLearningWorks.NeuralNetworks;

/// <summary>
/// Represents a fully connected neural network layer with an optional activation function.
/// </summary>
public sealed class DenseLayer
{
    private readonly Matrix _weights;
    private readonly Vector _bias;
    private readonly IActivationFunction? _activation;

    private Vector? _lastInput;
    private Vector? _lastPreActivation;

    /// <summary>
    /// Creates a fully connected layer without an activation function.
    /// </summary>
    public DenseLayer(int inputSize, int outputSize)
        : this(inputSize, outputSize, null)
    {
    }

    /// <summary>
    /// Creates a fully connected layer with an optional activation function.
    /// </summary>
    public DenseLayer(
        int inputSize,
        int outputSize,
        IActivationFunction? activation)
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
        _activation = activation;

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
    /// Gets the activation function.
    /// </summary>
    public IActivationFunction? Activation => _activation;

    /// <summary>
    /// Gets the gradients of the weights.
    /// </summary>
    public Matrix WeightGradients { get; }

    /// <summary>
    /// Gets the gradients of the bias.
    /// </summary>
    public Vector BiasGradients { get; }

    /// <summary>
    /// Computes:
    /// z = Wx + b
    /// a = activation(z)
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

        var preActivation = (_weights * input) + _bias;

        var preActivationCopy = new double[preActivation.Length];

        for (var i = 0; i < preActivation.Length; i++)
        {
            preActivationCopy[i] = preActivation[i];
        }

        _lastPreActivation = new Vector(preActivationCopy);

        if (_activation is null)
        {
            return preActivation;
        }

        var activated = new double[preActivation.Length];

        for (var i = 0; i < preActivation.Length; i++)
        {
            activated[i] = _activation.Forward(preActivation[i]);
        }

        return new Vector(activated);
    }

    /// <summary>
    /// Computes gradients for the layer.
    /// </summary>
    public Vector Backward(Vector outputGradient)
    {
        ArgumentNullException.ThrowIfNull(outputGradient);

        if (_lastInput is null || _lastPreActivation is null)
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

        var localGradient = new double[outputGradient.Length];

        for (var i = 0; i < outputGradient.Length; i++)
        {
            localGradient[i] = outputGradient[i];

            if (_activation is not null)
            {
                localGradient[i] *=
                    _activation.Derivative(_lastPreActivation[i]);
            }
        }

        for (var row = 0; row < _weights.Rows; row++)
        {
            BiasGradients[row] = localGradient[row];

            for (var column = 0; column < _weights.Columns; column++)
            {
                WeightGradients[row, column] =
                    localGradient[row] * _lastInput[column];
            }
        }

        var inputGradient = new double[_weights.Columns];

        for (var column = 0; column < _weights.Columns; column++)
        {
            var sum = 0.0;

            for (var row = 0; row < _weights.Rows; row++)
            {
                sum += _weights[row, column] * localGradient[row];
            }

            inputGradient[column] = sum;
        }

        return new Vector(inputGradient);
    }

    /// <summary>
    /// Updates weights and biases using gradient descent.
    /// </summary>
    public void UpdateParameters(double learningRate)
    {
        if (learningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate));
        }

        for (var row = 0; row < _weights.Rows; row++)
        {
            for (var column = 0; column < _weights.Columns; column++)
            {
                _weights[row, column] -=
                    learningRate * WeightGradients[row, column];
            }

            _bias[row] -=
                learningRate * BiasGradients[row];
        }
    }
}

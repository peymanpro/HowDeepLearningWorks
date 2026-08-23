using HowDeepLearningWorks.Mathematics;

namespace HowDeepLearningWorks.NeuralNetworks;

/// <summary>
/// Represents a sequential neural network composed of dense layers.
/// </summary>
public sealed class NeuralNetwork
{
    private readonly List<DenseLayer> _layers = new();

    /// <summary>
    /// Gets the layers in the network.
    /// </summary>
    public IReadOnlyList<DenseLayer> Layers => _layers;

    /// <summary>
    /// Adds a dense layer to the network.
    /// </summary>
    public void Add(DenseLayer layer)
    {
        ArgumentNullException.ThrowIfNull(layer);

        if (_layers.Count > 0)
        {
            var previousLayer = _layers[^1];

            if (previousLayer.Weights.Rows != layer.Weights.Columns)
            {
                throw new ArgumentException(
                    "The input size of the new layer must match " +
                    "the output size of the previous layer.",
                    nameof(layer));
            }
        }

        _layers.Add(layer);
    }

    /// <summary>
    /// Performs forward propagation through all layers.
    /// </summary>
    public Vector Forward(Vector input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (_layers.Count == 0)
        {
            throw new InvalidOperationException(
                "The network must contain at least one layer.");
        }

        var output = input;

        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    /// <summary>
    /// Performs backward propagation through all layers.
    /// </summary>
    public Vector Backward(Vector outputGradient)
    {
        ArgumentNullException.ThrowIfNull(outputGradient);

        if (_layers.Count == 0)
        {
            throw new InvalidOperationException(
                "The network must contain at least one layer.");
        }

        var gradient = outputGradient;

        for (var i = _layers.Count - 1; i >= 0; i--)
        {
            gradient = _layers[i].Backward(gradient);
        }

        return gradient;
    }
}

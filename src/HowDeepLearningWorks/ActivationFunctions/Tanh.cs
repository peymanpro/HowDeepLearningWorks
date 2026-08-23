namespace HowDeepLearningWorks.ActivationFunctions;

/// <summary>
/// Hyperbolic tangent activation function.
/// </summary>
public sealed class Tanh : IActivationFunction
{
    /// <inheritdoc />
    public double Forward(double value) => Math.Tanh(value);

    /// <inheritdoc />
    public double Derivative(double value)
    {
        var output = Math.Tanh(value);
        return 1.0 - output * output;
    }
}

namespace HowDeepLearningWorks.ActivationFunctions;

/// <summary>
/// Sigmoid activation function.
/// </summary>
public sealed class Sigmoid : IActivationFunction
{
    /// <inheritdoc />
    public double Forward(double value)
    {
        return 1.0 / (1.0 + Math.Exp(-value));
    }

    /// <inheritdoc />
    public double Derivative(double value)
    {
        var output = Forward(value);
        return output * (1.0 - output);
    }
}

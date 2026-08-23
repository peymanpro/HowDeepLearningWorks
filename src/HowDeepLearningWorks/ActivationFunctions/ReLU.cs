namespace HowDeepLearningWorks.ActivationFunctions;

/// <summary>
/// Rectified Linear Unit activation function.
/// </summary>
public sealed class ReLU : IActivationFunction
{
    /// <inheritdoc />
    public double Forward(double value) => Math.Max(0.0, value);

    /// <inheritdoc />
    public double Derivative(double value) => value > 0.0 ? 1.0 : 0.0;
}

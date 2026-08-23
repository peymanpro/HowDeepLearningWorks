namespace HowDeepLearningWorks.ActivationFunctions;

/// <summary>
/// Defines the contract for an activation function.
/// </summary>
public interface IActivationFunction
{
    /// <summary>
    /// Computes the activation value.
    /// </summary>
    double Forward(double value);

    /// <summary>
    /// Computes the derivative of the activation function.
    /// </summary>
    double Derivative(double value);
}

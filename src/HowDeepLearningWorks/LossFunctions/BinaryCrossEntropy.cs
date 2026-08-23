namespace HowDeepLearningWorks.LossFunctions;

/// <summary>
/// Computes Binary Cross Entropy loss for binary classification.
/// </summary>
public sealed class BinaryCrossEntropy
{
    private const double Epsilon = 1e-15;

    /// <summary>
    /// Computes the binary cross entropy loss.
    /// </summary>
    public double Forward(double prediction, double target)
    {
        ValidateProbability(prediction);
        ValidateTarget(target);

        var p = Math.Clamp(prediction, Epsilon, 1.0 - Epsilon);

        return -(target * Math.Log(p) +
                 (1.0 - target) * Math.Log(1.0 - p));
    }

    /// <summary>
    /// Computes the derivative of the loss with respect to the prediction.
    /// </summary>
    public double Derivative(double prediction, double target)
    {
        ValidateProbability(prediction);
        ValidateTarget(target);

        var p = Math.Clamp(prediction, Epsilon, 1.0 - Epsilon);

        return -(target / p) +
               ((1.0 - target) / (1.0 - p));
    }

    private static void ValidateProbability(double value)
    {
        if (double.IsNaN(value) || value < 0.0 || value > 1.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(value),
                "Prediction must be a value between 0 and 1.");
        }
    }

    private static void ValidateTarget(double value)
    {
        if (double.IsNaN(value) || value < 0.0 || value > 1.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(value),
                "Target must be a value between 0 and 1.");
        }
    }
}

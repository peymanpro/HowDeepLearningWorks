using HowDeepLearningWorks.ActivationFunctions;
using HowDeepLearningWorks.LossFunctions;
using HowDeepLearningWorks.Mathematics;
using HowDeepLearningWorks.NeuralNetworks;

namespace HowDeepLearningWorks.Tests;

public sealed class NeuralNetworkLearningTests
{
    [Fact]
    public void DenseLayerForwardAndBackward_CalculateExpectedValues()
    {
        var layer = new DenseLayer(2, 2);
        layer.Weights[0, 0] = 1.0;
        layer.Weights[0, 1] = 2.0;
        layer.Weights[1, 0] = 3.0;
        layer.Weights[1, 1] = 4.0;
        layer.Bias[0] = 0.5;
        layer.Bias[1] = 1.0;

        var output = layer.Forward(new Vector(new[] { 2.0, 3.0 }));
        AssertVectorEqual(new[] { 8.5, 19.0 }, output);

        var inputGradient = layer.Backward(new Vector(new[] { 0.1, 0.2 }));

        Assert.Equal(0.2, layer.WeightGradients[0, 0], 12);
        Assert.Equal(0.3, layer.WeightGradients[0, 1], 12);
        Assert.Equal(0.4, layer.WeightGradients[1, 0], 12);
        Assert.Equal(0.6, layer.WeightGradients[1, 1], 12);
        AssertVectorEqual(new[] { 0.1, 0.2 }, layer.BiasGradients);
        AssertVectorEqual(new[] { 0.7, 1.0 }, inputGradient);
    }

    [Fact]
    public void DenseLayerBackward_RequiresAnEarlierForwardPass()
    {
        var layer = new DenseLayer(2, 1);

        Assert.Throws<InvalidOperationException>(
            () => layer.Backward(new Vector(new[] { 1.0 })));
    }

    [Fact]
    public void NeuralNetwork_RejectsIncompatibleLayersAndEmptyForwardPass()
    {
        var network = new NeuralNetwork();

        Assert.Throws<InvalidOperationException>(
            () => network.Forward(new Vector(new[] { 1.0 })));

        network.Add(new DenseLayer(2, 3));

        Assert.Throws<ArgumentException>(
            () => network.Add(new DenseLayer(4, 1)));
    }

    [Fact]
    public void BackpropagationGradients_MatchFiniteDifferencesForWeightsAndBiases()
    {
        const double epsilon = 1e-5;
        const double tolerance = 1e-6;

        var network = CreateGradientCheckNetwork();
        var input = new Vector(new[] { 0.6, 0.8 });
        const double target = 1.0;
        var lossFunction = new BinaryCrossEntropy();

        var prediction = network.Forward(input);
        var lossGradient = lossFunction.Derivative(prediction[0], target);
        network.Backward(new Vector(new[] { lossGradient }));

        foreach (var layer in network.Layers)
        {
            for (var row = 0; row < layer.Weights.Rows; row++)
            {
                for (var column = 0; column < layer.Weights.Columns; column++)
                {
                    var original = layer.Weights[row, column];
                    var analytical = layer.WeightGradients[row, column];

                    layer.Weights[row, column] = original + epsilon;
                    var positiveLoss = EvaluateLoss(network, lossFunction, input, target);

                    layer.Weights[row, column] = original - epsilon;
                    var negativeLoss = EvaluateLoss(network, lossFunction, input, target);

                    layer.Weights[row, column] = original;

                    var numerical = (positiveLoss - negativeLoss) / (2.0 * epsilon);
                    AssertGradientClose(analytical, numerical, tolerance,
                        $"Weight gradient [{row}, {column}]");
                }
            }

            for (var index = 0; index < layer.Bias.Length; index++)
            {
                var original = layer.Bias[index];
                var analytical = layer.BiasGradients[index];

                layer.Bias[index] = original + epsilon;
                var positiveLoss = EvaluateLoss(network, lossFunction, input, target);

                layer.Bias[index] = original - epsilon;
                var negativeLoss = EvaluateLoss(network, lossFunction, input, target);

                layer.Bias[index] = original;

                var numerical = (positiveLoss - negativeLoss) / (2.0 * epsilon);
                AssertGradientClose(analytical, numerical, tolerance,
                    $"Bias gradient [{index}]");
            }
        }
    }

    [Fact]
    public void GradientDescentStep_DecreasesBinaryCrossEntropyForPositiveExample()
    {
        var layer = new DenseLayer(1, 1, new Sigmoid());
        var input = new Vector(new[] { 1.0 });
        const double target = 1.0;
        var lossFunction = new BinaryCrossEntropy();

        var before = lossFunction.Forward(layer.Forward(input)[0], target);
        var derivative = lossFunction.Derivative(layer.Forward(input)[0], target);
        layer.Backward(new Vector(new[] { derivative }));
        layer.UpdateParameters(0.1);
        var after = lossFunction.Forward(layer.Forward(input)[0], target);

        Assert.True(after < before, $"Expected loss to decrease, but it changed from {before} to {after}.");
    }

    private static NeuralNetwork CreateGradientCheckNetwork()
    {
        var network = new NeuralNetwork();

        var hidden = new DenseLayer(2, 2, new ReLU());
        hidden.Weights[0, 0] = 0.2;
        hidden.Weights[0, 1] = 0.1;
        hidden.Weights[1, 0] = 0.1;
        hidden.Weights[1, 1] = 0.3;
        hidden.Bias[0] = 0.1;
        hidden.Bias[1] = 0.1;

        var output = new DenseLayer(2, 1, new Sigmoid());
        output.Weights[0, 0] = 0.2;
        output.Weights[0, 1] = 0.4;
        output.Bias[0] = 0.1;

        network.Add(hidden);
        network.Add(output);
        return network;
    }

    private static double EvaluateLoss(
        NeuralNetwork network,
        BinaryCrossEntropy lossFunction,
        Vector input,
        double target)
    {
        var prediction = network.Forward(input);
        return lossFunction.Forward(prediction[0], target);
    }

    private static void AssertGradientClose(
        double analytical,
        double numerical,
        double tolerance,
        string parameterName)
    {
        Assert.True(
            Math.Abs(analytical - numerical) <= tolerance,
            $"{parameterName}: analytical={analytical:G10}, numerical={numerical:G10}, tolerance={tolerance:G3}");
    }

    private static void AssertVectorEqual(double[] expected, Vector actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (var index = 0; index < expected.Length; index++)
        {
            Assert.Equal(expected[index], actual[index], 12);
        }
    }
}

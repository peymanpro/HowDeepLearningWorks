using HowDeepLearningWorks.ActivationFunctions;
using HowDeepLearningWorks.Mathematics;
using HowDeepLearningWorks.NeuralNetworks;
using HowDeepLearningWorks.NeuralNetworks.Initialization;

namespace HowDeepLearningWorks.Tests;

public sealed class WeightInitializationTests
{
    [Fact]
    public void HeNormalInitializer_IsSeededAndHasExpectedVarianceScale()
    {
        const int fanOut = 128;
        const int fanIn = 64;
        const int seed = 2026;

        var first = new Matrix(fanOut, fanIn);
        var second = new Matrix(fanOut, fanIn);
        var initializer = new HeNormalInitializer();

        initializer.Initialize(first, new Random(seed));
        initializer.Initialize(second, new Random(seed));

        var squaredSum = 0.0;

        for (var row = 0; row < fanOut; row++)
        {
            for (var column = 0; column < fanIn; column++)
            {
                Assert.Equal(first[row, column], second[row, column], 14);
                squaredSum += first[row, column] * first[row, column];
            }
        }

        var meanSquare = squaredSum / (fanOut * fanIn);
        var expectedVariance = 2.0 / fanIn;

        Assert.InRange(meanSquare, expectedVariance * 0.8, expectedVariance * 1.2);
        Assert.NotEqual(first[0, 0], first[0, 1]);
    }

    [Fact]
    public void XavierUniformInitializer_IsSeededAndRespectsGlorotBounds()
    {
        const int fanOut = 12;
        const int fanIn = 5;
        const int seed = 99;
        var limit = Math.Sqrt(6.0 / (fanIn + fanOut));

        var first = new Matrix(fanOut, fanIn);
        var second = new Matrix(fanOut, fanIn);
        var initializer = new XavierUniformInitializer();

        initializer.Initialize(first, new Random(seed));
        initializer.Initialize(second, new Random(seed));

        for (var row = 0; row < fanOut; row++)
        {
            for (var column = 0; column < fanIn; column++)
            {
                Assert.Equal(first[row, column], second[row, column], 14);
                Assert.InRange(first[row, column], -limit, limit);
            }
        }

        Assert.NotEqual(first[0, 0], first[0, 1]);
    }

    [Fact]
    public void NeuralNetworkInitialization_IsReproducibleAndLeavesBiasesAtZero()
    {
        var first = CreateNetwork();
        var second = CreateNetwork();

        first.InitializeWeights(seed: 17);
        second.InitializeWeights(seed: 17);

        Assert.Equal(first.Layers.Count, second.Layers.Count);

        for (var layerIndex = 0; layerIndex < first.Layers.Count; layerIndex++)
        {
            var left = first.Layers[layerIndex];
            var right = second.Layers[layerIndex];

            Assert.Equal(left.Weights.Rows, right.Weights.Rows);
            Assert.Equal(left.Weights.Columns, right.Weights.Columns);

            for (var row = 0; row < left.Weights.Rows; row++)
            {
                for (var column = 0; column < left.Weights.Columns; column++)
                {
                    Assert.Equal(
                        left.Weights[row, column],
                        right.Weights[row, column],
                        14);
                }
            }

            for (var index = 0; index < left.Bias.Length; index++)
            {
                Assert.Equal(0.0, left.Bias[index]);
            }
        }

        Assert.NotEqual(
            first.Layers[0].Weights[0, 0],
            first.Layers[0].Weights[0, 1]);
    }

    [Fact]
    public void NeuralNetworkInitialization_RequiresAtLeastOneLayer()
    {
        var network = new NeuralNetwork();

        Assert.Throws<InvalidOperationException>(
            () => network.InitializeWeights(seed: 1));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Initializers_RejectEmptyMatrices(bool useHe)
    {
        var weights = new Matrix(0, 3);
        IWeightInitializer initializer = useHe
            ? new HeNormalInitializer()
            : new XavierUniformInitializer();

        Assert.Throws<ArgumentException>(
            () => initializer.Initialize(weights, new Random(1)));
    }

    private static NeuralNetwork CreateNetwork()
    {
        var network = new NeuralNetwork();
        network.Add(new DenseLayer(3, 4, new ReLU()));
        network.Add(new DenseLayer(4, 2, new Sigmoid()));
        return network;
    }
}

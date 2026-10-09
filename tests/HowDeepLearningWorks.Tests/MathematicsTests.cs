using HowDeepLearningWorks.ActivationFunctions;
using HowDeepLearningWorks.LossFunctions;
using HowDeepLearningWorks.Mathematics;

namespace HowDeepLearningWorks.Tests;

public sealed class MathematicsTests
{
    [Fact]
    public void VectorOperations_ReturnExpectedResults()
    {
        var left = new Vector(new[] { 1.0, 2.0, 3.0 });
        var right = new Vector(new[] { 4.0, 5.0, 6.0 });

        Assert.Equal(new[] { 5.0, 7.0, 9.0 }, (left + right).ToArray());
        Assert.Equal(new[] { 3.0, 3.0, 3.0 }, (right - left).ToArray());
        Assert.Equal(new[] { 2.0, 4.0, 6.0 }, (left * 2.0).ToArray());
        Assert.Equal(32.0, Vector.Dot(left, right), 10);
    }

    [Fact]
    public void VectorOperations_RejectDifferentLengths()
    {
        var left = new Vector(new[] { 1.0, 2.0 });
        var right = new Vector(new[] { 1.0 });

        Assert.Throws<ArgumentException>(() => _ = left + right);
        Assert.Throws<ArgumentException>(() => _ = Vector.Dot(left, right));
    }

    [Fact]
    public void MatrixOperations_ReturnExpectedResults()
    {
        var left = new Matrix(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        var right = new Matrix(new double[,]
        {
            { 5.0, 6.0 },
            { 7.0, 8.0 }
        });

        AssertMatrixEqual(
            new Matrix(new double[,] { { 19.0, 22.0 }, { 43.0, 50.0 } }),
            left * right);

        Assert.Equal(new[] { 17.0, 39.0 },
            (left * new Vector(new[] { 5.0, 6.0 })).ToArray());

        AssertMatrixEqual(
            new Matrix(new double[,] { { 1.0, 3.0 }, { 2.0, 4.0 } }),
            left.Transpose());
    }

    [Fact]
    public void MatrixMultiplication_RejectsIncompatibleDimensions()
    {
        var left = new Matrix(2, 3);
        var right = new Matrix(2, 2);

        Assert.Throws<ArgumentException>(() => _ = left * right);
        Assert.Throws<ArgumentException>(() => _ = left * new Vector(2));
    }

    [Fact]
    public void ActivationFunctions_ReturnExpectedValuesAndDerivatives()
    {
        var relu = new ReLU();
        var sigmoid = new Sigmoid();
        var tanh = new Tanh();

        Assert.Equal(0.0, relu.Forward(-2.0));
        Assert.Equal(1.0, relu.Derivative(2.0));
        Assert.Equal(0.0, relu.Derivative(0.0));

        Assert.Equal(0.5, sigmoid.Forward(0.0), 12);
        Assert.Equal(0.25, sigmoid.Derivative(0.0), 12);

        Assert.Equal(0.0, tanh.Forward(0.0), 12);
        Assert.Equal(1.0, tanh.Derivative(0.0), 12);
    }

    [Fact]
    public void BinaryCrossEntropy_ReturnsExpectedLossAndDerivative()
    {
        var loss = new BinaryCrossEntropy();

        Assert.Equal(-Math.Log(0.9), loss.Forward(0.9, 1.0), 12);
        Assert.Equal(-Math.Log(0.9), loss.Forward(0.1, 0.0), 12);
        Assert.Equal(-1.0 / 0.9, loss.Derivative(0.9, 1.0), 12);
        Assert.Equal(1.0 / 0.9, loss.Derivative(0.1, 0.0), 12);
    }

    [Theory]
    [InlineData(-0.1, 1.0)]
    [InlineData(1.1, 0.0)]
    [InlineData(double.NaN, 1.0)]
    [InlineData(0.5, -0.1)]
    [InlineData(0.5, 1.1)]
    public void BinaryCrossEntropy_RejectsInvalidInputs(double prediction, double target)
    {
        var loss = new BinaryCrossEntropy();

        Assert.Throws<ArgumentOutOfRangeException>(
            () => loss.Forward(prediction, target));
    }

    private static void AssertMatrixEqual(Matrix expected, Matrix actual)
    {
        Assert.Equal(expected.Rows, actual.Rows);
        Assert.Equal(expected.Columns, actual.Columns);

        for (var row = 0; row < expected.Rows; row++)
        {
            for (var column = 0; column < expected.Columns; column++)
            {
                Assert.Equal(expected[row, column], actual[row, column], 12);
            }
        }
    }
}

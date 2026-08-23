using HowDeepLearningWorks.ActivationFunctions;
using HowDeepLearningWorks.LossFunctions;
using HowDeepLearningWorks.Mathematics;

Console.WriteLine("HowDeepLearningWorks");
Console.WriteLine("====================");
Console.WriteLine();

RunPhase11Checks();
RunPhase12Checks();
RunPhase13Checks();

Console.WriteLine();
Console.WriteLine("All Phase 1 checks passed.");

static void RunPhase11Checks()
{
    var vectorA = new Vector(new[] { 1.0, 2.0, 3.0 });
    var vectorB = new Vector(new[] { 4.0, 5.0, 6.0 });

    AssertVectorEqual(
        "Vector addition",
        vectorA + vectorB,
        new[] { 5.0, 7.0, 9.0 });

    AssertVectorEqual(
        "Vector subtraction",
        vectorB - vectorA,
        new[] { 3.0, 3.0, 3.0 });

    AssertVectorEqual(
        "Vector scalar multiplication",
        vectorA * 2.0,
        new[] { 2.0, 4.0, 6.0 });

   AssertApproximately(
    "Vector dot product",
    Vector.Dot(vectorA, vectorB),
    32.0);

    var matrixA = new Matrix(new[,]
    {
        { 1.0, 2.0 },
        { 3.0, 4.0 }
    });

    var vectorC = new Vector(new[] { 5.0, 6.0 });

    AssertVectorEqual(
        "Matrix × Vector",
        matrixA * vectorC,
        new[] { 17.0, 39.0 });

    var matrixB = new Matrix(new[,]
    {
        { 5.0, 6.0 },
        { 7.0, 8.0 }
    });

    AssertMatrixEqual(
        "Matrix × Matrix",
        matrixA * matrixB,
        new[,]
        {
            { 19.0, 22.0 },
            { 43.0, 50.0 }
        });

    AssertMatrixEqual(
        "Matrix transpose",
        matrixA.Transpose(),
        new[,]
        {
            { 1.0, 3.0 },
            { 2.0, 4.0 }
        });

    Console.WriteLine("All Phase 1.1 checks passed.");
    Console.WriteLine();
}

static void RunPhase12Checks()
{
    var relu = new ReLU();

    AssertApproximately(
        "ReLU(-2)",
        relu.Forward(-2.0),
        0.0);

    AssertApproximately(
        "ReLU(3)",
        relu.Forward(3.0),
        3.0);

    AssertApproximately(
        "ReLU derivative(-2)",
        relu.Derivative(-2.0),
        0.0);

    AssertApproximately(
        "ReLU derivative(3)",
        relu.Derivative(3.0),
        1.0);

    Console.WriteLine("Phase 1.2 activation checks passed.");
    Console.WriteLine();
}

static void RunPhase13Checks()
{
    var loss = new BinaryCrossEntropy();

    AssertApproximately(
        "BCE(0.9, 1)",
        loss.Forward(0.9, 1.0),
        -Math.Log(0.9));

    AssertApproximately(
        "BCE(0.1, 0)",
        loss.Forward(0.1, 0.0),
        -Math.Log(0.9));

    AssertApproximately(
        "BCE derivative(0.9, 1)",
        loss.Derivative(0.9, 1.0),
        -1.0 / 0.9);

    AssertApproximately(
        "BCE derivative(0.1, 0)",
        loss.Derivative(0.1, 0.0),
        1.0 / 0.9);

    Console.WriteLine("Phase 1.3 loss checks passed.");
    Console.WriteLine();
}

static void AssertApproximately(
    string name,
    double actual,
    double expected,
    double tolerance = 1e-10)
{
    if (Math.Abs(actual - expected) > tolerance)
    {
        throw new Exception(
            $"{name} FAILED. Expected {expected}, actual {actual}.");
    }

    Console.WriteLine($"PASS: {name}");
}

static void AssertVectorEqual(
    string name,
    Vector actual,
    double[] expected,
    double tolerance = 1e-10)
{
    if (actual.Length != expected.Length)
    {
        throw new Exception(
            $"{name} FAILED. Vector lengths differ.");
    }

    for (var i = 0; i < expected.Length; i++)
    {
        if (Math.Abs(actual[i] - expected[i]) > tolerance)
        {
            throw new Exception(
                $"{name} FAILED at index {i}. " +
                $"Expected {expected[i]}, actual {actual[i]}.");
        }
    }

    Console.WriteLine($"PASS: {name}");
}

static void AssertMatrixEqual(
    string name,
    Matrix actual,
    double[,] expected,
    double tolerance = 1e-10)
{
    if (actual.Rows != expected.GetLength(0) ||
        actual.Columns != expected.GetLength(1))
    {
        throw new Exception(
            $"{name} FAILED. Matrix dimensions differ.");
    }

    for (var row = 0; row < actual.Rows; row++)
    {
        for (var column = 0; column < actual.Columns; column++)
        {
            if (Math.Abs(
                    actual[row, column] -
                    expected[row, column]) > tolerance)
            {
                throw new Exception(
                    $"{name} FAILED at [{row},{column}]. " +
                    $"Expected {expected[row, column]}, " +
                    $"actual {actual[row, column]}.");
            }
        }
    }

    Console.WriteLine($"PASS: {name}");
}
using HowDeepLearningWorks.ActivationFunctions;
using HowDeepLearningWorks.Mathematics;

RunPhase11Checks();
RunPhase12Checks();

Console.WriteLine();
Console.WriteLine("All Phase 1 checks passed.");

static void RunPhase11Checks()
{
    var vectorA = new Vector([1, 2, 3]);
    var vectorB = new Vector([4, 5, 6]);

    AssertVector("Vector addition", vectorA + vectorB, [5, 7, 9]);
    AssertVector("Vector subtraction", vectorB - vectorA, [3, 3, 3]);
    AssertVector("Vector scalar multiplication", vectorA * 2.0, [2, 4, 6]);
    AssertScalar("Vector dot product", Vector.Dot(vectorA, vectorB), 32.0);

    var matrixA = new Matrix(new double[,]
    {
        { 1, 2 },
        { 3, 4 }
    });

    var matrixB = new Matrix(new double[,]
    {
        { 5, 6 },
        { 7, 8 }
    });

    AssertVector(
        "Matrix × Vector",
        matrixA * new Vector([5, 6]),
        [17, 39]);

    AssertMatrix(
        "Matrix × Matrix",
        matrixA * matrixB,
        new double[,]
        {
            { 19, 22 },
            { 43, 50 }
        });

    AssertMatrix(
        "Matrix transpose",
        matrixA.Transpose(),
        new double[,]
        {
            { 1, 3 },
            { 2, 4 }
        });
}

static void RunPhase12Checks()
{
    var relu = new ReLU();

    AssertScalar("ReLU(-2)", relu.Forward(-2), 0);
    AssertScalar("ReLU(3)", relu.Forward(3), 3);
    AssertScalar("ReLU derivative(-2)", relu.Derivative(-2), 0);
    AssertScalar("ReLU derivative(3)", relu.Derivative(3), 1);

    var sigmoid = new Sigmoid();

    AssertApproximately(
        "Sigmoid(0)",
        sigmoid.Forward(0),
        0.5);

    AssertApproximately(
        "Sigmoid derivative(0)",
        sigmoid.Derivative(0),
        0.25);

    var tanh = new Tanh();

    AssertApproximately(
        "Tanh(0)",
        tanh.Forward(0),
        0);

    AssertApproximately(
        "Tanh derivative(0)",
        tanh.Derivative(0),
        1);

    Console.WriteLine();
    Console.WriteLine("Phase 1.2 activation checks passed.");
}

static void AssertVector(string name, Vector actual, double[] expected)
{
    if (actual.Length != expected.Length)
    {
        throw new InvalidOperationException($"{name}: length mismatch.");
    }

    for (var i = 0; i < expected.Length; i++)
    {
        AssertApproximately($"{name}[{i}]", actual[i], expected[i]);
    }

    Console.WriteLine($"PASS: {name}");
}

static void AssertScalar(string name, double actual, double expected)
{
    AssertApproximately(name, actual, expected);
    Console.WriteLine($"PASS: {name}");
}

static void AssertApproximately(string name, double actual, double expected)
{
    const double tolerance = 1e-12;

    if (Math.Abs(actual - expected) > tolerance)
    {
        throw new InvalidOperationException(
            $"{name}: expected {expected}, actual {actual}.");
    }
}

static void AssertMatrix(string name, Matrix actual, double[,] expected)
{
    if (actual.Rows != expected.GetLength(0) ||
        actual.Columns != expected.GetLength(1))
    {
        throw new InvalidOperationException($"{name}: dimension mismatch.");
    }

    for (var row = 0; row < actual.Rows; row++)
    {
        for (var column = 0; column < actual.Columns; column++)
        {
            AssertApproximately(
                $"{name}[{row},{column}]",
                actual[row, column],
                expected[row, column]);
        }
    }

    Console.WriteLine($"PASS: {name}");
}

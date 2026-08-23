using HowDeepLearningWorks.Mathematics;

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

var matrixVectorResult = matrixA * new Vector([5, 6]);

AssertVector("Matrix × Vector", matrixVectorResult, [17, 39]);

var matrixResult = matrixA * matrixB;

AssertMatrix("Matrix × Matrix", matrixResult, new double[,]
{
    { 19, 22 },
    { 43, 50 }
});

var transpose = matrixA.Transpose();

AssertMatrix("Matrix transpose", transpose, new double[,]
{
    { 1, 3 },
    { 2, 4 }
});

Console.WriteLine();
Console.WriteLine("All Phase 1.1 checks passed.");

static void AssertVector(string name, Vector actual, double[] expected)
{
    if (actual.Length != expected.Length)
    {
        throw new InvalidOperationException($"{name}: length mismatch.");
    }

    for (var i = 0; i < expected.Length; i++)
    {
        if (Math.Abs(actual[i] - expected[i]) > 1e-12)
        {
            throw new InvalidOperationException(
                $"{name}: mismatch at index {i}. Expected {expected[i]}, actual {actual[i]}.");
        }
    }

    Console.WriteLine($"PASS: {name}");
}

static void AssertScalar(string name, double actual, double expected)
{
    if (Math.Abs(actual - expected) > 1e-12)
    {
        throw new InvalidOperationException(
            $"{name}: Expected {expected}, actual {actual}.");
    }

    Console.WriteLine($"PASS: {name}");
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
            if (Math.Abs(actual[row, column] - expected[row, column]) > 1e-12)
            {
                throw new InvalidOperationException(
                    $"{name}: mismatch at [{row},{column}]. " +
                    $"Expected {expected[row, column]}, actual {actual[row, column]}.");
            }
        }
    }

    Console.WriteLine($"PASS: {name}");
}
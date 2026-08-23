using HowDeepLearningWorks.ActivationFunctions;
using HowDeepLearningWorks.LossFunctions;
using HowDeepLearningWorks.Mathematics;
using HowDeepLearningWorks.NeuralNetworks;

Console.WriteLine("HowDeepLearningWorks");
Console.WriteLine("====================");
Console.WriteLine();

RunPhase11Checks();
RunPhase12Checks();
RunPhase13Checks();
RunPhase21Checks();
RunPhase22Checks();
RunPhase23Checks();
RunPhase31Checks();

Console.WriteLine();
Console.WriteLine("All Phase 3 checks passed.");

static void RunPhase11Checks()
{
    var left = new Vector(new[] { 1.0, 2.0, 3.0 });
    var right = new Vector(new[] { 4.0, 5.0, 6.0 });

    AssertVectorEqual("Vector addition", left + right,
        new Vector(new[] { 5.0, 7.0, 9.0 }));

    AssertVectorEqual("Vector subtraction", right - left,
        new Vector(new[] { 3.0, 3.0, 3.0 }));

    AssertVectorEqual("Vector scalar multiplication", left * 2.0,
        new Vector(new[] { 2.0, 4.0, 6.0 }));

    AssertApproximately("Vector dot product",
        Vector.Dot(left, right), 32.0);

    var matrix = new Matrix(new double[,]
    {
        { 1.0, 2.0 },
        { 3.0, 4.0 }
    });

    var vector = new Vector(new[] { 5.0, 6.0 });

    AssertVectorEqual("Matrix × Vector",
        matrix * vector,
        new Vector(new[] { 17.0, 39.0 }));

    var matrixRight = new Matrix(new double[,]
    {
        { 5.0, 6.0 },
        { 7.0, 8.0 }
    });

    AssertMatrixEqual("Matrix × Matrix",
        matrix * matrixRight,
        new Matrix(new double[,]
        {
            { 19.0, 22.0 },
            { 43.0, 50.0 }
        }));

    AssertMatrixEqual("Matrix transpose",
        matrix.Transpose(),
        new Matrix(new double[,]
        {
            { 1.0, 3.0 },
            { 2.0, 4.0 }
        }));

    Console.WriteLine("All Phase 1.1 checks passed.");
    Console.WriteLine();
}

static void RunPhase12Checks()
{
    var relu = new ReLU();

    AssertApproximately("ReLU(-2)", relu.Forward(-2.0), 0.0);
    AssertApproximately("ReLU(3)", relu.Forward(3.0), 3.0);
    AssertApproximately("ReLU derivative(-2)", relu.Derivative(-2.0), 0.0);
    AssertApproximately("ReLU derivative(3)", relu.Derivative(3.0), 1.0);

    Console.WriteLine("Phase 1.2 activation checks passed.");
    Console.WriteLine();
}

static void RunPhase13Checks()
{
    var loss = new BinaryCrossEntropy();

    AssertApproximately("BCE(0.9, 1)",
        loss.Forward(0.9, 1.0), -Math.Log(0.9));

    AssertApproximately("BCE(0.1, 0)",
        loss.Forward(0.1, 0.0), -Math.Log(0.9));

    AssertApproximately("BCE derivative(0.9, 1)",
        loss.Derivative(0.9, 1.0), -1.0 / 0.9);

    AssertApproximately("BCE derivative(0.1, 0)",
        loss.Derivative(0.1, 0.0), 1.0 / 0.9);

    Console.WriteLine("Phase 1.3 loss checks passed.");
    Console.WriteLine();
}

static void RunPhase21Checks()
{
    var layer = new DenseLayer(2, 3);

    AssertEqual("DenseLayer input size",
        layer.Weights.Columns, 2);

    AssertEqual("DenseLayer output size",
        layer.Weights.Rows, 3);

    AssertEqual("DenseLayer bias size",
        layer.Bias.Length, 3);

    var input = new Vector(new[] { 2.0, 3.0 });
    var output = layer.Forward(input);

    AssertVectorEqual("DenseLayer forward with zero weights",
        output,
        new Vector(new[] { 0.0, 0.0, 0.0 }));

    Console.WriteLine("Phase 2.1 dense layer checks passed.");
    Console.WriteLine();
}

static void RunPhase22Checks()
{
    var layer = new DenseLayer(2, 2);

    layer.Weights[0, 0] = 1.0;
    layer.Weights[0, 1] = 2.0;
    layer.Weights[1, 0] = 3.0;
    layer.Weights[1, 1] = 4.0;

    layer.Bias[0] = 0.5;
    layer.Bias[1] = 1.0;

    var input = new Vector(new[] { 2.0, 3.0 });

    var forward = layer.Forward(input);

    AssertVectorEqual("DenseLayer forward",
        forward,
        new Vector(new[] { 8.5, 19.0 }));

    var outputGradient = new Vector(new[] { 0.1, 0.2 });

    var inputGradient = layer.Backward(outputGradient);

    AssertMatrixEqual("DenseLayer weight gradients",
        layer.WeightGradients,
        new Matrix(new double[,]
        {
            { 0.2, 0.3 },
            { 0.4, 0.6 }
        }));

    AssertVectorEqual("DenseLayer bias gradients",
        layer.BiasGradients,
        new Vector(new[] { 0.1, 0.2 }));

    AssertVectorEqual("DenseLayer input gradients",
        inputGradient,
        new Vector(new[] { 0.7, 1.0 }));

    Console.WriteLine("Phase 2.2 backpropagation checks passed.");
    Console.WriteLine();
}

static void RunPhase23Checks()
{
    var network = new NeuralNetwork();

    var firstLayer = new DenseLayer(2, 2);
    firstLayer.Weights[0, 0] = 1.0;
    firstLayer.Weights[0, 1] = 2.0;
    firstLayer.Weights[1, 0] = 3.0;
    firstLayer.Weights[1, 1] = 4.0;

    var secondLayer = new DenseLayer(2, 1);
    secondLayer.Weights[0, 0] = 5.0;
    secondLayer.Weights[0, 1] = 6.0;

    network.Add(firstLayer);
    network.Add(secondLayer);

    AssertEqual(
        "NeuralNetwork layer count",
        network.Layers.Count,
        2);

    var input = new Vector(new[] { 1.0, 2.0 });

    var output = network.Forward(input);

    AssertVectorEqual(
        "NeuralNetwork forward propagation",
        output,
        new Vector(new[] { 91.0 }));

    var outputGradient = new Vector(new[] { 1.0 });

    var inputGradient = network.Backward(outputGradient);

    AssertVectorEqual(
        "NeuralNetwork backward propagation",
        inputGradient,
        new Vector(new[] { 23.0, 34.0 }));

    AssertMatrixEqual(
        "NeuralNetwork first layer gradients",
        firstLayer.WeightGradients,
        new Matrix(new double[,]
        {
            { 5.0, 10.0 },
            { 6.0, 12.0 }
        }));

    AssertMatrixEqual(
        "NeuralNetwork second layer gradients",
        secondLayer.WeightGradients,
        new Matrix(new double[,]
        {
            { 5.0, 11.0 }
        }));

    AssertVectorEqual(
        "NeuralNetwork first layer bias gradients",
        firstLayer.BiasGradients,
        new Vector(new[] { 5.0, 6.0 }));

    AssertVectorEqual(
        "NeuralNetwork second layer bias gradients",
        secondLayer.BiasGradients,
        new Vector(new[] { 1.0 }));

    Console.WriteLine("Phase 2.3 neural network checks passed.");
    Console.WriteLine();
}

static void RunPhase31Checks()
{
    var reluLayer = new DenseLayer(2, 2, new ReLU());

    reluLayer.Weights[0, 0] = 1.0;
    reluLayer.Weights[0, 1] = -1.0;
    reluLayer.Weights[1, 0] = -2.0;
    reluLayer.Weights[1, 1] = 1.0;

    var input = new Vector(new[] { 2.0, 1.0 });

    var output = reluLayer.Forward(input);

    // z1 = 2 - 1 = 1  -> ReLU = 1
    // z2 = -4 + 1 = -3 -> ReLU = 0
    AssertVectorEqual(
        "DenseLayer ReLU forward",
        output,
        new Vector(new[] { 1.0, 0.0 }));

    // Incoming gradient is [1, 1].
    // ReLU derivative keeps gradient for z1
    // and removes it for z2.
    var inputGradient = reluLayer.Backward(
        new Vector(new[] { 1.0, 1.0 }));

    AssertVectorEqual(
        "DenseLayer ReLU backward",
        inputGradient,
        new Vector(new[] { 1.0, -1.0 }));

    AssertMatrixEqual(
        "DenseLayer ReLU weight gradients",
        reluLayer.WeightGradients,
        new Matrix(new double[,]
        {
            { 2.0, 1.0 },
            { 0.0, 0.0 }
        }));

    AssertVectorEqual(
        "DenseLayer ReLU bias gradients",
        reluLayer.BiasGradients,
        new Vector(new[] { 1.0, 0.0 }));

    var sigmoidLayer = new DenseLayer(1, 1, new Sigmoid());

    sigmoidLayer.Weights[0, 0] = 0.0;
    sigmoidLayer.Bias[0] = 0.0;

    var sigmoidOutput = sigmoidLayer.Forward(
        new Vector(new[] { 2.0 }));

    AssertApproximately(
        "DenseLayer Sigmoid forward",
        sigmoidOutput[0],
        0.5);

    var sigmoidInputGradient = sigmoidLayer.Backward(
        new Vector(new[] { 1.0 }));

    AssertApproximately(
        "DenseLayer Sigmoid backward",
        sigmoidInputGradient[0],
        0.0);

    Console.WriteLine("Phase 3.1 activation-aware dense layer checks passed.");
    Console.WriteLine();
}

static void AssertEqual(string name, int actual, int expected)
{
    if (actual != expected)
    {
        throw new InvalidOperationException(
            $"{name} failed. Expected {expected}, received {actual}.");
    }

    Console.WriteLine($"PASS: {name}");
}

static void AssertApproximately(
    string name,
    double actual,
    double expected,
    double tolerance = 1e-10)
{
    if (Math.Abs(actual - expected) > tolerance)
    {
        throw new InvalidOperationException(
            $"{name} failed. Expected {expected}, received {actual}.");
    }

    Console.WriteLine($"PASS: {name}");
}

static void AssertVectorEqual(
    string name,
    Vector actual,
    Vector expected,
    double tolerance = 1e-10)
{
    if (actual.Length != expected.Length)
    {
        throw new InvalidOperationException(
            $"{name} failed. Vector lengths differ.");
    }

    for (var i = 0; i < actual.Length; i++)
    {
        if (Math.Abs(actual[i] - expected[i]) > tolerance)
        {
            throw new InvalidOperationException(
                $"{name} failed at index {i}. " +
                $"Expected {expected[i]}, received {actual[i]}.");
        }
    }

    Console.WriteLine($"PASS: {name}");
}

static void AssertMatrixEqual(
    string name,
    Matrix actual,
    Matrix expected,
    double tolerance = 1e-10)
{
    if (actual.Rows != expected.Rows ||
        actual.Columns != expected.Columns)
    {
        throw new InvalidOperationException(
            $"{name} failed. Matrix dimensions differ.");
    }

    for (var row = 0; row < actual.Rows; row++)
    {
        for (var column = 0; column < actual.Columns; column++)
        {
            if (Math.Abs(actual[row, column] - expected[row, column]) > tolerance)
            {
                throw new InvalidOperationException(
                    $"{name} failed at [{row}, {column}]. " +
                    $"Expected {expected[row, column]}, " +
                    $"received {actual[row, column]}.");
            }
        }
    }

    Console.WriteLine($"PASS: {name}");
}

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
RunPhase32Checks();
RunPhase41Checks();
RunPhase51Checks();
RunPhase61Checks();
RunPhase71Checks();

Console.WriteLine();
Console.WriteLine("All Phase 7 checks passed.");

static void RunPhase11Checks()
{
    var left = new Vector(new[] { 1.0, 2.0, 3.0 });
    var right = new Vector(new[] { 4.0, 5.0, 6.0 });

    AssertVectorEqual(
        "Vector addition",
        left + right,
        new Vector(new[] { 5.0, 7.0, 9.0 }));

    AssertVectorEqual(
        "Vector subtraction",
        right - left,
        new Vector(new[] { 3.0, 3.0, 3.0 }));

    AssertVectorEqual(
        "Vector scalar multiplication",
        left * 2.0,
        new Vector(new[] { 2.0, 4.0, 6.0 }));

    AssertApproximately(
        "Vector dot product",
        Vector.Dot(left, right),
        32.0);

    var matrix = new Matrix(new double[,]
    {
        { 1.0, 2.0 },
        { 3.0, 4.0 }
    });

    AssertVectorEqual(
        "Matrix × Vector",
        matrix * new Vector(new[] { 5.0, 6.0 }),
        new Vector(new[] { 17.0, 39.0 }));

    var matrixRight = new Matrix(new double[,]
    {
        { 5.0, 6.0 },
        { 7.0, 8.0 }
    });

    AssertMatrixEqual(
        "Matrix × Matrix",
        matrix * matrixRight,
        new Matrix(new double[,]
        {
            { 19.0, 22.0 },
            { 43.0, 50.0 }
        }));

    AssertMatrixEqual(
        "Matrix transpose",
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

static void RunPhase21Checks()
{
    var layer = new DenseLayer(2, 3);

    AssertEqual(
        "DenseLayer input size",
        layer.Weights.Columns,
        2);

    AssertEqual(
        "DenseLayer output size",
        layer.Weights.Rows,
        3);

    AssertEqual(
        "DenseLayer bias size",
        layer.Bias.Length,
        3);

    var output = layer.Forward(
        new Vector(new[] { 2.0, 3.0 }));

    AssertVectorEqual(
        "DenseLayer forward with zero weights",
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

    var forward = layer.Forward(
        new Vector(new[] { 2.0, 3.0 }));

    AssertVectorEqual(
        "DenseLayer forward",
        forward,
        new Vector(new[] { 8.5, 19.0 }));

    var inputGradient = layer.Backward(
        new Vector(new[] { 0.1, 0.2 }));

    AssertMatrixEqual(
        "DenseLayer weight gradients",
        layer.WeightGradients,
        new Matrix(new double[,]
        {
            { 0.2, 0.3 },
            { 0.4, 0.6 }
        }));

    AssertVectorEqual(
        "DenseLayer bias gradients",
        layer.BiasGradients,
        new Vector(new[] { 0.1, 0.2 }));

    AssertVectorEqual(
        "DenseLayer input gradients",
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

    var output = network.Forward(
        new Vector(new[] { 1.0, 2.0 }));

    AssertVectorEqual(
        "NeuralNetwork forward propagation",
        output,
        new Vector(new[] { 91.0 }));

    var inputGradient = network.Backward(
        new Vector(new[] { 1.0 }));

    AssertVectorEqual(
        "NeuralNetwork backward propagation",
        inputGradient,
        new Vector(new[] { 23.0, 34.0 }));

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

    var output = reluLayer.Forward(
        new Vector(new[] { 2.0, 1.0 }));

    AssertVectorEqual(
        "DenseLayer ReLU forward",
        output,
        new Vector(new[] { 1.0, 0.0 }));

    var inputGradient = reluLayer.Backward(
        new Vector(new[] { 1.0, 1.0 }));

    AssertVectorEqual(
        "DenseLayer ReLU backward",
        inputGradient,
        new Vector(new[] { 1.0, -1.0 }));

    var sigmoidLayer = new DenseLayer(
        1,
        1,
        new Sigmoid());

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

    Console.WriteLine(
        "Phase 3.1 activation-aware dense layer checks passed.");

    Console.WriteLine();
}

static void RunPhase32Checks()
{
    var network = CreateFinalNetwork();

    AssertEqual(
        "Final network layer count",
        network.Layers.Count,
        5);

    AssertEqual(
        "Input layer size",
        network.Layers[0].Weights.Columns,
        4);

    AssertEqual(
        "Hidden layer 1 size",
        network.Layers[0].Weights.Rows,
        8);

    AssertEqual(
        "Hidden layer 2 size",
        network.Layers[1].Weights.Rows,
        8);

    AssertEqual(
        "Hidden layer 3 size",
        network.Layers[2].Weights.Rows,
        6);

    AssertEqual(
        "Hidden layer 4 size",
        network.Layers[3].Weights.Rows,
        4);

    AssertEqual(
        "Output layer size",
        network.Layers[4].Weights.Rows,
        1);

    var output = network.Forward(
        new Vector(new[] { 1.0, 0.5, -1.0, 2.0 }));

    AssertEqual(
        "Final network output size",
        output.Length,
        1);

    if (output[0] < 0.0 || output[0] > 1.0)
    {
        throw new InvalidOperationException(
            "Final network output is outside [0, 1].");
    }

    Console.WriteLine(
        $"Final network output: {output[0]:F6}");

    Console.WriteLine(
        "Phase 3.2 final network architecture checks passed.");

    Console.WriteLine();
}

static void RunPhase41Checks()
{
    const double epsilon = 1e-5;
    const double tolerance = 1e-4;

    var input = new Vector(new[]
    {
        0.5,
        0.75,
        1.0,
        1.25
    });

    const double target = 1.0;

    var lossFunction = new BinaryCrossEntropy();
    var network = CreateGradientCheckNetwork();

    var prediction = network.Forward(input);

    var loss = lossFunction.Forward(
        prediction[0],
        target);

    var predictionGradient =
        lossFunction.Derivative(
            prediction[0],
            target);

    network.Backward(
        new Vector(new[] { predictionGradient }));

    var checkedWeights = 0;

    for (var layerIndex = 0;
         layerIndex < network.Layers.Count;
         layerIndex++)
    {
        var layer = network.Layers[layerIndex];

        for (var row = 0; row < layer.Weights.Rows; row++)
        {
            for (var column = 0;
                 column < layer.Weights.Columns;
                 column++)
            {
                var original = layer.Weights[row, column];

                layer.Weights[row, column] =
                    original + epsilon;

                var positiveLoss = EvaluateLoss(
                    network,
                    lossFunction,
                    input,
                    target);

                layer.Weights[row, column] =
                    original - epsilon;

                var negativeLoss = EvaluateLoss(
                    network,
                    lossFunction,
                    input,
                    target);

                layer.Weights[row, column] =
                    original;

                var numericalGradient =
                    (positiveLoss - negativeLoss) /
                    (2.0 * epsilon);

                AssertApproximately(
                    $"Gradient L{layerIndex + 1} W[{row},{column}]",
                    layer.WeightGradients[row, column],
                    numericalGradient,
                    tolerance);

                checkedWeights++;
            }
        }
    }

    Console.WriteLine(
        $"Gradient checking passed for {checkedWeights} weights.");

    Console.WriteLine(
        "Phase 4.1 numerical gradient checking passed.");

    Console.WriteLine();
}

static void RunPhase51Checks()
{
    var layer = new DenseLayer(
        1,
        1,
        new Sigmoid());

    layer.Weights[0, 0] = 0.0;
    layer.Bias[0] = 0.0;

    var input = new Vector(new[] { 1.0 });
    const double target = 1.0;

    var lossFunction = new BinaryCrossEntropy();

    var predictionBefore =
        layer.Forward(input);

    var lossBefore =
        lossFunction.Forward(
            predictionBefore[0],
            target);

    var predictionGradient =
        lossFunction.Derivative(
            predictionBefore[0],
            target);

    layer.Backward(
        new Vector(new[] { predictionGradient }));

    layer.UpdateParameters(0.1);

    var predictionAfter =
        layer.Forward(input);

    var lossAfter =
        lossFunction.Forward(
            predictionAfter[0],
            target);

    AssertTrue(
        "Loss decreased after gradient descent",
        lossAfter < lossBefore);

    Console.WriteLine("Phase 5.1 gradient descent checks passed.");
    Console.WriteLine();
}

static void RunPhase61Checks()
{
    var network = CreateSmallTrainingNetwork();
    var lossFunction = new BinaryCrossEntropy();

    var trainingInputs = new[]
    {
        new Vector(new[] { 0.0, 0.0 }),
        new Vector(new[] { 0.0, 1.0 }),
        new Vector(new[] { 1.0, 0.0 }),
        new Vector(new[] { 1.0, 1.0 })
    };

    var trainingTargets = new[]
    {
        0.0,
        1.0,
        1.0,
        1.0
    };

    const int epochs = 100;
    const double learningRate = 0.1;

    var initialLoss = CalculateDatasetLoss(
        network,
        lossFunction,
        trainingInputs,
        trainingTargets);

    var finalLoss = initialLoss;

    for (var epoch = 1; epoch <= epochs; epoch++)
    {
        for (var sample = 0;
             sample < trainingInputs.Length;
             sample++)
        {
            var prediction =
                network.Forward(trainingInputs[sample]);

            var gradient =
                lossFunction.Derivative(
                    prediction[0],
                    trainingTargets[sample]);

            network.Backward(
                new Vector(new[] { gradient }));

            foreach (var layer in network.Layers)
            {
                layer.UpdateParameters(learningRate);
            }
        }

        finalLoss = CalculateDatasetLoss(
            network,
            lossFunction,
            trainingInputs,
            trainingTargets);
    }

    AssertTrue(
        "Training reduced dataset loss",
        finalLoss < initialLoss);

    Console.WriteLine(
        $"Training initial loss: {initialLoss:F6}");

    Console.WriteLine(
        $"Training final loss:   {finalLoss:F6}");

    Console.WriteLine(
        "Phase 6.1 training loop checks passed.");

    Console.WriteLine();
}

static void RunPhase71Checks()
{
    var dataset = CreateClassificationDataset();

    var network = CreateFinalNetworkForDataset();

    Train(
        network,
        dataset.TrainingInputs,
        dataset.TrainingTargets,
        epochs: 2000, learningRate: 0.05);

    var accuracy = CalculateAccuracy(
        network,
        dataset.TestInputs,
        dataset.TestTargets);

    Console.WriteLine();
    Console.WriteLine("========== TEST EVALUATION ==========");
    Console.WriteLine(
        $"Test samples: {dataset.TestInputs.Length}");
    Console.WriteLine(
        $"Correct:      {CountCorrectPredictions(
            network,
            dataset.TestInputs,
            dataset.TestTargets)}");
    Console.WriteLine(
        $"Accuracy:     {accuracy:P2}");
    Console.WriteLine();

    for (var i = 0;
         i < dataset.TestInputs.Length;
         i++)
    {
        var prediction =
            network.Forward(dataset.TestInputs[i])[0];

        var predictedClass =
            prediction >= 0.5 ? 1 : 0;

        Console.WriteLine(
            $"Test {i + 1}: " +
            $"expected={dataset.TestTargets[i]:0} " +
            $"predicted={predictedClass} " +
            $"score={prediction:F4}");
    }

    AssertTrue(
        "Test accuracy is at least 75%",
        accuracy >= 1.0);

    Console.WriteLine();
    Console.WriteLine(
        "Phase 7.1 train/test evaluation checks passed.");
    Console.WriteLine();
}

static void Train(
    NeuralNetwork network,
    Vector[] inputs,
    double[] targets,
    int epochs,
    double learningRate)
{
    var lossFunction = new BinaryCrossEntropy();

    for (var epoch = 1; epoch <= epochs; epoch++)
    {
        for (var sample = 0;
             sample < inputs.Length;
             sample++)
        {
            var prediction =
                network.Forward(inputs[sample]);

            var gradient =
                lossFunction.Derivative(
                    prediction[0],
                    targets[sample]);

            network.Backward(
                new Vector(new[] { gradient }));

            foreach (var layer in network.Layers)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }
}

static double CalculateAccuracy(
    NeuralNetwork network,
    Vector[] inputs,
    double[] targets)
{
    return (double)CountCorrectPredictions(
        network,
        inputs,
        targets) / inputs.Length;
}

static int CountCorrectPredictions(
    NeuralNetwork network,
    Vector[] inputs,
    double[] targets)
{
    var correct = 0;

    for (var i = 0; i < inputs.Length; i++)
    {
        var score =
            network.Forward(inputs[i])[0];

        var predictedClass =
            score >= 0.5 ? 1 : 0;

        var expectedClass =
            targets[i] >= 0.5 ? 1 : 0;

        if (predictedClass == expectedClass)
        {
            correct++;
        }
    }

    return correct;
}

static ClassificationDataset CreateClassificationDataset()
{
    return new ClassificationDataset(
        new[]
        {
            // Training class 0
            new Vector(new[] { 0.05, 0.05, 0.05, 0.05 }),
            new Vector(new[] { 0.10, 0.05, 0.10, 0.05 }),
            new Vector(new[] { 0.05, 0.10, 0.05, 0.10 }),
            new Vector(new[] { 0.10, 0.10, 0.05, 0.05 }),
            new Vector(new[] { 0.05, 0.05, 0.10, 0.10 }),
            new Vector(new[] { 0.10, 0.05, 0.05, 0.10 }),

            // Training class 1
            new Vector(new[] { 0.90, 0.90, 0.90, 0.90 }),
            new Vector(new[] { 0.85, 0.90, 0.85, 0.90 }),
            new Vector(new[] { 0.90, 0.85, 0.90, 0.85 }),
            new Vector(new[] { 0.85, 0.85, 0.90, 0.90 }),
            new Vector(new[] { 0.90, 0.90, 0.85, 0.85 }),
            new Vector(new[] { 0.85, 0.90, 0.90, 0.85 })
        },
        new[]
        {
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        },
        new[]
        {
            // Unseen class 0
            new Vector(new[] { 0.15, 0.10, 0.15, 0.10 }),
            new Vector(new[] { 0.20, 0.15, 0.10, 0.15 }),

            // Unseen class 1
            new Vector(new[] { 0.80, 0.85, 0.80, 0.85 }),
            new Vector(new[] { 0.75, 0.80, 0.85, 0.80 })
        },
        new[]
        {
            0.0,
            0.0,
            1.0,
            1.0
        });
}

static NeuralNetwork CreateFinalNetworkForDataset()
{
    var network = CreateFinalNetwork();

    InitializeNetworkForTraining(network);

    return network;
}

static void InitializeNetworkForTraining(
    NeuralNetwork network)
{
    for (var layerIndex = 0;
         layerIndex < network.Layers.Count;
         layerIndex++)
    {
        var layer = network.Layers[layerIndex];

        for (var row = 0;
             row < layer.Weights.Rows;
             row++)
        {
            for (var column = 0;
                 column < layer.Weights.Columns;
                 column++)
            {
                layer.Weights[row, column] = 0.05;
            }
        }

        for (var i = 0;
             i < layer.Bias.Length;
             i++)
        {
            layer.Bias[i] = 0.05;
        }
    }
}

static NeuralNetwork CreateSmallTrainingNetwork()
{
    var network = new NeuralNetwork();

    var first = new DenseLayer(
        2,
        2,
        new ReLU());

    var second = new DenseLayer(
        2,
        1,
        new Sigmoid());

    first.Weights[0, 0] = 0.1;
    first.Weights[0, 1] = 0.2;
    first.Weights[1, 0] = 0.2;
    first.Weights[1, 1] = 0.1;

    first.Bias[0] = 0.01;
    first.Bias[1] = 0.01;

    second.Weights[0, 0] = 0.2;
    second.Weights[0, 1] = 0.2;
    second.Bias[0] = 0.01;

    network.Add(first);
    network.Add(second);

    return network;
}

static NeuralNetwork CreateFinalNetwork()
{
    var network = new NeuralNetwork();

    network.Add(new DenseLayer(4, 8, new ReLU()));
    network.Add(new DenseLayer(8, 8, new ReLU()));
    network.Add(new DenseLayer(8, 6, new ReLU()));
    network.Add(new DenseLayer(6, 4, new ReLU()));
    network.Add(new DenseLayer(4, 1, new Sigmoid()));

    return network;
}

static NeuralNetwork CreateGradientCheckNetwork()
{
    var network = CreateFinalNetwork();

    for (var layerIndex = 0;
         layerIndex < network.Layers.Count;
         layerIndex++)
    {
        var layer = network.Layers[layerIndex];

        for (var row = 0;
             row < layer.Weights.Rows;
             row++)
        {
            for (var column = 0;
                 column < layer.Weights.Columns;
                 column++)
            {
                layer.Weights[row, column] =
                    0.01 +
                    (0.001 * (layerIndex + 1)) +
                    (0.0001 * (row + column));
            }
        }

        for (var i = 0;
             i < layer.Bias.Length;
             i++)
        {
            layer.Bias[i] = 0.1;
        }
    }

    return network;
}

static double CalculateDatasetLoss(
    NeuralNetwork network,
    BinaryCrossEntropy lossFunction,
    Vector[] inputs,
    double[] targets)
{
    var total = 0.0;

    for (var i = 0; i < inputs.Length; i++)
    {
        var prediction =
            network.Forward(inputs[i]);

        total += lossFunction.Forward(
            prediction[0],
            targets[i]);
    }

    return total / inputs.Length;
}

static double EvaluateLoss(
    NeuralNetwork network,
    BinaryCrossEntropy lossFunction,
    Vector input,
    double target)
{
    var prediction =
        network.Forward(input);

    return lossFunction.Forward(
        prediction[0],
        target);
}

static void AssertTrue(
    string name,
    bool condition)
{
    if (!condition)
    {
        throw new InvalidOperationException(
            $"{name} failed.");
    }

    Console.WriteLine($"PASS: {name}");
}

static void AssertEqual(
    string name,
    int actual,
    int expected)
{
    if (actual != expected)
    {
        throw new InvalidOperationException(
            $"{name} failed. " +
            $"Expected {expected}, received {actual}.");
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
            $"{name} failed. " +
            $"Expected {expected}, received {actual}.");
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
        if (Math.Abs(
                actual[i] -
                expected[i]) > tolerance)
        {
            throw new InvalidOperationException(
                $"{name} failed at index {i}. " +
                $"Expected {expected[i]}, " +
                $"received {actual[i]}.");
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

    for (var row = 0;
         row < actual.Rows;
         row++)
    {
        for (var column = 0;
             column < actual.Columns;
             column++)
        {
            if (Math.Abs(
                    actual[row, column] -
                    expected[row, column]) > tolerance)
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

sealed record ClassificationDataset(
    Vector[] TrainingInputs,
    double[] TrainingTargets,
    Vector[] TestInputs,
    double[] TestTargets);














using SharpNeuralNetwork.ActivationFunctions;
using SharpNeuralNetwork.LossFunctions;
using SharpNeuralNetwork.Mathematics;
using SharpNeuralNetwork.NeuralNetworks;

const int seed = 42;
const int epochs = 2000;
const double learningRate = 0.05;

var dataset = CreateDataset();
var network = CreateNetwork();
var lossFunction = new BinaryCrossEntropy();

network.InitializeWeights(seed);

var initialLoss = CalculateAverageLoss(
    network,
    lossFunction,
    dataset.TrainingInputs,
    dataset.TrainingTargets);

Train(
    network,
    lossFunction,
    dataset.TrainingInputs,
    dataset.TrainingTargets,
    epochs,
    learningRate);

var finalLoss = CalculateAverageLoss(
    network,
    lossFunction,
    dataset.TrainingInputs,
    dataset.TrainingTargets);

var accuracy = CalculateAccuracy(
    network,
    dataset.TestInputs,
    dataset.TestTargets);

Console.WriteLine("SharpNeuralNetwork");
Console.WriteLine("====================");
Console.WriteLine($"Seed:          {seed}");
Console.WriteLine($"Epochs:        {epochs}");
Console.WriteLine($"Learning rate: {learningRate:F3}");
Console.WriteLine($"Training loss: {initialLoss:F6} -> {finalLoss:F6}");
Console.WriteLine();
Console.WriteLine("Held-out synthetic test set");
Console.WriteLine($"Accuracy: {accuracy:P0} ({CountCorrect(network, dataset.TestInputs, dataset.TestTargets)}/{dataset.TestInputs.Length})");

for (var index = 0; index < dataset.TestInputs.Length; index++)
{
    var score = network.Forward(dataset.TestInputs[index])[0];
    var predictedClass = score >= 0.5 ? 1 : 0;
    Console.WriteLine(
        $"Sample {index + 1}: expected={dataset.TestTargets[index]:0}, predicted={predictedClass}, score={score:F4}");
}

if (finalLoss >= initialLoss)
{
    throw new InvalidOperationException("Training did not reduce the training loss.");
}

if (accuracy < 0.75)
{
    throw new InvalidOperationException("The demonstration did not reach its 75% sanity-check threshold.");
}

Console.WriteLine();
Console.WriteLine("Demo completed successfully.");
Console.WriteLine("Note: this tiny synthetic dataset is a pipeline sanity check, not a real-world benchmark.");

static NeuralNetwork CreateNetwork()
{
    var network = new NeuralNetwork();

    network.Add(new DenseLayer(4, 8, new ReLU()));
    network.Add(new DenseLayer(8, 8, new ReLU()));
    network.Add(new DenseLayer(8, 6, new ReLU()));
    network.Add(new DenseLayer(6, 4, new ReLU()));
    network.Add(new DenseLayer(4, 1, new Sigmoid()));

    return network;
}

static void Train(
    NeuralNetwork network,
    BinaryCrossEntropy lossFunction,
    Vector[] inputs,
    double[] targets,
    int epochs,
    double learningRate)
{
    for (var epoch = 0; epoch < epochs; epoch++)
    {
        for (var sample = 0; sample < inputs.Length; sample++)
        {
            var prediction = network.Forward(inputs[sample]);
            var lossGradient = lossFunction.Derivative(prediction[0], targets[sample]);

            network.Backward(new Vector(new[] { lossGradient }));

            foreach (var layer in network.Layers)
            {
                layer.UpdateParameters(learningRate);
            }
        }
    }
}

static double CalculateAverageLoss(
    NeuralNetwork network,
    BinaryCrossEntropy lossFunction,
    Vector[] inputs,
    double[] targets)
{
    var totalLoss = 0.0;

    for (var index = 0; index < inputs.Length; index++)
    {
        var prediction = network.Forward(inputs[index])[0];
        totalLoss += lossFunction.Forward(prediction, targets[index]);
    }

    return totalLoss / inputs.Length;
}

static double CalculateAccuracy(
    NeuralNetwork network,
    Vector[] inputs,
    double[] targets)
{
    return (double)CountCorrect(network, inputs, targets) / inputs.Length;
}

static int CountCorrect(
    NeuralNetwork network,
    Vector[] inputs,
    double[] targets)
{
    var correct = 0;

    for (var index = 0; index < inputs.Length; index++)
    {
        var prediction = network.Forward(inputs[index])[0];
        var predictedClass = prediction >= 0.5 ? 1.0 : 0.0;

        if (predictedClass == targets[index])
        {
            correct++;
        }
    }

    return correct;
}

static ClassificationDataset CreateDataset()
{
    return new ClassificationDataset(
        new[]
        {
            // Training examples for class 0
            new Vector(new[] { 0.05, 0.05, 0.05, 0.05 }),
            new Vector(new[] { 0.10, 0.05, 0.10, 0.05 }),
            new Vector(new[] { 0.05, 0.10, 0.05, 0.10 }),
            new Vector(new[] { 0.10, 0.10, 0.05, 0.05 }),
            new Vector(new[] { 0.05, 0.05, 0.10, 0.10 }),
            new Vector(new[] { 0.10, 0.05, 0.05, 0.10 }),

            // Training examples for class 1
            new Vector(new[] { 0.90, 0.90, 0.90, 0.90 }),
            new Vector(new[] { 0.85, 0.90, 0.85, 0.90 }),
            new Vector(new[] { 0.90, 0.85, 0.90, 0.85 }),
            new Vector(new[] { 0.85, 0.85, 0.90, 0.90 }),
            new Vector(new[] { 0.90, 0.90, 0.85, 0.85 }),
            new Vector(new[] { 0.85, 0.90, 0.90, 0.85 })
        },
        new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },
        new[]
        {
            new Vector(new[] { 0.15, 0.10, 0.15, 0.10 }),
            new Vector(new[] { 0.20, 0.15, 0.10, 0.15 }),
            new Vector(new[] { 0.80, 0.85, 0.80, 0.85 }),
            new Vector(new[] { 0.75, 0.80, 0.85, 0.80 })
        },
        new[] { 0.0, 0.0, 1.0, 1.0 });
}

sealed record ClassificationDataset(
    Vector[] TrainingInputs,
    double[] TrainingTargets,
    Vector[] TestInputs,
    double[] TestTargets);

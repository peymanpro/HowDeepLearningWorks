using System;

namespace HowDeepLearningWorks.Mathematics;

/// <summary>
/// Represents a one-dimensional vector of double-precision values.
/// </summary>
public sealed class Vector
{
    private readonly double[] _values;

    /// <summary>
    /// Gets the number of elements in the vector.
    /// </summary>
    public int Length => _values.Length;

    /// <summary>
    /// Initializes a vector from the supplied values.
    /// </summary>
    /// <param name="values">The vector values.</param>
    public Vector(IEnumerable<double> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        _values = values.ToArray();
    }

    /// <summary>
    /// Initializes a zero-filled vector.
    /// </summary>
    /// <param name="length">The vector length.</param>
    public Vector(int length)
    {
        if (length < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        _values = new double[length];
    }

    /// <summary>
    /// Gets or sets an element.
    /// </summary>
    public double this[int index]
    {
        get => _values[index];
        set => _values[index] = value;
    }

    /// <summary>
    /// Returns a copy of the underlying values.
    /// </summary>
    public double[] ToArray() => (double[])_values.Clone();

    /// <summary>
    /// Adds two vectors element by element.
    /// </summary>
    public static Vector operator +(Vector left, Vector right)
    {
        ValidateSameLength(left, right);

        var result = new Vector(left.Length);

        for (var i = 0; i < left.Length; i++)
        {
            result[i] = left[i] + right[i];
        }

        return result;
    }

    /// <summary>
    /// Subtracts two vectors element by element.
    /// </summary>
    public static Vector operator -(Vector left, Vector right)
    {
        ValidateSameLength(left, right);

        var result = new Vector(left.Length);

        for (var i = 0; i < left.Length; i++)
        {
            result[i] = left[i] - right[i];
        }

        return result;
    }

    /// <summary>
    /// Multiplies every element by a scalar.
    /// </summary>
    public static Vector operator *(Vector vector, double scalar)
    {
        ArgumentNullException.ThrowIfNull(vector);

        var result = new Vector(vector.Length);

        for (var i = 0; i < vector.Length; i++)
        {
            result[i] = vector[i] * scalar;
        }

        return result;
    }

    /// <summary>
    /// Computes the dot product of two vectors.
    /// </summary>
    public static double Dot(Vector left, Vector right)
    {
        ValidateSameLength(left, right);

        var result = 0.0;

        for (var i = 0; i < left.Length; i++)
        {
            result += left[i] * right[i];
        }

        return result;
    }

    private static void ValidateSameLength(Vector left, Vector right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Length != right.Length)
        {
            throw new ArgumentException("Vectors must have the same length.");
        }
    }
}

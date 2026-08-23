using System;

namespace HowDeepLearningWorks.Mathematics;

/// <summary>
/// Represents a two-dimensional matrix of double-precision values.
/// </summary>
public sealed class Matrix
{
    private readonly double[,] _values;

    /// <summary>
    /// Gets the number of rows.
    /// </summary>
    public int Rows { get; }

    /// <summary>
    /// Gets the number of columns.
    /// </summary>
    public int Columns { get; }

    /// <summary>
    /// Initializes a zero-filled matrix.
    /// </summary>
    public Matrix(int rows, int columns)
    {
        if (rows < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rows));
        }

        if (columns < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(columns));
        }

        Rows = rows;
        Columns = columns;
        _values = new double[rows, columns];
    }

    /// <summary>
    /// Initializes a matrix from a rectangular two-dimensional array.
    /// </summary>
    public Matrix(double[,] values)
    {
        ArgumentNullException.ThrowIfNull(values);

        Rows = values.GetLength(0);
        Columns = values.GetLength(1);
        _values = (double[,])values.Clone();
    }

    /// <summary>
    /// Gets or sets an element.
    /// </summary>
    public double this[int row, int column]
    {
        get => _values[row, column];
        set => _values[row, column] = value;
    }

    /// <summary>
    /// Adds two matrices element by element.
    /// </summary>
    public static Matrix operator +(Matrix left, Matrix right)
    {
        ValidateSameShape(left, right);

        var result = new Matrix(left.Rows, left.Columns);

        for (var row = 0; row < left.Rows; row++)
        {
            for (var column = 0; column < left.Columns; column++)
            {
                result[row, column] = left[row, column] + right[row, column];
            }
        }

        return result;
    }

    /// <summary>
    /// Subtracts two matrices element by element.
    /// </summary>
    public static Matrix operator -(Matrix left, Matrix right)
    {
        ValidateSameShape(left, right);

        var result = new Matrix(left.Rows, left.Columns);

        for (var row = 0; row < left.Rows; row++)
        {
            for (var column = 0; column < left.Columns; column++)
            {
                result[row, column] = left[row, column] - right[row, column];
            }
        }

        return result;
    }

    /// <summary>
    /// Multiplies every matrix element by a scalar.
    /// </summary>
    public static Matrix operator *(Matrix matrix, double scalar)
    {
        ArgumentNullException.ThrowIfNull(matrix);

        var result = new Matrix(matrix.Rows, matrix.Columns);

        for (var row = 0; row < matrix.Rows; row++)
        {
            for (var column = 0; column < matrix.Columns; column++)
            {
                result[row, column] = matrix[row, column] * scalar;
            }
        }

        return result;
    }

    /// <summary>
    /// Multiplies a matrix by a vector.
    /// </summary>
    public static Vector operator *(Matrix matrix, Vector vector)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ArgumentNullException.ThrowIfNull(vector);

        if (matrix.Columns != vector.Length)
        {
            throw new ArgumentException(
                "The matrix column count must equal the vector length.");
        }

        var result = new Vector(matrix.Rows);

        for (var row = 0; row < matrix.Rows; row++)
        {
            var sum = 0.0;

            for (var column = 0; column < matrix.Columns; column++)
            {
                sum += matrix[row, column] * vector[column];
            }

            result[row] = sum;
        }

        return result;
    }

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    public static Matrix operator *(Matrix left, Matrix right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Columns != right.Rows)
        {
            throw new ArgumentException(
                "The left matrix column count must equal the right matrix row count.");
        }

        var result = new Matrix(left.Rows, right.Columns);

        for (var row = 0; row < left.Rows; row++)
        {
            for (var column = 0; column < right.Columns; column++)
            {
                var sum = 0.0;

                for (var k = 0; k < left.Columns; k++)
                {
                    sum += left[row, k] * right[k, column];
                }

                result[row, column] = sum;
            }
        }

        return result;
    }

    /// <summary>
    /// Returns the transpose of the matrix.
    /// </summary>
    public Matrix Transpose()
    {
        var result = new Matrix(Columns, Rows);

        for (var row = 0; row < Rows; row++)
        {
            for (var column = 0; column < Columns; column++)
            {
                result[column, row] = this[row, column];
            }
        }

        return result;
    }

    private static void ValidateSameShape(Matrix left, Matrix right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.Rows != right.Rows || left.Columns != right.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions.");
        }
    }
}

using System;
using System.Linq;
using System.Text;

namespace nnmnist.Networks.Graph
{
    internal class Matrix
    {
        // emulate two-dimensional array by a single-dimensional array for efficiency
        // storage is row first
        
        public readonly int ColDim;
        public readonly int RowDim;
        public readonly float[] Storage;

        // a rowdim * coldim matrix filled with 0
        public Matrix(int rowDim, int colDim)
        {
            RowDim = rowDim;
            ColDim = colDim;
            Storage = new float[rowDim * colDim];
        }

        // a rowdim * coldim matrix filled with elements from valuefactory
        public Matrix(int rowDim, int colDim, Func<float> valueFactory)
        {
            RowDim = rowDim;
            ColDim = colDim;
            Storage = new float[rowDim * colDim];
            for (var i = 0; i < Storage.Length; i++)
                Storage[i] = valueFactory();
        }

        // a matrix from the provided values
        public Matrix(float[] v, int rowDim, int colDim)
        {
            Storage = v;
            RowDim = rowDim;
            ColDim = colDim;
        }

        // indexer
        // can index as a multi-dimensional array
        public float this[int row, int column]
        {
            get => Storage[row * ColDim + column];
            set => Storage[row * ColDim + column] = value;
        }

        // a deep copy
        public Matrix Copy()
        {
            var m = new Matrix(RowDim, ColDim);
            for (var i = 0; i < Storage.Length; ++i)
                m.Storage[i] = Storage[i];
            return m;
        }

        // a new matrix of the same size
        public Matrix Empty()
        {
            return new Matrix(RowDim, ColDim);
        }

        // this += a*b
        public void AddMultiply(Matrix a, Matrix b)
        {
            var nRow = RowDim;
            var nCol = ColDim;
            var nDot = a.ColDim;
            for (var i = 0; i < nRow; i++)
            {
                for (var k = 0; k < nDot; k++)
                {
                    var fac = a[i, k];
                    for (var j = 0; j < nCol; j++)
                        this[i, j] += fac * b[k, j];
                }
            }
        }

        // this += a*b
        // only columns in inds of this is calculated
        public void AddMultiply(int[] inds, Matrix a, Matrix b)
        {
            var nRow = RowDim;
            //var nCol = ColDim;
            var nDot = a.ColDim;
            for (var i = 0; i < nRow; i++)
            {
                for (var k = 0; k < nDot; k++)
                {
                    var fac = a[i, k];
                    foreach (var j in inds)
                        this[i, j] += fac * b[k, j];
                }
            }
        }

        // this += a'*b
        public void AddMultiplyTransA(Matrix a, Matrix b)
        {
            var nRow = RowDim;
            var nCol = ColDim;
            var nDot = b.RowDim;
            for (var k = 0; k < nDot; k++)
            {
                for (var i = 0; i < nRow; i++)
                {
                    var fac = a[k, i];
                    for (var j = 0; j < nCol; j++)
                        this[i, j] += fac * b[k, j];
                }
            }
        }

        // this += a'*b
        // only elements in binds of b is calculated
        // (top-k backprop)
        public void AddMultiplyTransA(Matrix a, Matrix b, int[][] binds)
        {
            var nRow = RowDim;
            //var nCol = ColDim;
            var nDot = b.RowDim;
            for (var k = 0; k < nDot; k++)
            {
                var inds = binds[k];
                for (var i = 0; i < nRow; i++)
                {
                    var fac = a[k, i];
                    foreach (var j in inds)
                        //for (var j = 0; j < nCol; j++)
                        this[i, j] += fac * b[k, j];
                }
            }
        }


        // this += a'*b
        // only columns in binds of b is calculated
        // (masked neurons)
        public void AddMultiplyTransA(Matrix a, Matrix b, int[] binds)
        {
            var nRow = RowDim;
            //var nCol = ColDim;
            var nDot = b.RowDim;
            for (var k = 0; k < nDot; k++)
            {
                //var inds = binds[k];
                for (var i = 0; i < nRow; i++)
                {
                    var fac = a[k, i];
                    foreach (var j in binds)
                        //for (var j = 0; j < nCol; j++)
                        this[i, j] += fac * b[k, j];
                }
            }
        }

        // this += a*b'
        public void AddMultiplyTransB(Matrix a, Matrix b)
        {
            var nRow = RowDim;
            var nCol = ColDim;
            var nDot = a.ColDim;
            for (var i = 0; i < nRow; i++)
            {
                for (var j = 0; j < nCol; j++)
                {
                    var sum = 0f;
                    for (var k = 0; k < nDot; k++)
                        sum += a[i, k] * b[j, k];
                    this[i, j] += sum;
                }
            }
        }

        // this += a*b'
        // only elements in ainds of a is calculated
        // (top-k backprop)
        public void AddMultiplyTransB(Matrix a, int[][] ainds, Matrix b)
        {
            var nRow = RowDim;
            var nCol = ColDim;
            //var nDot = a.ColDim;
            for (var i = 0; i < nRow; i++)
            {
                var inds = ainds[i];
                for (var j = 0; j < nCol; j++)
                {
                    var sum = 0f;
                    foreach (var k in inds)
                        //for (var k = 0; k < nDot; k++)
                        sum += a[i, k] * b[j, k];
                    this[i, j] += sum;
                }
            }
        }

        // this += a*b'
        // only columns in ainds of a is calculated
        // (masked neurons)
        public void AddMultiplyTransB(Matrix a, int[] ainds, Matrix b)
        {
            var nRow = RowDim;
            var nCol = ColDim;
            //var nDot = a.ColDim;
            for (var i = 0; i < nRow; i++)
            {
                //var inds = ainds[i];
                for (var j = 0; j < nCol; j++)
                {
                    var sum = 0f;
                    foreach (var k in ainds)
                        //for (var k = 0; k < nDot; k++)
                        sum += a[i, k] * b[j, k];
                    this[i, j] += sum;
                }
            }
        }

        // return the max element
        public float Max()
        {
            return Storage.Max();
        }

        // return the max elment in the row
        public float Max(int row)
        {
            var max = float.NegativeInfinity;
            for (var j = 0; j < ColDim; j++)
                max = Math.Max(max, this[row, j]);
            return max;
        }

        //  return the max elment in the row if not masked
        public float Max(int row, int[] mask)
        {
            var maxInd = -1;
            for (var i = 0; i < ColDim; i++)
            {
                if (mask[i] >= 0 && (maxInd < 0 || this[row, i] > this[row, maxInd]))
                    maxInd = i;
            }
            return this[row, maxInd];
        }

        // return the sum
        public float Sum()
        {
            return Storage.Sum();
        }

        // return the index of the max element
        public int MaxIndex()
        {
            var mxv = Storage[0];
            var mxi = 0;
            for (var i = 1; i < Storage.Length; i++)
            {
                mxi = Storage[i] > mxv ? i : mxi;
                mxv = Storage[i] > mxv ? Storage[i] : mxv;
            }
            return mxi;
        }

        // return the index of the max element in the row
        public int MaxIndex(int row)
        {
            var mxv = this[row, 0];
            var mxi = 0;
            for (var i = 1; i < ColDim; i++)
            {
                var tmp = this[row, i];
                mxi = tmp > mxv ? i : mxi;
                mxv = tmp > mxv ? tmp : mxv;
            }
            return mxi;
        }

        // return the indices of the max element in the rows
        public int[] MaxIndices()
        {
            var res = new int[RowDim];
            for (var row = 0; row < RowDim; row++)
            {
                var mxv = this[row, 0];
                var mxi = 0;
                for (var i = 1; i < ColDim; i++)
                {
                    var tmp = this[row, i];
                    mxi = tmp > mxv ? i : mxi;
                    mxv = tmp > mxv ? tmp : mxv;
                }
            }
            return res;
        }

        // print the tensor weights
        // useful in debugging
        // can also cause the watch window being slow
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append($"[{RowDim}:{ColDim}]");
            foreach (var val in Storage)
                sb.Append($" {val:##.0000}");
            return sb.ToString();
        }
    }
}
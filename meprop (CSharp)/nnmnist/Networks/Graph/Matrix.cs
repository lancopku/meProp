using System;
using System.Linq;
using System.Text;

namespace nnmnist.Networks.Graph
{
    class Matrix
    {
        public readonly int RowDim;
        public readonly int ColDim;
        public readonly double[] Storage;

        public Matrix(int rowDim, int colDim)
        {
            RowDim = rowDim;
            ColDim = colDim;
            Storage = new double[rowDim * colDim];
        }

        public Matrix(int rowDim, int colDim, Func<double> valueFactory)
        {
            RowDim = rowDim;
            ColDim = colDim;
            Storage = new double[rowDim * colDim];
            for (var i = 0; i < Storage.Length; i++)
            {
                Storage[i] = valueFactory();
            }
        }

        public Matrix(ref double[] v, int rowDim, int colDim)
        {
            Storage = v;
            RowDim = rowDim;
            ColDim = colDim;
        }

        public Matrix Copy()
        {
            var m = new Matrix(RowDim, ColDim);
            for (var i = 0; i < Storage.Length; ++i)
            {
                m.Storage[i] = Storage[i];
            }
            return m;
        }

        public double this[int row, int column]
        {
            get { return Storage[row * ColDim + column]; }
            set { Storage[row * ColDim + column] = value; }
        }

        public double[] this[int row]
        {
            get
            {
                var res = new double[ColDim];
                for (var i = 0; i < res.Length; ++i)
                {
                    res[i] = Storage[row * ColDim + i];
                }
                return res;
            }
        }

        public double Max()
        {
            return Storage.Max();
        }

        public double Sum()
        {
            return Storage.Sum();
        }

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

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append($"[{RowDim}:{ColDim}]");
            foreach (var val in Storage)
            {
                sb.Append($" {val:##.0000}");
            }
            return sb.ToString();
        }
    }


}

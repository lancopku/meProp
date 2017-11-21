using System.Text;
using nnmnist.Data;
using nnmnist.Networks.Inits;

namespace nnmnist.Networks.Graph
{
    internal class Tensor
    {
        // for param init
        // using given initializer
        public Tensor(int rowDimension, int columnDimension, IInit init, bool needGrad = true)
        {
            Row = rowDimension;
            Col = columnDimension;
            Capacity = rowDimension * columnDimension;
            W = init != null
                ? new Matrix(rowDimension, columnDimension, init.Next)
                : new Matrix(rowDimension, columnDimension);
            DW = needGrad ? new Matrix(rowDimension, columnDimension) : null;
            HW = null;
            HW2 = null;
        }

        private Tensor()
        {
        }

        // for embedding init
        // using given weights
        public Tensor(float[] f, bool needGrad = true)
        {
            Row = f.Length;
            Col = 1;
            Capacity = f.Length;
            W = new Matrix(f, f.Length, 1);
            DW = needGrad ? new Matrix(f.Length, 1) : null;
            HW = null;
            HW2 = null;
        }

        // for intermediate init
        // no need for history
        public Tensor(int rowDimension, int columnDimension)
        {
            Row = rowDimension;
            Col = columnDimension;
            Capacity = rowDimension * columnDimension;
            W = new Matrix(rowDimension, columnDimension);
            DW = new Matrix(rowDimension, columnDimension);
            HW = null;
            HW2 = null;
        }

        public Matrix W { get; private set; }
        public Matrix DW { get; private set; }
        public Matrix HW { get; set; }
        public Matrix HW2 { get; set; }

        public int Row { get; private set; }
        public int Col { get; private set; }
        public int Capacity { get; private set; }


        public static Tensor Input(Example[] examples)
        {
            var t = new Tensor(examples.Length, examples[0].Feature.Length, null, false);
            for (var i = 0; i < examples.Length; i++)
            {
                var input = examples[i].Feature;
                for (var j = 0; j < input.Length; j++)
                    t.W[i, j] = input[j];
            }
            return t;
        }

        public static Tensor Target(Example[] examples)
        {
            var t = new Tensor(examples.Length, 1, null, false);
            for (var i = 0; i < examples.Length; i++)
                t.W[i, 0] = examples[i].Label;
            return t;
        }

        public Tensor Empty()
        {
            return new Tensor
            {
                Row = Row,
                Col = Col,
                Capacity = Row * Col,
                W = new Matrix(Row, Col),
                DW = DW == null ? null : new Matrix(Row, Col),
                HW = null,
                HW2 = null
            };
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"[{Row}:{Col}]");
            for (var i = 0; i < Row; ++i)
            {
                for (var j = 0; j < Col; ++j)
                    sb.Append($" {W[i, j]:f4}");
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}
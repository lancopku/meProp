using System.Text;
using nnmnist.Networks.Inits;

namespace nnmnist.Networks.Graph
{
    class Tensor
    {
        public Matrix W { get; private set; }
        public  Matrix DW { get; private set; }
		public  Matrix HW { get; private set; }
		public  Matrix HW2 { get; private set; }

		public  int Row { get; private set; }
		public  int Col { get; private set; }
		public  int Capacity { get; private set; }

		// for param init
		// using given initializer
		public Tensor(int rowDimension, int columnDimension, IInit init)
        {
            Row = rowDimension;
            Col = columnDimension;
            Capacity = rowDimension*columnDimension;
            W = init != null
                ? new Matrix(rowDimension, columnDimension, init.Next)
                : new Matrix(rowDimension, columnDimension);
            DW = new Matrix(rowDimension, columnDimension);
            HW = new Matrix(rowDimension, columnDimension);
            HW2 = new Matrix(rowDimension, columnDimension);
        }

		private Tensor()
		{
			Row = -1;
			Col = -1;
			Capacity = 0;
			W = null;
			DW = null;
			HW = null;
			HW2 = null;
		}

        // for embedding init
        // using given weights
        public Tensor(ref double[] f)
        {
            Row = f.Length;
            Col = 1;
            Capacity = f.Length;
            W = new Matrix(ref f, f.Length, 1);
            DW = new Matrix(f.Length, 1);
            HW = new Matrix(f.Length, 1);
            HW2 = new Matrix(f.Length, 1);
        }

		public static Tensor Input(double[] f)
		{
			var t = new Tensor()
			{
				Row = f.Length,
				Col = 1,
				Capacity = f.Length,
				W = new Matrix(ref f, f.Length, 1),
				DW = new Matrix(f.Length, 1)
			};
			return t;
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


        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"[{Row}:{Col}]");
            for (var i = 0; i < Row; ++i)
            {
                for (var j = 0; j < Col; ++j)
                {
                    sb.Append($" {W[i, j]:f4}");
                }
                sb.AppendLine();
            }
            return sb.ToString();
        }
    }
}

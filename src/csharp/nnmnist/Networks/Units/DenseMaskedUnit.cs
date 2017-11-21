using nnmnist.Networks.Graph;

namespace nnmnist.Networks.Units
{
    internal class DenseMaskedUnit : DenseUnit
    {
        private readonly Record _r;

        public int[] Indices;


        public DenseMaskedUnit(NetBase net, int inputDim, int outputDim) : base(net, inputDim, outputDim)
        {
            _r = new Record(outputDim);
            Indices = _r.Indices();
        }

        public void Prune(double per)
        {
            _r.Update(per);
            Indices = _r.Indices();
        }

        private int Dim()
        {
            return _r.Dim;
        }

        private int Length()
        {
            return _r.Mask.Length;
        }

        public override string Info()
        {
            return $"{Dim()}({Length()})";
        }

        public override Tensor Step(Flow f, Tensor x)
        {
            return f.Mask(f.AddBias(f.Multiply(x, Wxh), Bh), _r.Mask);
        }

        public override Tensor StepTopK(Flow f, Tensor x, int[] inds, int k)
        {
            return f.AddBias(f.MultiplyTopRecord(x, Wxh, k, _r, inds), Bh, inds);
        }

        public override Tensor StepTopK(Flow f, Tensor x, int k)
        {
            return f.Mask(f.AddBias(f.MultiplyTopRecord(x, Wxh, k, _r), Bh), _r.Mask);
        }
    }
}
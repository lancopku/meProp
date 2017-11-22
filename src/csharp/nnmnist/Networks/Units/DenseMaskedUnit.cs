using nnmnist.Networks.Graph;

namespace nnmnist.Networks.Units
{
    internal class DenseMaskedUnit : DenseUnit
    {
        // the dense layer for mesimp
        // the indices specify the active neurons
        // the inactive neurons are not actually removed
        // it can be done after the training is done

        private readonly Record _r; // for collecting activeness in backprop

        public int[] Indices; // the active neurons


        public DenseMaskedUnit(NetBase net, int inputDim, int outputDim) : base(net, inputDim, outputDim)
        {
            _r = new Record(outputDim);
            Indices = _r.Indices();
        }

        // simplify the layer
        // also clear the information
        public void Prune(double per)
        {
            _r.Update(per);
            Indices = _r.Indices();
        }

        // the active dimension
        private int Dim()
        {
            return _r.Dim;
        }

        // the original dimension
        private int Length()
        {
            return _r.Mask.Length;
        }

        // the information about dimension
        public override string Info()
        {
            return $"{Dim()}({Length()})";
        }

        // normal forward, normal backward
        public override Tensor Step(Flow f, Tensor x)
        {
            return f.Mask(f.AddBias(f.Multiply(x, Wxh), Bh), _r.Mask);
        }

        // normal forward, top-k backward, collect activeness
        // inds control the neurons computed, originally for efficient dropout
        // not used currently (you could use this to see how the activeness of the neurons change when top-k backprop)
        public override Tensor StepTopK(Flow f, Tensor x, int[] inds, int k)
        {
            return f.AddBias(f.MultiplyTopRecord(x, Wxh, k, _r, inds), Bh, inds);
        }

        // masked forward, top-k backward, collect activeness
        public override Tensor StepTopK(Flow f, Tensor x, int k)
        {
            return f.Mask(f.AddBias(f.MultiplyTopRecord(x, Wxh, k, _r), Bh), _r.Mask);
        }
    }
}
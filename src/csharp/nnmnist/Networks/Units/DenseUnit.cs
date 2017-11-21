using nnmnist.Networks.Graph;
using nnmnist.Networks.Inits;

namespace nnmnist.Networks.Units
{
    internal class DenseUnit : IUnit
    {
        protected readonly Tensor Bh;
        private readonly NetBase _net;
        protected readonly Tensor Wxh;

        public DenseUnit(NetBase net, int inputDim, int outputDim)
        {
            _net = net;
            Wxh = new Tensor(inputDim, outputDim, new GlorotNormalInit(_net, inputDim, outputDim));
            Bh = new Tensor(1, outputDim, null);
            SubmitParameters(net);
        }


        public void SubmitParameters(NetBase net)
        {
            net.AddParam(Wxh);
            net.AddParam(Bh);
        }

        public virtual string Info()
        {
            return $"{Bh.Col}";
        }


        public virtual Tensor Step(Flow f, Tensor x)
        {
            return f.AddBias(f.Multiply(x, Wxh), Bh);
        }

        public virtual Tensor Step(Flow f, Tensor x, int[] inds)
        {
            return f.AddBias(f.Multiply(x, Wxh, inds), Bh, inds);
        }

        public virtual Tensor StepRand(Flow f, Tensor x, int k)
        {
            return f.AddBias(f.MultiplyRand(x, Wxh, k), Bh);
        }

        public virtual Tensor StepTopK(Flow f, Tensor x, int[] inds, int k)
        {
            return f.AddBias(f.MultiplyTop(x, Wxh, k, inds), Bh, inds);
        }

        public virtual Tensor StepTopK(Flow f, Tensor x, int k)
        {
            return f.AddBias(f.MultiplyTop(x, Wxh, k), Bh);
        }
    }
}
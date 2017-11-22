using nnmnist.Networks.Graph;
using nnmnist.Networks.Inits;

namespace nnmnist.Networks.Units
{
    internal class DenseUnit : IUnit
    {
        // a dense layer

        protected readonly Tensor Bh; // a row vector
        private readonly NetBase _net;
        protected readonly Tensor Wxh; // each column is a neuron

        public DenseUnit(NetBase net, int inputDim, int outputDim)
        {
            _net = net;
            Wxh = new Tensor(inputDim, outputDim, new GlorotNormalInit(_net, inputDim, outputDim));
            Bh = new Tensor(1, outputDim, null);
            SubmitParameters(net);
        }

        // the trainale parameters are referenced by the network for update
        public void SubmitParameters(NetBase net)
        {
            net.AddParam(Wxh);
            net.AddParam(Bh);
        }

        // the dimension information
        public virtual string Info()
        {
            return $"{Bh.Col}";
        }

        // normal forward, normal backward
        public virtual Tensor Step(Flow f, Tensor x)
        {
            return f.AddBias(f.Multiply(x, Wxh), Bh);
        }

        // normal forward, normal backward
        // inds control the neurons computed, originally for efficient dropout
        public virtual Tensor Step(Flow f, Tensor x, int[] inds)
        {
            return f.AddBias(f.Multiply(x, Wxh, inds), Bh, inds);
        }

        // normal forward, rand-k backward
        public virtual Tensor StepRand(Flow f, Tensor x, int k)
        {
            return f.AddBias(f.MultiplyRand(x, Wxh, k), Bh);
        }

        // normal forward, top-k backward
        // inds control the neurons computed, originally for efficient dropout
        public virtual Tensor StepTopK(Flow f, Tensor x, int[] inds, int k)
        {
            return f.AddBias(f.MultiplyTop(x, Wxh, k, inds), Bh, inds);
        }

        // normal forward, top-k backward
        public virtual Tensor StepTopK(Flow f, Tensor x, int k)
        {
            return f.AddBias(f.MultiplyTop(x, Wxh, k), Bh);
        }
    }
}
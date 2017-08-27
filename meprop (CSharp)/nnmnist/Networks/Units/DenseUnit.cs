using System;
using nnmnist.Networks.Graph;
using nnmnist.Networks.Inits;
using nnmnist.Common;

namespace nnmnist.Networks.Units
{
    class DenseUnit : IUnit
    {
        public readonly Tensor _wxh;
        public readonly Tensor _bh;

        public DenseUnit(NetBase net, int inputDim, int outputDim, RNG r)
        {
			_wxh = new Tensor(outputDim, inputDim, new GlorotNormalInit(inputDim, outputDim, r));
			_bh = new Tensor(outputDim, 1, null);
			//_wxh = new Tensor(outputDim, inputDim, new TruncatedNormalInit(0.1));
			//_bh = new Tensor(outputDim, 1, new IdentityInit(0.1));
			net.AddParam(_wxh);
			net.AddParam(_bh);
        }

        public void SubmitParameters(NetBase net)
        {
            net.AddParam(_wxh);
            net.AddParam(_bh);
        }

        public Tensor Step(Flow f, Tensor x)
        {
            return f.Add(f.MvMultiply(_wxh, x), _bh);
        }

		public Tensor StepTime(Flow f, Tensor x)
		{
			return f.Add(f.MvMultiplyTimed(_wxh, x), _bh);
		}

		public Tensor StepRand(Flow f, Tensor x, int k)
		{
			if (k <= 0 || k >= _bh.Capacity)
			{
				return f.Add(f.MvMultiplyTimed(_wxh, x), _bh);
			}
			return f.Add(f.MvMultiplyRand(_wxh, x, k), _bh);
		}

		public Tensor StepTopK(Flow f, Tensor x, int k)
		{
			if(k<=0 || k >= _bh.Capacity)
			{
				return f.Add(f.MvMultiplyTimed(_wxh, x), _bh);
			}
			return f.Add(f.MvMultiplyTop(_wxh, x, k), _bh);
		}
	}
}

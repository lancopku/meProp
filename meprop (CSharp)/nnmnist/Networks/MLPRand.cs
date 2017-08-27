using System;
using nnmnist.Networks.Graph;
using nnmnist.Networks.Opts;
using System.Linq;

namespace nnmnist.Networks.Units
{

	class MLPRand : MLP
	{
		public override NetType Type => NetType.mlprand;
		int _k;

		public MLPRand(Config conf, int nInput, OptBase opt) : base(conf, nInput, opt)
		{
			_k = conf.TrainTop;
		}

		protected override void BuildOne(Flow f, double[] input, int y, out Tensor prob)
		{

			var x = Tensor.Input(input);

			var hs = new Tensor[_h.Length];
			for (var i = 0; i < _h.Length; i++)
			{
				if (i == 0 && i == _h.Length - 1)
				{
					hs[i] = _h[i].Step(f, x);
				}
				else if (i == 0)
				{
					hs[i] = f.Dropout(f.ReLU(_h[i].StepRand(f, x, _k)), keep);
				}
				else if (i == _h.Length - 1)
				{
					hs[i] = _h[i].Step(f, hs[i - 1]);
				}
				else
				{
					hs[i] = f.Dropout(f.ReLU(_h[i].StepRand(f, hs[i - 1], _k)), keep);
				}
			}
			prob = f.SoftmaxWithCrossEntropy(hs.Last(), y);
		}
	}
}

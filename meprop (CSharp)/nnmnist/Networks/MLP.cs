using System.Linq;
using nnmnist.Common;
using nnmnist.Networks.Graph;
using nnmnist.Networks.Opts;

namespace nnmnist.Networks.Units
{

    class MLP : NetBase
    {
		protected readonly DenseUnit[] _h;

        public override NetType Type => NetType.mlp;
		public int Layer => _h.Length;
		public double keep;

        public MLP(Config conf, int nInput, OptBase opt) : base(conf, nInput, opt)
        {
            var r = new RNG(conf.RandomSeed);
			var layer = conf.Layer;
			var nOutput = conf.LabelCount;
			var nHidden = conf.HiddenSize;
			_h = new DenseUnit[conf.Layer];
			if(conf.Layer == 1)
			{
				_h[0] = new DenseUnit(this, nInput, nOutput, r);
			}
			else
			{
				for(var i = 0; i < conf.Layer; i++)
				{
					if (i == 0)
					{
						_h[i] = new DenseUnit(this, nInput, nHidden, r);
					}else if (i == layer - 1)
					{
						_h[i] = new DenseUnit(this, nHidden, nOutput,r );
					}
					else
					{
						_h[i] = new DenseUnit(this, nHidden, nHidden,r);
					}
				}
			}
			keep = 1.0 - conf.DropProb;
        }


        protected override void BuildOne(Flow f, double[] input, int y, out Tensor prob)
        {
            var x = Tensor.Input(input);

			var hs = new Tensor[_h.Length];
			for(var i = 0; i < _h.Length; i++)
			{
				if (i == 0)
				{
					hs[i] = f.Dropout(f.ReLU(_h[i].StepTime(f, x)), keep);
				}
				else if (i == _h.Length - 1) {
					hs[i] = _h[i].Step(f, hs[i - 1]);
				}
				else
				{
					hs[i] = f.Dropout(f.ReLU(_h[i].StepTime(f, hs[i - 1])), keep);
				}
			}
            prob = f.SoftmaxWithCrossEntropy(hs.Last(), y);
        }
    }
}

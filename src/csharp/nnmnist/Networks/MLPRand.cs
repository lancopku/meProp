using System.Linq;
using nnmnist.Networks.Graph;
using nnmnist.Networks.Opts;

namespace nnmnist.Networks
{
	class MLPRand : MLP
	{
		public override NetType Type => NetType.mlprand;
	    private readonly int _k;

		public MLPRand(Config conf, int nInput, int nOuput, OptBase opt) : base(conf, nInput, nOuput, opt)
		{
		    _k = conf.TrainTop;
		}

	    protected override (Tensor res, Tensor loss) Forward(Flow f, Tensor x, Tensor y)
        {

            var h = new Tensor[Conf.Layers + 1];
            h[0] = x;

            for (var i = 0; i < Conf.Layers; i++)
            {
                if (i == Conf.Layers - 1)
                {
                    h[i + 1] = Layers[i].Step(f, h[i]);
                }
                else
                {
                    h[i + 1] = f.Rectifier(Layers[i].StepRand(f, h[i], _k));
                }
            }

            (var prob, var loss) = f.SoftmaxWithCrossEntropy(h.Last(), y);

            return (prob, loss);
        }
	}
}

using System.Linq;
using nnmnist.Networks.Graph;
using nnmnist.Networks.Opts;

namespace nnmnist.Networks
{
	class MLPTop : MLP { 
		public override NetType Type => NetType.mlptop;
	    private readonly int _k;

		public MLPTop(Config conf, int nInput, int nOuput, OptBase opt) : base(conf, nInput, nOuput, opt)
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
	                h[i+1] = Layers[i].Step(f, h[i]);
	            }
	            else
	            {
	                h[i + 1] = f.Rectifier(Layers[i].StepTopK(f, h[i], _k));
	            }
	        }


	        (var prob, var loss) = f.SoftmaxWithCrossEntropy(h.Last(), y);

	        return (prob, loss);
	    }
    }
}

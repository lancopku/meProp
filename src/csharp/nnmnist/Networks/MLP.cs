using System.Linq;
using nnmnist.Networks.Graph;
using nnmnist.Networks.Opts;
using nnmnist.Networks.Units;

namespace nnmnist.Networks
{
    internal class MLP : NetBase
    {
        protected readonly DenseUnit[] Layers;

        public MLP(Config conf, int nInput, int nOuput, OptBase opt) : base(conf, nInput, nOuput, opt)
        {
            Layers = new DenseUnit[Conf.Layers];

            for (var i = 0; i < Layers.Length; i++)
            {
                if (i == 0 && i == Conf.Layers - 1)
                    Layers[i] = new DenseUnit(this, InDim, OutDim);
                else if (i == 0)
                    Layers[i] = new DenseUnit(this, InDim, HidDim);
                else if (i == Conf.Layers - 1)
                    Layers[i] = new DenseUnit(this, HidDim, OutDim);
                else
                    Layers[i] = new DenseUnit(this, HidDim, HidDim);
            }
        }

        public override NetType Type => NetType.mlp;


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
                    h[i + 1] = f.Rectifier(Layers[i].Step(f, h[i]));
                }
            }

            (var prob, var loss) = f.SoftmaxWithCrossEntropy(h.Last(), y);

            return (prob, loss);
        }
    }
}
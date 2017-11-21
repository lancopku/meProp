using nnmnist.Networks.Graph;

namespace nnmnist.Networks.Units
{
    internal class LookupUnit : IUnit
    {
        private readonly Tensor[] _embed;
        private readonly NetBase _net;

        public LookupUnit(NetBase net, float[][] embed)
        {
            _net = net;
            _embed = new Tensor[embed.Length];
            for (var i = 0; i < _embed.Length; i++)
                _embed[i] = new Tensor(embed[i]);
        }

        public void SubmitParameters(NetBase net)
        {
            // no fixed parameters
        }


        public Tensor Step(Flow f, int id)
        {
            return f.Lookup(_embed, id);
        }
    }
}
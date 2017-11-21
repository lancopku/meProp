using System;

namespace nnmnist.Networks.Inits
{
    class TruncatedNormalInit : IInit
    {
        private readonly float _std;
        private readonly NetBase _net;

        public TruncatedNormalInit(NetBase net, float std)
        {
            _net = net;
			_std = std;
        }

        public float Next()
        {
			var val = _net.Rand.GetNormal(0, _std);
			while (val > 2 * _std || val <2*-_std)
			{
				val = _net.Rand.GetNormal(0, _std);
			}
			return val;
        }
        public override string ToString()
        {
            return $"TruncatedNormalInit[0, {_std}]";
        }
    }
}

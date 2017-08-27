using nnmnist.Common;
using System;

namespace nnmnist.Networks.Inits
{
    class TruncatedNormalInit : IInit
    {
        private readonly double _std;
        private RNG _rand;

        public TruncatedNormalInit(double std, RNG r)
        {
			_std = std;
            _rand = r;
        }

        public double Next()
        {
			var val = _rand.GetNormal(0, _std);
			while (val > 2 * _std || val <2*-_std)
			{
				val = _rand.GetNormal(0, _std);
			}
			return val;
        }
        public override string ToString()
        {
            return $"TruncatedNormalInit[0, {_std}]";
        }
    }
}

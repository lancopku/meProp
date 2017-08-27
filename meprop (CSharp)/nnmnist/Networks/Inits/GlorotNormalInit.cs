using nnmnist.Common;
using System;

namespace nnmnist.Networks.Inits
{
    class GlorotNormalInit : IInit
    {
        private readonly int _fanIn;
        private readonly int _fanOut;
        private readonly double _factor;
        private readonly RNG _r;

        public GlorotNormalInit(int fanIn, int fanOut, RNG r)
        {
            _fanIn = fanIn;
            _fanOut = fanOut;
            _factor = Math.Sqrt(6.0 / (fanIn + fanOut));
            _r = r;
        }

        public double Next()
        {
            return _r.GetNormal(0, _factor);
        }
        public override string ToString()
        {
            return $"GlorotNormalInit[{_fanIn}->{_fanOut}]";
        }
    }
}

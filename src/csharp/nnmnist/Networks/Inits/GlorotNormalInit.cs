using System;

namespace nnmnist.Networks.Inits
{
    class GlorotNormalInit : IInit
    {
        private readonly int _fanIn;
        private readonly int _fanOut;
        private readonly float _factor;
        private readonly NetBase _net;

        public GlorotNormalInit(NetBase net, int fanIn, int fanOut)
        {
            _net = net;
            _fanIn = fanIn;
            _fanOut = fanOut;
            _factor = (float)Math.Sqrt(6.0f / (fanIn + fanOut));
        }

        public float Next()
        {
            return _net.Rand.GetNormal(0, _factor);
        }
        public override string ToString()
        {
            return $"GlorotNormalInit[{_fanIn}->{_fanOut}]";
        }
    }
}

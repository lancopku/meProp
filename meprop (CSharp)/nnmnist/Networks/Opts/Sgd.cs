using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using nnmnist.Networks.Graph;

namespace nnmnist.Networks.Opts
{
    class Sgd : OptBase
    {
        private readonly double _decayRate;
        private int _pastCnt;
        private readonly int _setSize;


        public Sgd(double lr, double l2RegFactor, double clipRange, double dr, int setSize) : base(OptType.sgd, lr, clipRange, l2RegFactor)
        {
            _decayRate = dr;
            _setSize = setSize;
            _pastCnt = 0;
        }

        public override void Update(IEnumerable<Tensor> parameters)
        {
            Interlocked.Increment(ref _pastCnt);
            var newlr = LearningRate * Math.Pow(_decayRate, (double)_pastCnt / _setSize);
            foreach (var p in parameters)
            {
                var w = p.W.Storage;
                var d = p.DW.Storage;
                for (var i = 0; i < w.Length; i++)
                {
                    var dw = d[i];
					if (ClipRange > 0)
						if (dw > ClipRange)
                    {
                        dw = ClipRange;
                    }
                    else if (dw < -ClipRange)
                    {
                        dw = -ClipRange;
                    }
                    w[i] -= dw * newlr;
                    d[i] = 0;
                }
            }
        }

        public override void Update(IEnumerable<Tensor> parameters, double scale)
        {
            Interlocked.Increment(ref _pastCnt);
            var newlr = LearningRate * Math.Pow(_decayRate, (double)_pastCnt / _setSize);
            foreach (var p in parameters)
            {
                var w = p.W.Storage;
                var d = p.DW.Storage;
                for (var i = 0; i < w.Length; i++)
                {
                    var dw = d[i]*scale;
					if (ClipRange > 0)
						if (dw > ClipRange)
                    {
                        dw = ClipRange;
                    }
                    else if (dw < -ClipRange)
                    {
                        dw = -ClipRange;
                    }
                    w[i] -= dw * newlr;
                    d[i] = 0;
                }
            }
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append(base.ToString());
            sb.AppendLine($"DecayRate = {_decayRate:g2}");
            return sb.ToString();
        }
    }
}

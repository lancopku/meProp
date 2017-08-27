using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using nnmnist.Networks.Graph;

namespace nnmnist.Networks.Opts
{
    class Adam : OptBase
    {

        private readonly double _decayRate;
        private readonly double _keepRate;
        private readonly double _decayRate2;
        private readonly double _keepRate2;
        private readonly double _eps;
        private int _times;

        // lr = 0.001, dr = 0.999, dr2 = 0.9, eps = 1e-8
        public Adam(double lr=0.001, double l2RegFactor=0, double clipRange=5.0, double dr=0.999, double dr2=0.9, double eps=1e-8) : base(OptType.adam, lr, clipRange, l2RegFactor)
        {
            _decayRate = dr;
            _keepRate = 1 - dr;
            _decayRate2 = dr2;
            _keepRate2 = 1 - dr2;
            _eps = eps;
            _times = 0;
        }

        public override void Update(IEnumerable<Tensor> parameters)
        {
            Interlocked.Increment(ref _times);
            var bc1 = 1 - Math.Pow(_decayRate, _times);
            var bc2 = 1 - Math.Pow(_decayRate2, _times);
			var lr = LearningRate * Math.Sqrt(bc1) / (bc2);
            foreach (var p in parameters)
            {
                var w = p.W.Storage;
                var d = p.DW.Storage;
                var h = p.HW.Storage;
                var h2 = p.HW2.Storage;
                for (var i = 0; i < w.Length; i++)
                {
                    var dw = d[i] + L2RegFactor * w[i];
					if (ClipRange > 0)
					{
						if (dw > ClipRange)
						{
							dw = ClipRange;
						}
						else if (dw < -ClipRange)
						{
							dw = -ClipRange;
						}
					}
                    var hw = h[i] * _decayRate + dw * dw * _keepRate;
                    h[i] = hw;
                    var hw2 = h2[i] * _decayRate2 + dw * _keepRate2;
                    h2[i] = hw2;
                    w[i] -= lr * hw2 / (Math.Sqrt(hw) + _eps);
                    d[i] = 0;
                }
            }
        }

        public override void Update(IEnumerable<Tensor> parameters, double scale)
        {
            Interlocked.Increment(ref _times);
            var bc1 = 1 - Math.Pow(_decayRate, _times);
            var bc2 = 1 - Math.Pow(_decayRate2, _times);
			var lr = LearningRate * Math.Sqrt(bc1) / (bc2);
			foreach (var p in parameters)
            {
                var w = p.W.Storage;
                var d = p.DW.Storage;
                var h = p.HW.Storage;
                var h2 = p.HW2.Storage;
                for (var i = 0; i < w.Length; i++)
                {
                    var dw = d[i]*scale + L2RegFactor * w[i];
					if (ClipRange > 0)
					{
						if (dw > ClipRange)
						{
							dw = ClipRange;
						}
						else if (dw < -ClipRange)
						{
							dw = -ClipRange;
						}
					}

					var hw = h[i] * _decayRate + dw * dw * _keepRate;
                    h[i] = hw;
                    var hw2 = h2[i] * _decayRate2 + dw * _keepRate2;
                    h2[i] = hw2;
					w[i] -= lr * hw2 / (Math.Sqrt(hw) + _eps);
					d[i] = 0;
                }
            }
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append(base.ToString());
            sb.AppendLine($"beta1 = {_decayRate2:g2}");
            sb.AppendLine($"beta2 = {_decayRate:g2}");
            sb.AppendLine($"Eps = {_eps:g2}");
            return sb.ToString();
        }

    }
}

using System;
using System.Collections.Generic;
using System.Text;
using nnmnist.Networks.Graph;

namespace nnmnist.Networks.Opts
{
    class AdaGrad : OptBase
    {
        private readonly float _eps;


        public AdaGrad(float lr, float l2RegFactor, float clipRange, float eps) : base(OptType.adagrad, lr, clipRange, l2RegFactor)
        {
            _eps = eps;
        }

        public override void Update(IEnumerable<Tensor> parameters)
        {
            foreach (var p in parameters)
            {
                var w = p.W.Storage;
                var d = p.DW.Storage;
                var h = p.HW.Storage;
                for (var i = 0; i < w.Length; i++)
                {
                    var dw = d[i] + L2RegFactor * w[i];
					if (ClipRange > 0)
						if (dw > ClipRange)
                    {
                        dw = ClipRange;
                    }
                    else if (dw < -ClipRange)
                    {
                        dw = -ClipRange;
                    }
                    var hw = h[i] + dw * dw;
                    h[i] = hw;
                    w[i] -= dw * (LearningRate / (float)Math.Sqrt(hw + _eps));
                    d[i] = 0;
                }
            }
        }

        public override void Prepare(IEnumerable<Tensor> parameters)
        {
            foreach (var pi in parameters)
            {
                pi.HW = pi.DW.Empty();
            }
        }


        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append(base.ToString());
            sb.AppendLine($"Eps = {_eps:g2}");
            return sb.ToString();
        }
    }
}

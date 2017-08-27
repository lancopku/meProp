using System.Collections.Generic;
using System.Text;
using nnmnist.Networks.Graph;

namespace nnmnist.Networks.Opts
{
    abstract class OptBase
    {
        protected readonly double ClipRange;
        protected readonly double L2RegFactor;
        protected readonly double LearningRate;
        public readonly OptType Type;

        protected OptBase(OptType type, double lr, double clipRange, double l2RegFactor)
        {
            ClipRange = clipRange;
            L2RegFactor = l2RegFactor;
            LearningRate = lr;
            Type = type;
        }

        public abstract void Update(IEnumerable<Tensor> parameters);


        public abstract void Update(IEnumerable<Tensor> parameters, double scale);


        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Type = {Type}");
            sb.AppendLine($"LearningRate = {LearningRate:g2}");
            sb.AppendLine($"ClipRange = {ClipRange:g2}");
            sb.AppendLine($"L2RegFactor = {L2RegFactor:g2}");
            return sb.ToString();
        }
    }
}

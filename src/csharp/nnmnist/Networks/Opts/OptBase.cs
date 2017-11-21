using System.Collections.Generic;
using System.Text;
using nnmnist.Networks.Graph;

namespace nnmnist.Networks.Opts
{
    internal abstract class OptBase
    {
        protected readonly float ClipRange;
        protected readonly float L2RegFactor;
        protected readonly float LearningRate;
        public readonly OptType Type;

        protected OptBase(OptType type, float lr, float clipRange, float l2RegFactor)
        {
            ClipRange = clipRange;
            L2RegFactor = l2RegFactor;
            LearningRate = lr;
            Type = type;
        }

        public abstract void Update(IEnumerable<Tensor> parameters);

        public virtual void Prepare(IEnumerable<Tensor> parameters)
        {
        }


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
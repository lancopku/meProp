using System;
using System.Collections.Generic;
using System.Text;
using nnmnist.Common;
using nnmnist.Data;
using nnmnist.Networks.Graph;
using nnmnist.Networks.Opts;

namespace nnmnist.Networks
{
    internal abstract class NetBase
    {
        private readonly OptBase _opt;
        protected readonly Config Conf;
        public readonly List<Tensor> FixedParam;
        protected readonly int HidDim;
        protected readonly int InDim;
        protected readonly int OutDim;
        public readonly RandomNumberGenerator Rand;
        private bool _isTraining;


        protected NetBase(Config conf, int idim, int odim, OptBase opt)
        {
            _opt = opt;
            Conf = conf;
            InDim = idim;
            OutDim = odim;
            HidDim = Conf.HiddenSize;
            FixedParam = new List<Tensor>();
            Rand = new RandomNumberGenerator(conf.RandomSeed);
            _isTraining = false;
        }

        public abstract NetType Type { get; }


        public void AddParam(Tensor param)
        {
            FixedParam.Add(param);
        }


        protected abstract (Tensor res, Tensor loss) Forward(Flow f, Tensor x, Tensor y);


        public void Train()
        {
            _isTraining = true;
        }

        public void Eval()
        {
            _isTraining = false;
        }


        public (Flow f, Tensor res, Tensor loss) Forward(Example[] ex)
        {
            var f = new Flow(this, _isTraining);
            var x = Tensor.Input(ex);
            var y = Tensor.Target(ex);
            if(_isTraining)
                Timer.Forward.Start();
            (var res, var loss) = Forward(f, x, y);
            if(_isTraining)
                Timer.Forward.Stop();

            return (f, res, loss);
        }

        public void Backward(Flow f)
        {
            Timer.Backward.Start();
            f.Backward();
            Timer.Backward.Stop();
        }

        public void Update()
        {
            Timer.Update.Start();
            _opt.Update(FixedParam);
            Timer.Update.Stop();
        }


        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Type = {Type}");
            sb.AppendLine($"nInput = {InDim}");
            sb.AppendLine($"nOutput = {OutDim}");
            sb.AppendLine($"nHidden = {HidDim}");
            return sb.ToString();
        }
    }
}
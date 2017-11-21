using System;
using nnmnist.Data;
using nnmnist.Networks;
using nnmnist.Networks.Opts;

namespace nnmnist.Application
{
    internal class Classifier
    {
        private readonly Config _conf;

        private readonly DataSet _dataset;
        public NetBase Net;
        public OptBase Opt;


        public Classifier(Config conf, DataSet dataset)
        {
            _conf = conf;
            _dataset = dataset;
            InitOpt(_conf.OptType);
            InitNet(_conf.NetType);
            Opt.Prepare(Net.FixedParam);
        }


        private void InitOpt(OptType optType)
        {
            switch (optType)
            {
                case OptType.sgd:
                    Opt = new Sgd(_conf.LearningRate, _conf.L2RegFactor, _conf.ClipBound, _conf.DecayRate,
                        _dataset.Examples.Count);
                    break;
                case OptType.adagrad:
                    Opt = new AdaGrad(_conf.LearningRate, _conf.L2RegFactor, _conf.ClipBound, _conf.Eps);
                    break;
                case OptType.adam:
                    Opt = new Adam(l2RegFactor: _conf.L2RegFactor, clipRange: _conf.ClipBound);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(optType), optType, null);
            }
        }

        private void InitNet(NetType netType)
        {
            var eDim = _dataset.Examples[0].Feature.Length;
            var oDim = _conf.LabelCount;

            switch (netType)
            {
                case NetType.mlp:
                    Net = new MLP(_conf, eDim, oDim, Opt);
                    break;
                case NetType.mlptop:
                    Net = new MLPTop(_conf, eDim, oDim, Opt);
                    break;
                case NetType.mlprand:
                    Net = new MLPRand(_conf, eDim, oDim, Opt);
                    break;
                case NetType.mlpvar:
                    Net = new MLPVar(_conf, eDim, oDim, Opt);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(netType), netType, null);
            }
        }
    }
}
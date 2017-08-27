using System;
using nnmnist.Networks;
using nnmnist.Networks.Opts;
using nnmnist.Networks.Units;

namespace nnmnist.Common
{

    class Classifier
    {
        public NetBase Net;
        public OptBase Opt;
		private Config _conf;
        private readonly DataSet _dataset;


        public Classifier(Config conf, DataSet dataset)
        {
			_conf = conf;
			_dataset = dataset;          
            InitNet(_conf.NetType, _conf.OptType);
        }

        private void InitNet(NetType netType, OptType optType)
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

            var eDim = _dataset.Examples[0].Feature.Length;
            var hDim = _conf.HiddenSize;
            var oDim = _conf.LabelCount;

            switch (netType)
            {
                case NetType.mlp:
                    Net = new MLP(_conf, eDim, Opt);
                    break;
                case NetType.mlptop:
                    Net = new MLPTop(_conf, eDim, Opt);
                    break;
                case NetType.mlprand:
                    Net = new MLPRand(_conf, eDim, Opt);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(netType), netType, null);
            }
        }


        public double Train(Example ex)
        {
            var feature = ex.Feature;
            var label = ex.Label;
            return Net.TrainOne(feature, label);
        }


        public int Predict(double[] feature)
        {
            return Net.PredictOne(feature);
        }
    }
}

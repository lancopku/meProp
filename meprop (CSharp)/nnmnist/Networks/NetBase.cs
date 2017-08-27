using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using nnmnist.Common;
using nnmnist.Networks.Graph;
using nnmnist.Networks.Opts;

namespace nnmnist.Networks
{
    abstract class NetBase
    {
        [ThreadStatic] private static RNG _rand;
        protected readonly List<Tensor> FixedParam;
        private readonly OptBase _opt;
        protected readonly int InDim;
        protected readonly int OutDim;
        protected readonly int HidDim;
        private int _seed;

        public abstract NetType Type { get; }

        protected NetBase(Config conf, int idim, OptBase opt)
        {
            _opt = opt;
            InDim = idim;
			OutDim = conf.LabelCount;
			HidDim = conf.HiddenSize;
            FixedParam = new List<Tensor>();
			_seed = conf.RandomSeed;
        }


        public void AddParam(Tensor param)
        {
            FixedParam.Add(param);
        }



        protected abstract void BuildOne(Flow f, double [] input, int gold, out Tensor prob);

        private double Evaluate(int y, Tensor probs)
        {
			return -Math.Log(probs.W[y, 0]);
        }

        private void Forward(Flow f)
        {
            f.Forward();
        }

        private void Backward(Flow f)
        {
            f.Backward();
        }


        public double TrainOne(double [] input, int y)
        {
            if(_rand == null)
            {
                _rand = new RNG(_seed);
            }
			var f = new Flow(true, _rand);



			Timer.build.Start();
			BuildOne(f, input, y, out var probs);
			Timer.build.Stop();


			Timer.forward.Start();
            Forward(f);
			Timer.forward.Stop();


			Timer.backward.Start();
            Backward(f);
			Timer.backward.Stop();


			Timer.update.Start();
            _opt.Update(FixedParam);
			Timer.update.Stop();

			return Evaluate(y, probs);

        }

        public double TrainOneWithoutUpdate(double[] input, int y)
        {
            if (_rand == null)
            {
                _rand = new RNG(_seed);
            }
            var f = new Flow(true, _rand);



			Timer.build.Start();
			BuildOne(f, input, y, out var probs);
			Timer.build.Stop();


			Timer.forward.Start();
            Forward(f);
			Timer.forward.Stop();

			Timer.backward.Start();
            Backward(f);
			Timer.backward.Stop();
			return Evaluate(y, probs);
        }

        public void Update(double scale)
        {
			Timer.update.Start();
            _opt.Update(FixedParam, scale);
			Timer.update.Stop();

        }

        public int PredictOne(double [] input)
        {
            if (_rand == null)
            {
                _rand = new RNG(_seed);
            }
            var f = new Flow(false, _rand);

			BuildOne(f, input, -1, out var prob);

            Forward(f);

            return prob.W.MaxIndex();
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

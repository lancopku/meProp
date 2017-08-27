using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using nnmnist.Common;

namespace nnmnist.Simple
{
    class Mnist
    {

        private readonly Classifier _classifier;

        private readonly DataSet _set;
		private readonly DataSet _dev;
		private readonly DataSet _tst;

        private int _iter;
        private int _epoch;
		private Config _conf;

		public Mnist(Config conf)
		{
			_conf = conf;

			if (!File.Exists(_conf.TrainFile) || !File.Exists(_conf.TestFile))
			{
				throw new Exception("files do not exist");
			}

			var trn = FileUtil.ReadFromFile(_conf.TrainFile, _conf.TrainLabelFile);

			var tst = FileUtil.ReadFromFile(_conf.TestFile, _conf.TestLabelFile);

            var dataRandom = new RNG(_conf.RandomSeed);
			_set = new DataSet(trn.Skip(5000).ToList(), dataRandom);
			_dev = new DataSet(trn.Take(5000).ToList(), dataRandom);
			_tst = new DataSet(tst, dataRandom);
			Global.Logger.WriteLine($"Info: #in: {_set.Examples[0].Feature.Length}, #out: {_conf.LabelCount}");
			Global.Logger.WriteLine($"Info: #Trn: {_set.Count}");
			Global.Logger.WriteLine($"Info: #Dev: {_dev.Count}");
			Global.Logger.WriteLine($"Info: #Tst: {_tst.Count}");

			_classifier = new Classifier(_conf, _set);

		}

        public void RunIter()
        {
            Global.Logger.WriteLine($"{Config.Sep}{Config.Sep} Run Start");
            var bestUas = 0.0;
            for (_epoch = 0; _epoch < _conf.MaxEpoch; _epoch++)
            {
				// shuffle at the start of each training epoch
                _set.Shuffle();
				// divide the whole dataset into several smaller sets for evaluation
                var folds = DataUtil.PartitionIntoFolds(_set.RandomExamples, _conf.IterPerEpoch);
                for (_iter = 0; _iter < _conf.IterPerEpoch; _iter++)
                {
                    var fold = folds[_iter];
                    switch (_conf.TrainType)
                    {
                        case TrainType.sequential:
                            TrainOnlineSequential(fold);
                            break;
                        case TrainType.lockFree:
                            TrainOnlineLockfree(fold);
                            break;
                        case TrainType.minibatch:
                            TrainMinibatchSequential(fold);
                            break;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                    var uas = Develop(_dev);
                    
                    if (uas > bestUas)
                    {
                        bestUas = uas;
                         var tuas=   Develop(_tst, "tst");
						Global.Logger.WriteLine($"$ {bestUas:f2} | {tuas:f2}");

					}

                }
				Common.Timer.PrintTiming();
				Common.Timer.Save();
				Common.Timer.Clear();
            }
			Common.Timer.PrintAvgTiming();
            Global.Logger.WriteLine($"{Config.Sep}{Config.Sep} Run End");
        }

        private double Develop(DataSet set, string name = "dev")
        {
			var nTrained = 0;
			var sw = new Stopwatch();
			sw.Start();
			Parallel.For(0, set.Count, new ParallelOptions() { MaxDegreeOfParallelism = _conf.TrainingThreads }, i =>
			{
				set.Examples[i].PredictedLabel = _classifier.Predict(set.Examples[i].Feature);
				Interlocked.Increment(ref nTrained);
				Global.Logger.WriteConsole($"{nTrained * 100.0 / set.Count:f2}\r");
			});
			sw.Stop();
			var correct = set.Examples.Count(x => x.PredictedLabel == x.Label);
			var acc = correct * 100.0 / set.Count;
			Global.Logger.WriteLine($"\t {name}: acc: {acc:f2}({correct}/{set.Count}), time = {sw.ElapsedMilliseconds / 1000.0}s");
            return acc;
        }


        public void TrainOnlineSequential(List<Example> examples)
        {
            var nTrained = 0;
            var total = examples.Count;
			var sw = new Stopwatch();
			sw.Start();
			var loss = 0d;
            foreach (var t in examples)
            {
                loss += _classifier.Train(t);
                Interlocked.Increment(ref nTrained);
				Global.Logger.WriteConsole($"{nTrained * 100.0 / total:f2}\r");
			}
			sw.Stop();
			Global.Logger.WriteLine(
			   $"{_epoch:00}:{_iter:00}, loss = {loss/examples.Count:f4}, time = {sw.ElapsedMilliseconds / 1000.0}s");
		}


        public void TrainMinibatchSequential(List<Example> examples)
        {
            var nTrained = 0;
            var minibatchSize = _conf.MinibatchSize;
            var count = examples.Count / minibatchSize;

            var minibatches = DataUtil.PartitionIntoFolds(examples, count);
			var sw = new Stopwatch();
			sw.Start();
			var loss = 0d;

			for (var i = 0; i < count; ++i)
            {
				var minibatch = minibatches[i];
				if (_conf.TrainingThreads == 1)
				{
					for (var c = 0; c < minibatch.Count; c++)
					{
						var ex = minibatch[c];
						loss += _classifier.Net.TrainOneWithoutUpdate(ex.Feature, ex.Label);
						Interlocked.Increment(ref nTrained);
						Global.Logger.WriteConsole($"{nTrained * 100.0 / examples.Count:f2}\r");
					}
				}
				else
				{
					Parallel.For(0, minibatch.Count, new ParallelOptions() { MaxDegreeOfParallelism = _conf.TrainingThreads }, c =>
					{
						var ex = minibatch[c];
						loss += _classifier.Net.TrainOneWithoutUpdate(ex.Feature, ex.Label);
						Interlocked.Increment(ref nTrained);
						Global.Logger.WriteConsole($"{nTrained * 100.0 / examples.Count:f2}\r");
					});
					
				}
				_classifier.Net.Update(1.0/minibatch.Count);

			}
			sw.Stop();
			Global.Logger.WriteLine(
			   $"{_epoch:00}:{_iter:00}, loss = {loss / examples.Count:f4}, time = {sw.ElapsedMilliseconds / 1000.0}s");
		}


        public void TrainOnlineLockfree(List<Example> examples)
        {
            var nTrained = 0;
            var total = examples.Count;
			var loss = 0d;
			var sw = new Stopwatch();
			sw.Start();
			Parallel.For(0, examples.Count, new ParallelOptions() { MaxDegreeOfParallelism = _conf.TrainingThreads }, i =>
            {
                loss += _classifier.Train(examples[i]);
                Interlocked.Increment(ref nTrained);
				Global.Logger.WriteConsole($"{nTrained * 100.0 / total:f2}\r");
			});
			sw.Stop();
			Global.Logger.WriteLine(
			   $"{_epoch:00}:{_iter:00}, loss = {loss / examples.Count:f4}, time = {sw.ElapsedMilliseconds / 1000.0}s");
		}
    }
}

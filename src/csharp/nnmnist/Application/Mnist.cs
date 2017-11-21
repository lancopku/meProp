using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using nnmnist.Common;
using nnmnist.Data;
using nnmnist.Networks;

namespace nnmnist.Application
{
    class Mnist
    {

        private readonly Classifier _classifier; // neural network classifier

        private readonly DataSet _set; // training set (has its own RNG; changes of the classifier won't affect the data order)
		private readonly DataSet _dev; // development set
		private readonly DataSet _tst; // test set

        private int _batch; // curret batch id in an epoch; not mini-batch; start from 0
        private int _epoch; // current epoch; start from 0; this is Iter in the paper
		private readonly Config _conf; // the configuration

		public Mnist(Config conf)
		{
			_conf = conf;

            // Load datasets
			if (!File.Exists(_conf.TrainFile) || !File.Exists(_conf.TestFile))
			{
				throw new Exception("files do not exist");
			}
			var trn = FileUtil.ReadFromFile(_conf.TrainFile, _conf.TrainLabelFile);
			var tst = FileUtil.ReadFromFile(_conf.TestFile, _conf.TestLabelFile);

            // Split into training, devlopment, and test sets
			_set = new DataSet(trn.Skip(5000).ToList(), new RandomNumberGenerator(conf.RandomSeed)); // has its own RNG
			_dev = new DataSet(trn.Take(5000).ToList(), null);
			_tst = new DataSet(tst, null);

            // Print information of the datasets
			Global.Logger.WriteLine($"Info: #in: {_set.Examples[0].Feature.Length}, #out: {_conf.LabelCount}");
			Global.Logger.WriteLine($"Info: #Trn: {_set.Count}");
			Global.Logger.WriteLine($"Info: #Dev: {_dev.Count}");
			Global.Logger.WriteLine($"Info: #Tst: {_tst.Count}");

            // Build the neural network classifier
			_classifier = new Classifier(_conf, _set);

		}

        // Run the training, and the evaluation
        // In each epoch, the training set is divided into batches
        // After the classifier is trained on a batch, 
        //   the evaluation is conducted
        // The evaluation runs on the development set first
        // If the result is better,
        //   it then runs on the test set
        // The final result on the test set is used to report
        // Simplification also takes place after the training of a batch
        // so batch number is also associated with how many times the classifier is simplified in an epoch
        public void Run()
        {
            Global.Logger.WriteLine($"{Config.Sep}{Config.Sep} Run Start");
            var bestUas = 0.0;
            //var bestPUas = 0.0;

            // run epoch times
            for (_epoch = 0; _epoch < _conf.MaxEpoch; _epoch++)
            {
                // shuffle the training set
                _set.Shuffle();

                // split the traning set into batches
                var batches = _set.RandomExamples.PartitionIntoBatches(_conf.IterPerEpoch);

                // train and evaluate based on the batches
                for (_batch = 0; _batch < _conf.IterPerEpoch; _batch++)
                {
                    var batch = batches[_batch];

                    // train on the batch
                    TrainBatch(batch);

                    // evaluate on the development set
                    var uas = Develop(_dev);


                    if (uas > bestUas)
                    {
                        // if the result on the development set is better,
                        //   evaluate on the test set
                        bestUas = uas;
                        var tuas = Develop(_tst, "tst");
                        Global.Logger.WriteLine($"$ {bestUas:f2} | {tuas:f2}");

                    }



                    if (_classifier.Net is MLPVar form)
                    {
                        // if the classifier is to be simpilfied,
                        //   conduct the simplification

                        // notice the cycle mechanism
                        // the whole training process is divided into several cycles
                        // at the start of each cycle, the optimizer is reinitialized
                        // at even number cycle, conduct simplification in training
                        // at odd number cycle, conduct normal training (still only backprop the top-k)

                        // in the paper, a stage is a pair of an even number cycle and an odd nubmer cycle


                        // at the start of each cycle,
                        //   reinitialize the optimizer
                        // it also takes place at the begining of the normal training
                        // in the paper, due to the change of the term (cycle -> stage)
                        //   this detail seems to be missing
                        if ((_epoch + 1) % _conf.TrainCycle == 0)
                        {
                            _classifier.Opt.Prepare(_classifier.Net.FixedParam);
                        }
                        
                        
                        if ((_epoch / _conf.TrainCycle) % 2 == 0)
                        {
                            // if the cycle number if even, simplify the classifier
                            // the inactive neurons are not actually removed
                            // they are masked, so that they are not effective
                            // it should be easy to remove them after the training is done
                            form.Prune(_conf.Prune);

                            // print the dimenisons after the simplification
                            Global.Logger.WriteLine(
                                $"Dims: {string.Join(", ", form.Layers.Select(x => x.Info()))}");

                            // check how the simplification affect the performance of the classifier
                            // this is not necessary anymore,
                            //   sometimes the dimensions do not change
                            //   even if the dimensions do change, the performance drops
                            // but we do keep this part of the codes after the reworking of the codes,
                            //   becasue this is what we do in the experiments
                            //   and the log file has this kind of information

                            //var puas = Develop(_dev);
                            //if (puas > bestPUas)
                            //{
                            //    bestPUas = puas;
                            //    var tuas = Develop(_tst, "tst");
                            //    Global.Logger.WriteLine($"$$ {bestPUas:f2} | {tuas:f2}");
                            //}
                        }
                    }


                }

                // we have timed the forward propagation, the backward propagation, and the update
                // this print the timing of this epoch
                Timer.PrintTiming();
                Timer.Save();
                Timer.Clear();
            }

            // this pring the average timing per epoch
            Timer.PrintAvgTiming();
            Global.Logger.WriteLine($"{Config.Sep}{Config.Sep} Run End");
        }

        private double Develop(DataSet set, string name = "dev")
        {
            _classifier.Net.Eval();
			var nTrained = 0;
            float[] losses = null;
            var count = 0;
			var sw = new Stopwatch();
			sw.Start();
            foreach (var mb in set.Examples.GetMiniBatches(_conf.MinibatchSize))
            {
                (_, var prob, var loss) = _classifier.Net.Forward(mb);
                if (losses != null)
                {
                    losses.ListAdd(loss.W.Storage);
                }
                else
                {
                    losses = loss.W.Storage;
                }
                count++;
                for (var i = 0; i < mb.Length; i++)
                {
                    mb[i].PredictedLabel = prob.W.MaxIndex(i);
                }
                nTrained += mb.Length;
                Global.Logger.WriteConsole($"{nTrained * 100.0 / set.Count:f2}\r");
            }
			sw.Stop();
            losses.ListDivide(count);
            var correct = set.Examples.Count(x => x.PredictedLabel == x.Label);
			var acc = correct * 100.0 / set.Count;
			Global.Logger.WriteLine($"\t {name}: loss: {string.Join(", ", losses.Select(x => $"{x:f4}"))}, acc: {acc:f2}({correct}/{set.Count}), time = {sw.ElapsedMilliseconds / 1000.0}s");
            return acc;
        }


        // train on a batch of examples
        // evaluation and simplification take place on the level of batch
        // the classifier takes into mini-batches
        // so the loss is actually the the average loss of the mini-batches
        public void TrainBatch(List<Example> examples)
        {
            _classifier.Net.Train();
            var nTrained = 0;
            float[] lossAccumulated = null;
            var countMinibatch = 0;
            var sw = new Stopwatch();
			sw.Start();

            foreach (var mb in examples.GetMiniBatches(_conf.MinibatchSize))
            {
                (var f, var prob, var loss) = _classifier.Net.Forward(mb);
                _classifier.Net.Backward(f);
                _classifier.Net.Update();
                if (lossAccumulated != null)
                {
                    lossAccumulated.ListAdd(loss.W.Storage);
                }
                else
                {
                    lossAccumulated = loss.W.Storage;
                }
                countMinibatch++;
                for (var i = 0; i < mb.Length; i++)
                {
                    mb[i].PredictedLabel = prob.W.MaxIndex(i);
                }
                nTrained += mb.Length;
                Global.Logger.WriteConsole($"{nTrained * 100.0 / examples.Count:f2}\r");
            }
			sw.Stop();
            lossAccumulated.ListDivide(countMinibatch);
            var correct = examples.Count(x => x.PredictedLabel == x.Label);
            var acc = correct * 100.0 / examples.Count;
            Global.Logger.WriteLine(
			   $"{_epoch:00}:{_batch:00}: loss: {string.Join(", ", lossAccumulated.Select(x => $"{x:f4}"))}, acc: {acc:f2}({correct}/{examples.Count}), time = {sw.ElapsedMilliseconds / 1000.0}s");
		}

    }
}

using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using System.Text;

namespace nnmnist
{
	// the supported optimizers
	enum OptType { sgd, adagrad, adam}

	// the suported training settings (how the main loop will be run, th is the number of TrainingThreads)
	//   sequential: one example at a time
	//   lockFree: n example at a time (no sychronization)
	//               (th threads doing forward prop, backward prop and parameter update with no lock)
	//   minibatch: n example at a time (synchronize before update)
	//               (th threads doing forward prop, backward prop, but only one thread updating parameters)
	// please notice: to better demonstrate the theoretical computational cost,
	//                the minibatch implementation is different from most current frameworks,
	//                but the result should be the same
    enum TrainType
    {
        sequential,
        lockFree,
        minibatch
    }

	// the type of implemented networks
	//   mlp: baseline
	//   mlprand: meProp with random selection
	//   mlptop: meProp with top-k selection
    enum NetType { mlp, mlprand, mlptop}


    class Config
    {
        public const string Sep = "##########"; // for logging

        public int TrainingThreads; // if > 1, training will be done in parallel (no locking)
		                            // (currently the timing and loss are broken when th>1)
        public int MinibatchSize; // if TrainType is minibatch, the size of minibatch
        public int MaxEpoch; // how many epoch to run before training stops
        public int IterPerEpoch; // how many evaluations to run in each epoch
		public int Layer; // the number of layers in MLP (including the output layer)
        public double LearningRate; // learning rate of the optimizer (no effect on Adam)
        public double DecayRate; // effective in SGD and AdaGrad
        public double Eps; // effective in AdaGrad
        public double L2RegFactor; // 
        public double DropProb; // if >0 and <1, the probability which each hidden layer's output will be dropped with 
		public double ClipBound; // if > 0, truncate the gradients when updating parameters
        public int HiddenSize; // the size of hidden layer
        public int TrainTop; // k in meProp (top or rand)
		public int LabelCount; // the size of output layer
        public double DataSeqScale ; // how many data to use in training

        public string TrainFile; // should be of format ubyte-idx3
		public string TrainLabelFile; // should be of format ubyte-idx1
		public string TestLabelFile; // should be of format ubyte-idx3
		public string TestFile; // should be of format ubyte-idx1

        public int RandomSeed; // if >=0, the seed of random number generator, there are two RNG in the main procedure, one for data shuffling, one for neural network initialization (graph level)
		                       // if using mulitple traning threads, each thread will have an independent RNG for dropout and random selection (flow level)

		[JsonConverter(typeof(StringEnumConverter))] public NetType NetType;
		[JsonConverter(typeof(StringEnumConverter))] public OptType OptType;
		[JsonConverter(typeof(StringEnumConverter))] public TrainType TrainType;


		// defaults
		public Config()
		{
			TrainingThreads = 1;
			MinibatchSize = 10;
			MaxEpoch = 20;
			IterPerEpoch = 11;
			LearningRate = 0.001;
			LabelCount = 10;
			DecayRate = 0.9;
			Eps = 1e-6;
			L2RegFactor = 0;
			DropProb = 0;
			ClipBound = 5;
			HiddenSize = 500;
			Layer = 3;

			TrainTop = 30;
			DataSeqScale = 1;
			TrainFile = "train-images.idx3-ubyte";
			TrainLabelFile = "train-labels.idx1-ubyte";
			TestFile = "t10k-labels.idx1-ubyte";
			TestLabelFile = "t10k-images.idx3-ubyte";

			NetType = NetType.mlp;
			OptType = OptType.adam;
			TrainType = TrainType.minibatch;
            RandomSeed = 1300012976;
		}

		// the identifier for this config, used in logging
        public string Name()
        {
            var name = $"{NetType}-{OptType}-h{HiddenSize}-i{IterPerEpoch}-m{MinibatchSize}";
            if (DropProb>0 && DropProb <1)
            {
                name = $"{name}-d{DropProb}";
            }
            if (NetType == NetType.mlprand || NetType == NetType.mlptop)
            {
                name = $"{name}-t{TrainTop}";
            }
            return name;
        }

        public static Config ReadFromJson(string file)
		{
			var conf = JsonConvert.DeserializeObject<Config>(File.ReadAllText(file, Encoding.UTF8));
			return conf;
		}

		public void WriteToJson(string file)
		{
			File.WriteAllText(file, JsonConvert.SerializeObject(this, Formatting.Indented), Encoding.UTF8);
		}


		public override string ToString()
		{
			var infos = this.GetType().GetFields();
			var sb = new StringBuilder();

			foreach (var info in infos.OrderBy(x => x.Name))
			{
				var value = info.GetValue(this) ?? "(null)";
				sb.AppendLine(info.Name + ": " + value.ToString());
			}
			return sb.ToString();
		}
	}
}
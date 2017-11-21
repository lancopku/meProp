using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace nnmnist
{
    // the supported optimizers
    internal enum OptType
    {
        sgd,
        adagrad,
        adam
    }

    // the type of implemented networks
    //   mlp: baseline
    //   mlprand: meProp with random selection
    //   mlptop: meProp with top-k selection
    //   mlpvar: meSimp
    internal enum NetType
    {
        mlp,
        mlprand,
        mlptop,
        mlpvar
    }


    internal class Config
    {
        public const string Sep = "##########"; // for logging
        public float ClipBound; // if > 0, truncate the gradients when updating parameters
        public float DecayRate; // effective in SGD and AdaGrad
        public float Eps; // effective in AdaGrad
        public int HiddenSize; // the size of hidden layer
        public int IterPerEpoch; // how many evaluations or simplifications to run in each epoch
        public float L2RegFactor;
        public int LabelCount; // the size of output layer
        public int Layers; // the number of layers in MLP (including the output layer)
        public float LearningRate; // learning rate of the optimizer (no effect on Adam)
        public int MaxEpoch; // how many epoch to run before training stops
        public int MinibatchSize; // the size of minibatches

        [JsonConverter(typeof(StringEnumConverter))] public NetType NetType;
        [JsonConverter(typeof(StringEnumConverter))] public OptType OptType;
        public float Prune; // prune rate for meSimp
        public int RandomSeed; // if >=0, the seed of random number generator, there are two RNG in the main procedure, one for data shuffling, one for neural network (graph level)
        public string TestFile; // should be of format ubyte-idx3
        public string TestLabelFile; // should be of format ubyte-idx1
        public int TrainCycle; // number of epochs in a cycle (|stage|=2*|cycle|), effective for meSimp
        public string TrainFile; // should be of format ubyte-idx3

        public string TrainLabelFile;  // should be of format ubyte-idx1
        public int TrainTop; // k in meProp (top or rand)


        public Config()
        {
            MinibatchSize = 100;
            MaxEpoch = 20;
            IterPerEpoch = 8;
            LearningRate = 0.001f;
            LabelCount = 10;
            DecayRate = 0.09f;
            Eps = 1e-6f;
            L2RegFactor = 0f;
            ClipBound = 5;
            HiddenSize = 500;
            TrainTop = 160;
            Prune = 0.1f;
            TrainFile = null;
            TrainLabelFile = null;
            TestFile = null;
            TestLabelFile = null;
            TrainCycle = 5;
            RandomSeed = -1;
            Layers = 3;

            NetType = NetType.mlp;
            OptType = OptType.adam;
        }


        public override string ToString()
        {
            var infos = GetType().GetFields();
            var sb = new StringBuilder();

            foreach (var info in infos.OrderBy(x => x.Name))
            {
                var value = info.GetValue(this) ?? "(null)";
                sb.AppendLine(info.Name + ": " + value);
            }
            return sb.ToString();
        }

        // the identifier for this config, used in logging
        public string Name()
        {
            var name = $"{NetType}-{OptType}-h{HiddenSize}-i{IterPerEpoch}-m{MinibatchSize}";
            if (NetType == NetType.mlprand || NetType == NetType.mlptop || NetType == NetType.mlpvar)
                name = $"{name}-t{TrainTop}";
            if (NetType == NetType.mlpvar)
                name = $"{name}-p{Prune}";
            if (NetType == NetType.mlpvar && TrainCycle > 0 && TrainCycle < MaxEpoch)
                name = $"{name}-c{TrainCycle}";
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
    }
}
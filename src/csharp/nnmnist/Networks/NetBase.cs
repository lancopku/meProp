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
        // abstract class for a neural network classifier
        // define methods: Forward, Backward, and Update, which should be adequate for an simple classifier

        private readonly OptBase _opt; // the associated optimizer, optimizer will initialize the gradient history of FixedParam
        protected readonly Config Conf;
        public readonly List<Tensor> FixedParam; // the parameters of the neural network that need updating
        protected readonly int HidDim;
        protected readonly int InDim;
        protected readonly int OutDim;
        public readonly RandomNumberGenerator Rand; // RNG for initialization and operations in the graph (e.g., dropout)
        private bool _isTraining; // the state of the current model; if false, no gradient operation will be recorded


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

        // the inherited classes should implement this method
        // this method defines the procedure to get the classification results
        //   and the loss w.r.t. the input
        // x is of size minibatch size * input dimension
        // y is of size minibatch size * 1
        protected abstract (Tensor res, Tensor loss) Forward(Flow f, Tensor x, Tensor y);

        // set the neural network in the training mode
        // the gradient operations will be recorded
        // also affects dropout
        public void Train()
        {
            _isTraining = true;
        }

        // set the neural network in the evaluation mode
        // the graident operations will not be recorded
        public void Eval()
        {
            _isTraining = false;
        }

        // Forward propagation of the neural network
        // Input is constrained to the MNIST example
        // Time it when training
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

        // Forward propagation of the neural network
        // Time it when training
        public void Backward(Flow f)
        {
            Timer.Backward.Start();
            f.Backward();
            Timer.Backward.Stop();
        }

        // Update the parameters of the neural network
        // Time it when training
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
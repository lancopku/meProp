using System.Collections.Generic;
using System.Linq;
using System.Security.Policy;

namespace nnmnist.Networks.Units
{
    internal class Record
    {
        // record of the activeness of the neurons in a dense layer

        private readonly int[] _hit; // how many times a neuron in top-k in backprop
        public readonly bool[] Mask; // the active neurons, true stands for active
        private int _refer; // the total times of backprop


        public Record(int dim)
        {
            Mask = new bool[dim];
            for (var i = 0; i < Mask.Length; i++)
                Mask[i] = true;
            _hit = new int[dim];
            _refer = 0;
        }


        public int Dim => Mask.Count(x => x);

        // call once per backprop
        public void Store(int[][] inds)
        {
            foreach (var i in inds)
            {
                _refer++;
                foreach (var j in i)
                    _hit[j]++;
            }
        }

        // simplify the layer
        public int Update(double percent)
        {
            var count = 0;
            var th = (int) (_refer * percent);
            for (var i = 0; i < Mask.Length; i++)
            {
                if (_hit[i] < th)
                    Mask[i] = false;
                else
                    count++;
                _hit[i] = 0;
            }
            _refer = 0;
            return count;
        }

        // the indices of the active neurons
        public int[] Indices()
        {
            var list = new List<int>();
            for (var i = 0; i < Mask.Length; i++)
            {
                if (Mask[i])
                {
                    list.Add(i);
                }
            }
            return list.ToArray();
        }

        public override string ToString()
        {
            return string.Join(" ", _hit.Select(x => $"{(double) x / _refer:f2}"));
        }
    }
}
using System.Collections.Generic;
using System.Linq;
using System.Security.Policy;

namespace nnmnist.Networks.Units
{
    internal class Record
    {
        private readonly int[] _hit;
        public readonly bool[] Mask;
        private int _refer;


        public Record(int dim)
        {
            Mask = new bool[dim];
            for (var i = 0; i < Mask.Length; i++)
                Mask[i] = true;
            _hit = new int[dim];
            _refer = 0;
        }


        public int Dim => Mask.Count(x => x);

        public void Store(int[][] inds)
        {
            foreach (var i in inds)
            {
                _refer++;
                foreach (var j in i)
                    _hit[j]++;
            }
        }


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
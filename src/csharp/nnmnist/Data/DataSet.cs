using System.Collections.Generic;
using nnmnist.Common;

namespace nnmnist.Data
{
    internal class DataSet
    {
        private readonly RandomNumberGenerator _rand;
        public readonly List<Example> Examples;
        public readonly List<Example> RandomExamples;


        public DataSet(List<Example> examples, RandomNumberGenerator rand)
        {
            _rand = rand;
            Examples = examples;
            foreach (var item in Examples)
                item.Scale();
            RandomExamples = new List<Example>(Examples);
        }

        public int Count => Examples?.Count ?? 0;


        public void Shuffle()
        {
            var n = RandomExamples.Count;
            for (var i = 0; i < n; ++i)
            {
                var idx = i + _rand.GetInt(n - i);
                var temp = RandomExamples[i];
                RandomExamples[i] = RandomExamples[idx];
                RandomExamples[idx] = temp;
            }
        }
    }
}
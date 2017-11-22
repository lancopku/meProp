using System.Collections.Generic;
using nnmnist.Common;

namespace nnmnist.Data
{
    internal class DataSet
    {
        // for keeping the examples of a data split

        private readonly RandomNumberGenerator _rand; // for shuffling
        public readonly List<Example> Examples; // to keep the original order
        public readonly List<Example> RandomExamples; // shuffled, only for the training set


        public DataSet(List<Example> examples, RandomNumberGenerator rand)
        {
            _rand = rand;
            Examples = examples;
            foreach (var item in Examples)
                item.Scale();
            RandomExamples = new List<Example>(Examples);
        }

        public int Count => Examples?.Count ?? 0;

        // a very basic shuffling scheme
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
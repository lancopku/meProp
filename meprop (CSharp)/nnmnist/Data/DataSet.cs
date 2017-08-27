using System.Collections.Generic;
using System.Linq;
using nnmnist.Simple;
using System;

namespace nnmnist.Common
{
    class DataSet
    {
        public readonly List<Example> Examples;
        public List<Example> RandomExamples;

        public int Count => Examples?.Count ?? 0;
        private RNG _dataRand;

        public DataSet(List<Example> examples, RNG r)
        {
            _dataRand = r;
            Examples = examples;
			foreach (var item in Examples)
			{
				item.Scale();
			}
			RandomExamples = new List<Example>(Examples);
		}

        public DataSet(List<Example> examples, int randomSeed=-1)
        {
            _dataRand = new RNG(randomSeed);
            Examples = examples;
            foreach (var item in Examples)
            {
                item.Scale();
            }
            RandomExamples = new List<Example>(Examples);
        }



        public void Shuffle()
        {
            var n = RandomExamples.Count;
            for (var i = 0; i < n; ++i)
            {
                var idx = i + _dataRand.Next(n - i);
                var temp = RandomExamples[i];
                RandomExamples[i] = RandomExamples[idx];
                RandomExamples[idx] = temp;
            }
        }

    }
}

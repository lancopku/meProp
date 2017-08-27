using System.Collections.Generic;
using System.Linq;
using nnmnist.Simple;

namespace nnmnist.Common
{
    static class DataUtil
    {

        public static void Shuffle<T>(this IList<T> list, RNG rand)
        {
            var count = list.Count;
            while (count > 1)
            {
                count--;
                var k = rand.Next(count + 1);
                var v = list[k];
                list[k] = list[count];
                list[count] = v;
            }
        }

        public static List<T> GetRandomSubList<T>(List<T> input, int subsetSize, RNG rand)
        {
            var inputSize = input.Count;
            if (subsetSize > inputSize)
            {
                subsetSize = inputSize;
            }

            
            for (var i = 0; i < subsetSize; i++)
            {
                var indexToSwap = i + rand.Next(inputSize - i);
                var temp = input[i];
                input[i] = input[indexToSwap];
                input[indexToSwap] = temp;
            }
            return input.GetRange(0, subsetSize);
        }


        public static List<List<T>> PartitionIntoFolds<T>(List<T> Storage, int foldCount)
        {
            if (foldCount == 0)
            {
                return new List<List<T>> { Storage };
            }
            var folds = new List<List<T>>();
            var count = Storage.Count;
            var foldSize = count / foldCount;
            var remainder = count % foldCount;
            var start = 0;
            for (var foldNum = 0; foldNum < foldCount; foldNum++)
            {
                var size = foldSize;
                if (foldNum < remainder)
                {
                    size++;
                }
                folds.Add(Storage.GetRange(start, size));
                start += size;
            }
            return folds;
        }
    }
}

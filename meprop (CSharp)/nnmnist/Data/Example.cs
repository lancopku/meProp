using System.Collections.Generic;

namespace nnmnist.Common
{
    class Example
    {
		public int[] values;
        public double[] Feature;
        public readonly int Label;
		public int PredictedLabel;

        public Example(int[] feature, int label)
        {
            values = feature;
            Label = label;
			PredictedLabel = -1;
        }

		public void Scale()
		{
			Feature = new double[values.Length];
			for (int i = 0; i < Feature.Length; i++)
			{
				Feature[i] = ((double)values[i])/255.0;
			}
		}
	


		public void Normalize(double mean=0.1307, double std=0.3083)
		{
			Feature = new double[values.Length];
			for (int i = 0; i < Feature.Length; i++)
			{
				Feature[i] = ((double)values[i] - mean) / std;
			}
		}

    }
}

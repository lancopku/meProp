namespace nnmnist.Data
{
    internal class Example
    {
        public readonly int Label;
        public float[] Feature;
        public int PredictedLabel;
        public int[] Values;

        public Example(int[] feature, int label)
        {
            Values = feature;
            Label = label;
            PredictedLabel = -1;
        }

        public void Scale()
        {
            Feature = new float[Values.Length];
            for (var i = 0; i < Feature.Length; i++)
                Feature[i] = Values[i] / 255.0f;
        }

        public void Normalize(float mean = 0.1307f, float std = 0.3083f)
        {
            Feature = new float[Values.Length];
            for (var i = 0; i < Feature.Length; i++)
                Feature[i] = (Values[i] - mean) / std;
        }
    }
}
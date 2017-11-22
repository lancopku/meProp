namespace nnmnist.Data
{
    internal class Example
    {
        // a data example for MNIST

        public readonly int Label; // ground-truth label
        public float[] Feature; // the input, scaled into 0-1
        public int PredictedLabel; // predicted label
        public int[] Values; // the raw values, 0-255

        public Example(int[] feature, int label)
        {
            Values = feature;
            Label = label;
            PredictedLabel = -1;
        }

        // scale the int values to 0-1
        // or the values in the neural networks can easily explode (overflow)
        public void Scale()
        {
            Feature = new float[Values.Length];
            for (var i = 0; i < Feature.Length; i++)
                Feature[i] = Values[i] / 255.0f;
        }

        // normalize the value
        public void Normalize(float mean = 0.1307f, float std = 0.3083f)
        {
            Feature = new float[Values.Length];
            for (var i = 0; i < Feature.Length; i++)
                Feature[i] = (Values[i] - mean) / std;
        }
    }
}
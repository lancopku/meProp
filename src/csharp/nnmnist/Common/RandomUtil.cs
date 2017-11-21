using System;

namespace nnmnist.Common
{

    public class RandomNumberGenerator
    {
        private readonly Random _rand;

        public RandomNumberGenerator(int seed = -1)
        {
            if (seed < 0)
                _rand = new Random();
            else
                _rand = new Random(seed);
        }

        public int GetInt(int upperBound)
        {
            return _rand.Next(upperBound);
        }

        public int GetInt()
        {
            return _rand.Next();
        }


        public float GetFloat(float lowerBound, float upperBound)
        {
            return (float) _rand.NextDouble() * (upperBound - lowerBound) + lowerBound;
        }

        public double GetDouble(double lowerBound, double upperBound)
        {
            return _rand.NextDouble() * (upperBound - lowerBound) + lowerBound;
        }


        public double[] GetDouble(int len, double lowerBound, double upperBound)
        {
            var x = new double[len];
            for (var i = 0; i < x.Length; i++)
                x[i] = _rand.NextDouble() * (upperBound - lowerBound) + lowerBound;
            return x;
        }

        public float GetNormal(double mean, double stddev)
        {
            var u1 = _rand.NextDouble();
            var u2 = _rand.NextDouble();
            var randomStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return (float) (mean + stddev * randomStdNormal);
        }

        public int GetInt(float lowerBound, float upperBound)
        {
            return (int) (_rand.NextDouble() * (upperBound - lowerBound) + lowerBound);
        }
    }
}
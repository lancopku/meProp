using System;
using System.Threading;

namespace nnmnist.Common
{
	public class RNG
	{
		private readonly Random Random;

        public RNG(int randomSeed = -1)
        {
            Random = randomSeed<0?new Random(): new Random(randomSeed);
        }

        public double NextDouble()
        {
            return Random.NextDouble();
        }

        public int Next()
        {
            return Random.Next();
        }

        public int Next(int maxValue)
        {
            return Random.Next(maxValue);
        }

		public int GetIntExclusive(int upperBound)
		{
			return Random.Next(upperBound);
		}

		public float GetFloat(float lowerBound, float upperBound)
		{
			return (float)Random.NextDouble() * (upperBound - lowerBound) + lowerBound;
		}
		public double GetDouble(double lowerBound, double upperBound)
		{
			return Random.NextDouble() * (upperBound - lowerBound) + lowerBound;
		}

		public double[] GetDouble(int len, double lowerBound, double upperBound)
		{
			var x = new double[len];
			for (var i = 0; i < x.Length; i++)
			{
				x[i] = Random.NextDouble() * (upperBound - lowerBound) + lowerBound;
			}
			return x;
		}

		public double GetNormal(double mean, double stddev)
		{
			var u1 = Random.NextDouble();
			var u2 = Random.NextDouble();
			var randomStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
			return mean + stddev * randomStdNormal;
		}

		public int GetInt(float lowerBound, float upperBound)
		{
			return (int)(Random.NextDouble() * (upperBound - lowerBound) + lowerBound);
		}
	}
}

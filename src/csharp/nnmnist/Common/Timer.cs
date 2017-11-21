using System.Diagnostics;

namespace nnmnist.Common
{

	static class Timer
	{
		public static readonly Stopwatch Forward;
		public static readonly Stopwatch Backward;
		public static readonly Stopwatch Update;
		private static long _fp;
		private static long _bp;
		private static long _u;
		private static int _times;


	    static Timer()
		{
			Forward = new Stopwatch();
			Backward = new Stopwatch();
			Update = new Stopwatch();
		}

		public static void Save()
		{
			_times += 1;
			_fp += Forward.ElapsedTicks;
			_bp += Backward.ElapsedTicks;
			_u += Update.ElapsedTicks;
		}


		public static void Clear()
		{
			Forward.Reset();
			Backward.Reset();
			Update.Reset();
		}

		public static void PrintTiming()
		{
			Global.Logger.WriteLine($"Time: fp: {Forward.ElapsedMilliseconds/1000.0:f2}s, bp: {Backward.ElapsedMilliseconds/1000.0:f2}s, update: {Update.ElapsedMilliseconds/1000.0:f2}s");
		}

		public static void PrintAvgTiming()
		{
			var bpt = _bp * 1.0 / Stopwatch.Frequency / _times;
			var fpt = _fp * 1.0 / Stopwatch.Frequency / _times;
			var ut = _u * 1.0 / Stopwatch.Frequency / _times;

			Global.Logger.WriteLine($"AvgTime: fp: {fpt:f2}s, bp: {bpt:f2}s, update:{ut:f2}s");
		}
	}
}

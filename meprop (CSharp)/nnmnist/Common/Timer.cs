using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace nnmnist.Common
{

	static class Timer
	{
		public static Stopwatch build;
		public static Stopwatch forward;
		public static Stopwatch backward;
		public static Stopwatch update;
		public static Stopwatch select;
		public static Stopwatch mv;
		private static long b;
		private static long fp;
		private static long bp;
		private static long u;
		private static long s;
		private static long m;
		private static int times;


	    static Timer()
		{
			build = new Stopwatch();
			forward = new Stopwatch();
			backward = new Stopwatch();
			update = new Stopwatch();
			select = new Stopwatch();
			mv = new Stopwatch();
		}

		public static void Save()
		{
			times += 1;
			b += build.ElapsedTicks;
			fp += forward.ElapsedTicks;
			bp += backward.ElapsedTicks;
			u += update.ElapsedTicks;
			s += select.ElapsedTicks;
			m += mv.ElapsedTicks;
		}


		public static void Clear()
		{
			build.Reset();
			forward.Reset();
			backward.Reset();
			update.Reset();
			select.Reset();
			mv.Reset();
		}

		public static void PrintTiming()
		{
			Global.Logger.WriteLine($"Time: build: {build.ElapsedMilliseconds/1000.0:f2}s, fp: {forward.ElapsedMilliseconds/1000.0:f2}s, bp: {backward.ElapsedMilliseconds/1000.0:f2}s, update:{update.ElapsedMilliseconds/1000.0:f2}s, select: {select.ElapsedMilliseconds/1000.0:f2}s, mv: {mv.ElapsedMilliseconds/1000.0:f2}s");
		}

		public static void PrintAvgTiming()
		{
			var bt = b * 1.0 / Stopwatch.Frequency / times;
			var bpt = bp * 1.0 / Stopwatch.Frequency / times;
			var fpt = fp * 1.0 / Stopwatch.Frequency / times;
			var ut = u * 1.0 / Stopwatch.Frequency / times;
			var st = s * 1.0 / Stopwatch.Frequency / times;
			var mt = m * 1.0 / Stopwatch.Frequency / times;

			Global.Logger.WriteLine($"AvgTime: build: {bt:f2}s, fp: {fpt:f2}s, bp: {bpt:f2}s, update: {ut:f2}s, select: {st:f2}s, mv: {mt:f2}s");
		}
	}
}

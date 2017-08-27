using System;
using System.Linq;
using nnmnist.Common;
using System.Collections.Generic;

namespace nnmnist.Networks.Graph
{
	class Flow
	{
		private readonly List<Action> BackwardOps;
		private readonly List<Action> _forwardOps;
        private readonly RNG _rand;

		protected static readonly Func<double, double> SigmoidFunc = x => 1.0f / (1 + Math.Exp(-x));
		protected readonly bool IsTrain;

		public Flow(bool train, RNG rand)
		{
            _rand = rand;
			_forwardOps = new List<Action>();
			BackwardOps = train ? new List<Action>() : null;
			IsTrain = train;
		}

		protected void AddBackOp(Action act)
		{
			BackwardOps.Add(act);
		}

		protected void AddForOp(Action act)
		{
			_forwardOps.Add(act);
		}

		public void Forward()
		{
			var count = _forwardOps.Count;
			for (var i = 0; i < count; ++i)
			{
				_forwardOps[i]();
			}
		}

		public void Backward()
		{
			for (var i = BackwardOps.Count - 1; i >= 0; --i)
			{
				BackwardOps[i]();
			}
		}


		#region Utility
		[ThreadStatic] private static TopNHeap _tops;

		private int[] GetAbsTopsHeap(Matrix m, int k)
		{
			if (_tops == null)
			{
				_tops = new TopNHeap(k);
			}
			var res = _tops.GetAbsTop(m.Storage, k);
			return res;
		}

		private int[] GetRandK(int end, int k)
		{
			var res = new int[k];

			var needed = k;
			var back = end;
			while (needed > 0)
			{
				if (_rand.GetDouble(0, 1) < (double)needed / back)
				{
					res[needed - 1] = back - 1;
					needed--;
				}
				back--;
			}
			return res;
		}
#endregion Utility

		#region Shape

		public virtual Tensor Concat(params Tensor[] list)
		{
			var res = new Tensor(list.Sum(t => t.Capacity), 1);
			AddForOp(delegate
			{
				for (int t = 0, i = 0; t < list.Length; t++)
				{
					for (var j = 0; j < list[t].W.Storage.Length; j++, i++)
					{
						res.W.Storage[i] = list[t].W.Storage[j];
					}
				}
			});
			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (int t = 0, i = 0; t < list.Length; t++)
					{
						for (var j = 0; j < list[t].W.Storage.Length; j++, i++)
						{
							list[t].DW.Storage[j] += res.DW.Storage[i];
						}
					}
				});
			}

			return res;
		}

		#endregion Shape

		#region Elementwise

		public virtual Tensor Add(Tensor l, Tensor r)
		{

			var res = new Tensor(l.Row, 1);

			AddForOp(delegate
			{
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					res.W.Storage[i] = l.W.Storage[i] + r.W.Storage[i];
				}
			});
			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (var i = 0; i < res.DW.Storage.Length; i++)
					{
						l.DW.Storage[i] += res.DW.Storage[i];
						r.DW.Storage[i] += res.DW.Storage[i];
					}
				});
			}

			return res;
		}

		public virtual Tensor Minus(double l, Tensor t)
		{

			var res = new Tensor(t.Row, 1);
			AddForOp(delegate
			{
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					res.W.Storage[i] = l - t.W.Storage[i];
				}
			});
			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (var i = 0; i < t.DW.Storage.Length; i++)
					{
						t.DW.Storage[i] -= res.DW.Storage[i];
					}
				});
			}

			return res;
		}


		public virtual Tensor ElementwiseMultiply(Tensor l, Tensor r)
		{

			var res = new Tensor(l.Row, 1);

			AddForOp(delegate
			{
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					res.W.Storage[i] = l.W.Storage[i] * r.W.Storage[i];
				}
			});
			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (var i = 0; i < res.DW.Storage.Length; i++)
					{
						l.DW.Storage[i] += r.W.Storage[i] * res.DW.Storage[i];
						r.DW.Storage[i] += l.W.Storage[i] * res.DW.Storage[i];
					}
				});
			}

			return res;
		}

		#endregion Elementwise

		#region Activation

		public virtual Tensor Tanh(Tensor t)
		{

			var res = new Tensor(t.Row, 1);
			AddForOp(delegate
			{
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					res.W.Storage[i] = Math.Tanh(t.W.Storage[i]);
				}
			});
			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (var i = 0; i < t.DW.Storage.Length; i++)
					{
						t.DW.Storage[i] += (1 - res.W.Storage[i] * res.W.Storage[i]) * res.DW.Storage[i];
					}
				});
			}

			return res;
		}

		public virtual Tensor Sigmoid(Tensor t)
		{

			var res = new Tensor(t.Row, 1);

			AddForOp(delegate
			{
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					res.W.Storage[i] = SigmoidFunc(t.W.Storage[i]);
				}
			});
			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (var i = 0; i < t.DW.Storage.Length; i++)
					{
						t.DW.Storage[i] += (1 - res.W.Storage[i]) * res.W.Storage[i] * res.DW.Storage[i];
					}
				});
			}

			return res;
		}


		public virtual Tensor ReLU(Tensor t)
		{

			var res = new Tensor(t.Row, 1);

			AddForOp(delegate
			{
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					res.W.Storage[i] = t.W.Storage[i] <= 0 ? 0 : t.W.Storage[i];
				}
			});

			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (var i = 0; i < t.DW.Storage.Length; i++)
					{
						t.DW.Storage[i] += t.W.Storage[i] <= 0 ? 0 : res.DW.Storage[i];
					}
				});
			}

			return res;
		}


		public virtual Tensor Cubic(Tensor t)
		{

			var res = new Tensor(t.Row, 1);
			AddForOp(delegate
			{
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					res.W.Storage[i] = t.W.Storage[i] * t.W.Storage[i] * t.W.Storage[i];
				}
			});
			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (var i = 0; i < t.DW.Storage.Length; i++)
					{
						t.DW.Storage[i] += 3 * t.W.Storage[i] * t.W.Storage[i] * res.DW.Storage[i];
					}
				});
			}

			return res;
		}


		public virtual Tensor Dropout(Tensor t, double keep)
		{
			if (!IsTrain || keep >= 1.0 || keep == 0)
			{
				return t;
			}

			var res = new Tensor(t.Row, 1);
			var random = _rand;
			var keeps = new bool[t.Row];
			var factor = 1.0 / keep;

			for (var i = 0; i < keeps.Length; i++)
			{
				if (random.NextDouble() < keep)
				{
					keeps[i] = true;
				}
				else
				{
					keeps[i] = false;
				}
			}

			AddForOp(delegate
			{
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					if (keeps[i])
					{
						res.W.Storage[i] = t.W.Storage[i];
					}
				}
			});

			AddBackOp(delegate
			{
				for (var i = 0; i < t.DW.Storage.Length; i++)
				{
					if (keeps[i])
					{
						t.DW.Storage[i] += res.DW.Storage[i] * factor;
					}
				}
			});

			return res;
		}


		#endregion Activation

		#region Loss

		public virtual Tensor SoftmaxWithCrossEntropy(Tensor t, int gold = -1)
		{

			var res = new Tensor(t.Row, 1);

			AddForOp(delegate
			{
				var max = t.W.Max();
				var sum = 0d;
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					res.W.Storage[i] = Math.Exp(t.W.Storage[i] - max);
					sum += res.W.Storage[i];
				}
				for (var i = 0; i < res.W.Storage.Length; i++)
				{
					res.W.Storage[i] /= sum;
				}
			});
			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (var i = 0; i < t.DW.Storage.Length; i++)
					{
						t.DW.Storage[i] = res.W.Storage[i];
					}
					t.DW.Storage[gold] -= 1;
				});
			}

			return res;
		}
		#endregion Loss

		#region Matrix Multiply
		public virtual Tensor MvMultiply(Tensor l, Tensor r)
		{
			var nIn = l.Col;
			var nOut = l.Row;
			var res = new Tensor(nOut, 1);


			AddForOp(delegate
			{
				for (var i = 0; i < nOut; i++)
				{
					var sum = 0d;
					for (var j = 0; j < nIn; j++)
					{
						sum += l.W[i, j] * r.W.Storage[j];
					}
					res.W.Storage[i] = sum;
				}
			});

			if (IsTrain)
			{
				AddBackOp(delegate
				{
					for (var i = 0; i < nOut; i++)
					{
						var fac = res.DW.Storage[i];
						for (var j = 0; j < nIn; j++)
						{
							l.DW[i, j] += r.W.Storage[j] * fac;
							r.DW.Storage[j] += l.W[i, j] * fac;
						}
					}
				});
			}

			return res;
		}

		public virtual Tensor MvMultiplyTimed(Tensor l, Tensor r)
		{
			var nIn = l.Col;
			var nOut = l.Row;
			var res = new Tensor(nOut, 1);


			AddForOp(delegate
			{
				for (var i = 0; i < nOut; i++)
				{
					var sum = 0d;
					for (var j = 0; j < nIn; j++)
					{
						sum += l.W[i, j] * r.W.Storage[j];
					}
					res.W.Storage[i] = sum;
				}
			});

			if (IsTrain)
			{
				AddBackOp(delegate
				{
					Timer.mv.Start();
					for (var i = 0; i < nOut; i++)
					{
						var fac = res.DW.Storage[i];
						for (var j = 0; j < nIn; j++)
						{
							l.DW[i, j] += r.W.Storage[j] * fac;
							r.DW.Storage[j] += l.W[i, j] * fac;
						}
					}
					Timer.mv.Stop();
				});
			}

			return res;
		}


		public virtual Tensor MvMultiplyTop(Tensor l, Tensor r, int k)
		{
			var nIn = l.Col;
			var nOut = l.Row;
			var res = new Tensor(nOut, 1);

			AddForOp(delegate
			{
				for (var i = 0; i < nOut; i++)
				{
					var sum = 0d;
					for (var j = 0; j < nIn; j++)
					{
						sum += l.W[i, j] * r.W.Storage[j];
					}
					res.W.Storage[i] = sum;
				}
			});

			if (IsTrain)
			{
				AddBackOp(delegate
				{
					Timer.mv.Start();
					Timer.select.Start();
					var inds = GetAbsTopsHeap(res.DW, k);
					Timer.select.Stop();
					for (var c = 0; c < inds.Length; ++c)
					{
						var i = inds[c];
						var fac = res.DW.Storage[i];
						for (var j = 0; j < nIn; j++)
						{
							l.DW[i, j] += r.W.Storage[j] * fac;
							r.DW.Storage[j] += l.W[i, j] * fac;
						}
					}
					Timer.mv.Stop();
				});
			}
			return res;
		}

		public virtual Tensor MvMultiplyRand(Tensor l, Tensor r, int k)
		{
			var nIn = l.Col;
			var nOut = l.Row;
			var res = new Tensor(nOut, 1);


			AddForOp(delegate
			{
				for (var i = 0; i < nOut; i++)
				{
					var sum = 0d;
					for (var j = 0; j < nIn; j++)
					{
						sum += l.W[i, j] * r.W.Storage[j];
					}
					res.W.Storage[i] = sum;
				}
			});

			if (IsTrain)
			{
				AddBackOp(delegate
				{
					Timer.mv.Start();
					Timer.select.Start();
					var inds = GetRandK(res.DW.RowDim, k);
					Timer.select.Stop();
					for (var c = 0; c < inds.Length; ++c)
					{
						var i = inds[c];
						var fac = res.DW.Storage[i];
						for (var j = 0; j < nIn; j++)
						{
							l.DW[i, j] += r.W.Storage[j] * fac;
							r.DW.Storage[j] += l.W[i, j] * fac;
						}
					}
					Timer.mv.Stop();
				});
			}

			return res;
		}

		#endregion Matrix Multiply
	}
}
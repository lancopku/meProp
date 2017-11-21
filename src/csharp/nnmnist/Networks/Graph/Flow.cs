using System;
using System.Collections.Generic;
using System.Linq;
using nnmnist.Common;
using nnmnist.Networks.Units;

namespace nnmnist.Networks.Graph
{
    internal class Flow
    {
        protected static readonly Func<float, float> SigmoidFunc = x => 1.0f / (1 + (float) Math.Exp(-x));
        private readonly List<Action> _backwardOps;
        private readonly NetBase _net;
        protected readonly bool IsTrain;

        public Flow(NetBase net, bool train)
        {
            _net = net;
            _backwardOps = train ? new List<Action>() : null;
            IsTrain = train;
        }

        protected void AddBackOp(Action act)
        {
            _backwardOps.Add(act);
        }


        public void Backward()
        {
            for (var i = _backwardOps.Count - 1; i >= 0; --i)
                _backwardOps[i]();
        }


        public virtual Tensor Lookup(Tensor[] e, int id)
        {
            return e[id];
        }


        public virtual Tensor Concat(params Tensor[] list)
        {
            var res = new Tensor(list.Sum(t => t.Capacity), 1);


            for (int t = 0, i = 0; t < list.Length; t++)
            {
                for (var j = 0; j < list[t].W.Storage.Length; j++, i++)
                    res.W.Storage[i] = list[t].W.Storage[j];
            }

            if (!IsTrain) return res;
            AddBackOp(delegate
            {
                for (int t = 0, i = 0; t < list.Length; t++)
                {
                    for (var j = 0; j < list[t].W.Storage.Length; j++, i++)
                        list[t].DW.Storage[j] += res.DW.Storage[i];
                }
            });

            return res;
        }


        public virtual Tensor Add(Tensor l, Tensor r)
        {
            var res = l.Empty();


            for (var i = 0; i < res.W.Storage.Length; i++)
                res.W.Storage[i] = l.W.Storage[i] + r.W.Storage[i];

            if (!IsTrain) return res;
            AddBackOp(delegate
            {
                for (var i = 0; i < res.DW.Storage.Length; i++)
                {
                    l.DW.Storage[i] += res.DW.Storage[i];
                    r.DW.Storage[i] += res.DW.Storage[i];
                }
            });

            return res;
        }

        public virtual Tensor AddBias(Tensor l, Tensor bias)
        {
            var res = l.Empty();

            var dim = bias.Col;
            for (var i = 0; i < res.W.Storage.Length; i++)
                res.W.Storage[i] = l.W.Storage[i] + bias.W.Storage[i % dim];


            if (!IsTrain) return res;

            if (l.DW != null && bias.DW != null)
            {
                AddBackOp(delegate
                {
                    for (var i = 0; i < res.DW.Storage.Length; i++)
                    {
                        l.DW.Storage[i] += res.DW.Storage[i];
                        bias.DW.Storage[i % dim] += res.DW.Storage[i];
                    }
                });
            }
            else if (l.DW != null)
            {
                AddBackOp(delegate
                {
                    for (var i = 0; i < res.DW.Storage.Length; i++)
                        l.DW.Storage[i] += res.DW.Storage[i];
                });
            }
            else if (bias.DW != null)
            {
                AddBackOp(delegate
                {
                    for (var i = 0; i < res.DW.Storage.Length; i++)
                        bias.DW.Storage[i % dim] += res.DW.Storage[i];
                });
            }

            return res;
        }

        public virtual Tensor AddBias(Tensor l, Tensor bias, int[] inds)
        {
            var res = l.Empty();

            for (var i = 0; i < l.Row; i++)
            {
                foreach (var j in inds)
                    res.W[i, j] = l.W[i, j] + bias.W.Storage[j];
            }

            if (!IsTrain) return res;

            if (l.DW != null && bias.DW != null)
            {
                AddBackOp(delegate
                {
                    for (var i = 0; i < l.Row; i++)
                    {
                        foreach (var j in inds)
                        {
                            var t = res.DW[i, j];
                            l.DW[i, j] += t;
                            bias.DW.Storage[j] += t;
                        }
                    }
                });
            }
            else if (l.DW != null)
            {
                AddBackOp(delegate
                {
                    for (var i = 0; i < l.Row; i++)
                    {
                        foreach (var j in inds)
                        {
                            var t = res.DW[i, j];
                            l.DW[i, j] += t;
                        }
                    }
                });
            }
            else if (bias.DW != null)
            {
                AddBackOp(delegate
                {
                    for (var i = 0; i < l.Row; i++)
                    {
                        foreach (var j in inds)
                        {
                            var t = res.DW[i, j];
                            bias.DW.Storage[j] += t;
                        }
                    }
                });
            }

            return res;
        }

        public virtual Tensor Add(Tensor l, Tensor r, int[] inds)
        {
            var res = l.Empty();


            for (var i = 0; i < l.Row; i++)
            {
                foreach (var j in inds)
                    res.W[i, j] = l.W[i, j] + r.W[i, j];
            }

            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < l.Row; i++)
                {
                    foreach (var j in inds)
                    {
                        l.DW[i, j] += res.DW[i, j];
                        r.DW[i, j] += res.DW[i, j];
                    }
                }
            });


            return res;
        }


        public virtual Tensor Minus(float l, Tensor t)
        {
            var res = t.Empty();

            for (var i = 0; i < res.W.Storage.Length; i++)
                res.W.Storage[i] = l - t.W.Storage[i];

            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.DW.Storage.Length; i++)
                    t.DW.Storage[i] -= res.DW.Storage[i];
            });


            return res;
        }


        public virtual Tensor Mask(Tensor t, bool[] mask)
        {
            var res = t.Empty();

            for (var i = 0; i < t.Row; i++)
            {
                for (var j = 0; j < t.Col; j++)
                {
                    if (mask[j])
                        res.W[i, j] = t.W[i, j];
                }
            }


            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.Row; i++)
                {
                    for (var j = 0; j < t.Col; j++)
                    {
                        if (mask[j])
                            t.DW[i, j] += res.DW[i, j];
                    }
                }
            });


            return res;
        }


        public virtual Tensor ElementwiseMultiply(Tensor l, Tensor r)
        {
            var res = l.Empty();


            for (var i = 0; i < res.W.Storage.Length; i++)
                res.W.Storage[i] = l.W.Storage[i] * r.W.Storage[i];

            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < res.DW.Storage.Length; i++)
                {
                    l.DW.Storage[i] += r.W.Storage[i] * res.DW.Storage[i];
                    r.DW.Storage[i] += l.W.Storage[i] * res.DW.Storage[i];
                }
            });


            return res;
        }


        public virtual Tensor Tanh(Tensor t)
        {
            var res = new Tensor(t.Row, t.Col);

            for (var i = 0; i < res.W.Storage.Length; i++)
                res.W.Storage[i] = (float) Math.Tanh(t.W.Storage[i]);

            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.DW.Storage.Length; i++)
                    t.DW.Storage[i] += (1 - res.W.Storage[i] * res.W.Storage[i]) * res.DW.Storage[i];
            });


            return res;
        }

        public virtual Tensor Sigmoid(Tensor t)
        {
            var res = t.Empty();


            for (var i = 0; i < res.W.Storage.Length; i++)
                res.W.Storage[i] = SigmoidFunc(t.W.Storage[i]);

            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.DW.Storage.Length; i++)
                    t.DW.Storage[i] += (1 - res.W.Storage[i]) * res.W.Storage[i] * res.DW.Storage[i];
            });


            return res;
        }


        public virtual Tensor HardSigmoid(Tensor t)
        {
            var res = t.Empty();


            for (var i = 0; i < res.W.Storage.Length; i++)
            {
                var val = t.W.Storage[i];
                var rval = 0f;
                if (val >= 2.5f)
                    rval = 1f;
                else if (val > -2.5f)
                    rval = 0.2f * val + 0.5f;
                res.W.Storage[i] = rval;
            }

            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.DW.Storage.Length; i++)
                {
                    var val = t.W.Storage[i];
                    if (val < 2.5f && val > -2.5f)
                        t.DW.Storage[i] = 0.2f * res.DW.Storage[i];
                    else
                        t.DW.Storage[i] = 0f;
                }
            });


            return res;
        }

        public virtual Tensor HardTanh(Tensor t, float min, float max)
        {
            var res = t.Empty();


            for (var i = 0; i < res.W.Storage.Length; i++)
            {
                var val = t.W.Storage[i];
                var rval = min;
                if (val >= max)
                    rval = max;
                else if (val > min)
                    rval = val;
                res.W.Storage[i] = rval;
            }


            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.DW.Storage.Length; i++)
                {
                    var val = t.W.Storage[i];
                    if (val < max && val > min)
                        t.DW.Storage[i] = res.DW.Storage[i];
                    else
                        t.DW.Storage[i] = 0;
                }
            });


            return res;
        }


        public virtual Tensor Rectifier(Tensor t)
        {
            var res = t.Empty();


            for (var i = 0; i < res.W.Storage.Length; i++)
                res.W.Storage[i] = t.W.Storage[i] <= 0 ? 0 : t.W.Storage[i];


            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.DW.Storage.Length; i++)
                    t.DW.Storage[i] += t.W.Storage[i] <= 0 ? 0 : res.DW.Storage[i];
            });


            return res;
        }

        public virtual Tensor Rectifier(Tensor t, int[] inds)
        {
            var res = t.Empty();

            for (var i = 0; i < res.Row; i++)
            {
                foreach (var j in inds)
                    res.W[i, j] = t.W[i, j] <= 0 ? 0 : t.W[i, j];
            }
            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.Row; i++)
                {
                    foreach (var j in inds)
                        t.DW[i, j] += t.W[i, j] <= 0 ? 0 : res.DW[i, j];
                }
            });

            return res;
        }


        public virtual Tensor Cubic(Tensor t)
        {
            var res = t.Empty();

            for (var i = 0; i < res.W.Storage.Length; i++)
                res.W.Storage[i] = t.W.Storage[i] * t.W.Storage[i] * t.W.Storage[i];

            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.DW.Storage.Length; i++)
                    t.DW.Storage[i] += 3 * t.W.Storage[i] * t.W.Storage[i] * res.DW.Storage[i];
            });


            return res;
        }

        public virtual Tensor Cubic(Tensor t, int[] inds)
        {
            var res = t.Empty();

            for (var i = 0; i < res.Row; i++)
            {
                foreach (var j in inds)
                {
                    var val = t.W[i, j];
                    res.W[i, j] = val * val * val;
                }
            }
            if (!IsTrain) return res;

            AddBackOp(delegate
            {
                for (var i = 0; i < t.Row; i++)
                {
                    foreach (var j in inds)
                        t.DW[i, j] += 3 * t.W[i, j] * t.W[i, j] * res.DW[i, j];
                }
            });


            return res;
        }


        public virtual (Tensor, Tensor) SoftmaxWithCrossEntropy(Tensor t, Tensor y = null)
        {
            var res = t.Empty();
            var loss = new Tensor(1, 1);

            for (var i = 0; i < res.Row; i++)
            {
                var max = t.W.Max(i);
                var sum = 0f;
                for (var j = 0; j < res.Col; j++)
                {
                    res.W[i, j] = (float) Math.Exp(t.W[i, j] - max);
                    sum += res.W[i, j];
                }
                for (var j = 0; j < res.Col; j++)
                    res.W[i, j] /= sum;
                var g = (int) (y?.W.Storage[i] ?? -1);
                if (g >= 0 && g < t.Col && res.W[i,g]>0)
                    loss.W.Storage[0] -= (float) Math.Log(res.W[i, g]);
            }
            loss.W.Storage[0] /= t.Row;

            if (!IsTrain || y == null) return (res, loss);

            AddBackOp(delegate
            {
                for (var i = 0; i < t.DW.Storage.Length; i++)
                    t.DW.Storage[i] += res.W.Storage[i];// / t.Row;
                var delta = 1f / t.Row;
                for (var i = 0; i < t.Row; i++)
                    t.DW[i, (int) y.W.Storage[i]] -= 1;//delta;
            });

            return (res, loss);
        }

        //public virtual (Tensor, Tensor) SoftmaxWithCrossEntropy(Tensor t, int[][] mask = null)
        //{
        //    var res = t.Empty();
        //    var loss = new Tensor(1, 1);

        //    for (var i = 0; i < t.Row; i++)
        //    {
        //        if (mask == null)
        //        {
        //            var max = t.W.Max(i);
        //            var sum = 0f;
        //            for (var j = 0; j < res.Col; j++)
        //            {
        //                res.W[i, j] = (float) Math.Exp(t.W[i, j] - max);
        //                sum += res.W[i, j];
        //            }
        //            for (var j = 0; j < res.Col; j++)
        //                res.W[i, j] /= sum;
        //        }
        //        else
        //        {
        //            var sum = 0f;
        //            var max = t.W.Max(i, mask[i]);
        //            for (var j = 0; j < res.Col; i++)
        //            {
        //                if (mask[i][j] >= 0)
        //                {
        //                    res.W[i, j] = (float) Math.Exp(t.W.Storage[i] - max);
        //                    sum += res.W[i, j];
        //                }
        //            }
        //            for (var j = 0; i < res.Col; i++)
        //            {
        //                res.W.Storage[i] /= sum;
        //                if (mask[i][j] > 0)
        //                    loss.W.Storage[0] -= (float) Math.Log(res.W[i, j]);
        //            }
        //        }
        //    }
        //    if (!IsTrain || mask == null) return (res, loss);

        //    AddBackOp(delegate
        //    {
        //        for (var i = 0; i < t.Row; i++)
        //        {
        //            for (var j = 0; j < t.Col; j++)
        //            {
        //                if (mask[i][j] >= 0)
        //                    t.DW[i, j] += (res.W[i, j] - mask[i][j]) / t.Row;
        //            }
        //        }
        //    });

        //    return (res, loss);
        //}


        public virtual Tensor Multiply(Tensor l, Tensor r)
        {
            var res = new Tensor(l.Row, r.Col);

            res.W.AddMultiply(l.W, r.W);

            if (!IsTrain) return res;

            if (l.DW != null && r.DW != null)
            {
                AddBackOp(delegate
                {
                    l.DW.AddMultiplyTransB(res.DW, r.W);
                    r.DW.AddMultiplyTransA(l.W, res.DW);
                });
            }
            else if (l.DW != null)
            {
                AddBackOp(delegate { l.DW.AddMultiplyTransB(res.DW, r.W); });
            }
            else if (r.DW != null)
            {
                AddBackOp(delegate { r.DW.AddMultiplyTransA(l.W, res.DW); });
            }

            return res;
        }

        public virtual Tensor Multiply(Tensor l, Tensor r, int[] inds)
        {
            var res = new Tensor(l.Row, r.Col);

            res.W.AddMultiply(inds, l.W, r.W);

            if (!IsTrain) return res;

            if (l.DW != null && r.DW != null)
            {
                AddBackOp(delegate
                {
                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }
            else if (l.DW != null)
            {
                AddBackOp(delegate { l.DW.AddMultiplyTransB(res.DW, inds, r.W); });
            }
            else if (r.DW != null)
            {
                AddBackOp(delegate { r.DW.AddMultiplyTransA(l.W, res.DW, inds); });
            }

            return res;
        }

        public virtual Tensor MultiplyTop(Tensor l, Tensor r, int k)
        {
            var res = new Tensor(l.Row, r.Col);


            res.W.AddMultiply(l.W, r.W);

            if (!IsTrain) return res;
            if (l.DW != null && r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);

                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }
            else if (l.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                });
            }
            else if (r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }
            return res;
        }


        public virtual Tensor MultiplyTop(Tensor l, Tensor r, int k, int[] outinds)
        {
            var res = new Tensor(l.Row, r.Col);


            res.W.AddMultiply(outinds, l.W, r.W);

            if (!IsTrain) return res;
            if (l.DW != null && r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);

                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }
            else if (l.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                });
            }
            else if (r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }
            return res;
        }

        public virtual Tensor MultiplyTopRecord(Tensor l, Tensor r, int k, Record re)
        {
            var res = new Tensor(l.Row, r.Col);


            res.W.AddMultiply(l.W, r.W);

            if (!IsTrain) return res;
            if (l.DW != null && r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    re.Store(inds);
                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }
            else if (l.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    re.Store(inds);
                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                });
            }
            else if (r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    re.Store(inds);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }

            return res;
        }

        public virtual Tensor MultiplyTopRecord(Tensor l, Tensor r, int k, Record re, int[] outinds)
        {
            var res = new Tensor(l.Row, r.Col);


            res.W.AddMultiply(outinds, l.W, r.W);

            if (!IsTrain) return res;
            if (l.DW != null && r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    re.Store(inds);
                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }
            else if (l.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    re.Store(inds);
                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                });
            }
            else if (r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetAbsTopsHeap(res.DW, k);
                    re.Store(inds);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }

            return res;
        }

        public virtual Tensor MultiplyRand(Tensor l, Tensor r, int k)
        {
            var res = new Tensor(l.Row, r.Col);


            res.W.AddMultiply(l.W, r.W);

            if (!IsTrain) return res;
            if (l.DW != null && r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetRandK(res.Row, res.Col, k);

                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }
            else if (l.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetRandK(res.Row, res.Col, k);

                    l.DW.AddMultiplyTransB(res.DW, inds, r.W);
                });
            }
            else if (r.DW != null)
            {
                AddBackOp(delegate
                {
                    var inds = GetRandK(res.Row, res.Col, k);
                    r.DW.AddMultiplyTransA(l.W, res.DW, inds);
                });
            }

            return res;
        }


        #region Utility

        [ThreadStatic] private static List<TopNHeap> _tops;

        private int[][] GetAbsTopsHeap(Matrix m, int k)
        {
            var res = new int[m.RowDim][];
            if (_tops == null)
            {
                _tops = new List<TopNHeap>();
                while (_tops.Count < m.RowDim)
                    _tops.Add(new TopNHeap(k));
            }
            for (var i = 0; i < m.RowDim; i++)
                res[i] = _tops[i].GetAbsTop(m, i, k);
            return res;
        }

        private int[][] GetRandK(int row, int end, int k)
        {
            var res = new int[row][];
            for (var i = 0; i < row; i++)
            {
                res[i] = new int[k];
                var needed = k;
                var back = end;
                while (needed > 0)
                {
                    if (_net.Rand.GetFloat(0, 1) < (float) needed / back)
                    {
                        res[i][needed - 1] = back - 1;
                        needed--;
                    }
                    back--;
                }
            }
            return res;
        }

        #endregion Utility
    }
}
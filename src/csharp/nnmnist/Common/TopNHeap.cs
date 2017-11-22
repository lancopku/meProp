using System;
using nnmnist.Networks.Graph;

namespace nnmnist.Common
{
    internal class TopNHeap
    {
        // for fast top-k selection (the top-k indices)
        // based on min heap


        private readonly int[] _iHeap;
        private readonly float[] _vHeap;
        private readonly int _nTop;

        public TopNHeap(int nTop)
        {
            _nTop = nTop;
            _vHeap = new float[nTop];
            _iHeap = new int[nTop];
        }


        // wrapper to get the top indices from the row of the m
        // notice the returned indices array are reused in terms of memory space
        public int[] GetAbsTop(Matrix m, int row, int top)
        {
            if (top > _nTop)
                // not enough space
                // the space is fixed
                throw new ArgumentException("top too big");
            if (top > m.ColDim)
                // array has fewer elements than top
                throw new ArgumentException("value too short");
            for (var i = 0; i < top; i++)
            {
                _vHeap[i] = Math.Abs(m[row, i]);
                _iHeap[i] = i;
            }
            MakeHeap(top);
            for (var i = top; i < m.ColDim; i++)
            {
                var abs = Math.Abs(m[row, i]);
                if (abs > _vHeap[0])
                {
                    _vHeap[0] = abs;
                    _iHeap[0] = i;
                    ShiftDown(0, top);
                }
            }
            return _iHeap;
        }

        // initialize the heap
        private void MakeHeap(int top)
        {
            for (var i = top >> 1; i >= 0; i--)
                ShiftDown(i, top);
        }

        // push into elements, and adjust the heap
        private void ShiftDown(int cur, int top)
        {
            while (true)
            {
                int lc = (cur << 1) + 1, rc = lc + 1;
                if (lc >= top)
                    return;
                int nxt;
                if (rc >= top)
                    nxt = lc;
                else
                    nxt = _vHeap[lc] > _vHeap[rc] ? rc : lc;
                if (_vHeap[cur] > _vHeap[nxt])
                {
                    var ftmp = _vHeap[cur];
                    _vHeap[cur] = _vHeap[nxt];
                    _vHeap[nxt] = ftmp;
                    var itmp = _iHeap[cur];
                    _iHeap[cur] = _iHeap[nxt];
                    _iHeap[nxt] = itmp;
                    cur = nxt;
                }
                else
                    return;
            }
        }
    }
}
namespace nnmnist.Networks.Inits
{
    class IdentityInit : IInit
    {
        private readonly double _val;

        public IdentityInit(double val)
        {
            _val = val;
        }

        public double Next()
        {
            return _val;
        }

        public override string ToString()
        {
            return $"IdentityInit[{_val}]";
        }
    }
}

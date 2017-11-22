namespace nnmnist.Networks.Units
{
    interface IUnit
    {
        // the parameters need updating is stored in the net
        void SubmitParameters(NetBase net);
    }
}

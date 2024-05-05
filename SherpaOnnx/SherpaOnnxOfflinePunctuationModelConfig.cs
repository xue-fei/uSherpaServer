namespace SherpaOnnx
{
    public struct SherpaOnnxOfflinePunctuationModelConfig 
    {
        public string ctTransformer = "";
        public int numThreads = 1;
        public bool debug = true;
        public string provider = "cpu";

        public SherpaOnnxOfflinePunctuationModelConfig (string ctTransformer, int numThreads, bool debug, string provider)
        {
            this.ctTransformer = ctTransformer;
            this.numThreads = numThreads;
            this.debug = debug;
            this.provider = provider;
        }
    }
}
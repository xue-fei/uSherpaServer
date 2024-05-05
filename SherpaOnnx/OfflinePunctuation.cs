using System.Runtime.InteropServices;

namespace SherpaOnnx
{
    public class OfflinePunctuation : IDisposable
    {
        private HandleRef _handle;

        public OfflinePunctuation(SherpaOnnxOfflinePunctuationConfig config)
        {
            IntPtr intPtr = OfflinePunctuation.SherpaOnnxCreateOfflinePunctuation(config);
            this._handle = new HandleRef(this, intPtr);
        }

        public string AddPunctuation(string text)
        {
           return SherpaOfflinePunctuationAddPunct(this._handle.Handle,text);
        }

        public void Dispose()
        {
            this.Cleanup();
            GC.SuppressFinalize(this);
        }

        ~OfflinePunctuation()
        {
            this.Cleanup();
        }

        private void Cleanup()
        {
            OfflinePunctuation.DestroyOfflinePunctuation(this._handle.Handle);
            this._handle = new HandleRef(this, IntPtr.Zero);
        }

        [DllImport("sherpa-onnx-c-api", EntryPoint = "SherpaOnnxDestroyOfflinePunctuation")]
        private static extern IntPtr DestroyOfflinePunctuation(IntPtr handle);

        [DllImport("sherpa-onnx-c-api", EntryPoint = "SherpaOnnxCreateOfflinePunctuation")]
        private static extern IntPtr SherpaOnnxCreateOfflinePunctuation(SherpaOnnxOfflinePunctuationConfig config);

        [DllImport("sherpa-onnx-c-api", EntryPoint = "SherpaOfflinePunctuationAddPunct")]
        private static extern string SherpaOfflinePunctuationAddPunct(IntPtr ptr, string text);
    }
}
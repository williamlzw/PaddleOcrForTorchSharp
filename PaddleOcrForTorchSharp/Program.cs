using OpenCvSharp;
using OCREngineHelp;


namespace OCREngineTest
{
    static class Program
    {
        public static void Main()
        {
            test_engine();
        }

        public static void test_engine()
        {
            OCRParameter parameter = new OCRParameter();
            parameter.UseGpu = false;
            OCRModelConfig config = new OCRModelConfig();
            config.detModelPath = "data\\plate_det.dat";
            config.recModelPath = "data\\plate_rec.dat";
            config.clsModelPath = "data\\text_cls.dat";
            config.keysPath = "data\\ppocr_keys_v1.txt";
            OCREngine engine = new OCREngine(config, parameter);
            string imgPath = "data\\plate0.png";
            var time0 = DateTime.Now;
            OCRResult ocrResult = engine.Detect(imgPath);
            var time1 = DateTime.Now;
            Console.WriteLine((time1 - time0).TotalMilliseconds);
            engine.ShowResult(ocrResult); 
            var ret = engine.DrawResult(ocrResult, imgPath);
            Cv2.ImWrite("detresult.jpg", ret);
        }
    }
}

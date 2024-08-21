using System.Runtime.InteropServices;
using SVTRNetHelp;
using DBNetHelp;
using ConverterHelp;
using TorchSharp;
using ProcessHelp;
using OpenCvSharp;
using ClsNetHelp;

namespace OCREngineHelp
{
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public class OCRResult
    {
        /// <summary>
        /// 文本块列表
        /// </summary>
        public List<TextBlock> TextBlocks { get; set; } = new List<TextBlock>();
        /// <summary>
        /// 识别结果文本
        /// </summary>
        public string Text => this.ToString();
        public override string ToString() => string.Join("\r\n", TextBlocks.Select(x => x.Text).ToArray());

    }
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public class OCRPoint
    {
        /// <summary>
        /// X坐标，单位像素
        /// </summary>
        public int X { get; set; }
        /// <summary>
        /// Y坐标，单位像素
        /// </summary>
        public int Y { get; set; }
        public OCRPoint(int x, int y)
        {
            X = x;
            Y = y;
        }
        public override string ToString() => $"({X},{Y})";
    }
    /// <summary>
    /// 识别的文本块
    /// </summary>
    public class TextBlock
    {
        public List<OCRPoint> BoxPoints { get; set; } = new List<OCRPoint>();
        public string Text { get; set; }
        /// <summary>
        /// 得分
        /// </summary>
        public float Score { get; set; }
        public override string ToString()
        {
            string str = string.Join(",", BoxPoints.Select(x => x.ToString()).ToArray());
            return $"{Text},{Score},[{str}]";
        }
    }
    public class OCRParameter
    {
        public bool UseGpu = false;
        public int GpuId = 0;
        public double BoxThresh = 0.3;
        public double BoxScoreThresh = 0.6;
        public double UnClipRatio = 1.6;
    }

    public class OCRModelConfig
    {
        public string detModelPath;
        public string recModelPath;
        public string clsModelPath;
        public string keysPath;
    }

    public class OCREngine
    {
        private SVTR m_recModel;
        private DBNet m_detModel;
        private ClsNet m_clsModel;
        private OCRParameter m_parameter;
        private string m_character = "";
        private CTCLabelConverter m_converter;
        private torch.Device m_device;
        private torchvision.ITransform m_normalizeDetOperator;
        private torchvision.ITransform m_transformOperator;


        public OCREngine(OCRModelConfig config, OCRParameter parameter)
        {
            if (!File.Exists(config.detModelPath)) throw new FileNotFoundException(config.detModelPath);
            if (!File.Exists(config.recModelPath)) throw new FileNotFoundException(config.recModelPath);
            if (!File.Exists(config.clsModelPath)) throw new FileNotFoundException(config.clsModelPath);
            if (!File.Exists(config.keysPath)) throw new FileNotFoundException(config.keysPath);
            m_parameter = parameter;

            var description = $"cpu";
            if (parameter.UseGpu)
            {
                description = "cuda:0";
            }
            m_device = torch.device(description);

            m_recModel = new SVTR(3);
            m_recModel.load(config.recModelPath);
            m_recModel.eval();
            m_recModel.to(m_device);

            m_detModel = new DBNet(3);
            m_detModel.load(config.detModelPath);
            m_detModel.eval();
            m_detModel.to(m_device);

            m_clsModel = new ClsNet(3);
            m_clsModel.load(config.clsModelPath);
            m_clsModel.eval();
            m_clsModel.to(m_device);
            var lines = File.ReadAllLines(config.keysPath);
            foreach (var line in lines)
            {
                m_character += line.TrimEnd();
            }
            m_converter = new CTCLabelConverter(m_character);
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
            m_normalizeDetOperator = torchvision.transforms.Normalize(new double[] { 0.485, 0.456, 0.406 }, new double[] { 0.229, 0.224, 0.225 });
            m_transformOperator = torchvision.transforms.ConvertImageDType(torch.ScalarType.Float32);
        }

        public OCRResult Detect(string imagefile)
        {
            if (!System.IO.File.Exists(imagefile)) throw new Exception($"文件{imagefile}不存在");
            OCRResult result = new OCRResult();
            var detMat = Cv2.ImRead(imagefile);
            var Mat = detMat.Clone();
            int height = (int)detMat.Height;
            int width = (int)detMat.Width;
            var (resizeHeight, resizeWidth, ratioHeight, ratioWidth) = PreProcess.GetSizeAndRatio(height, width);
            Cv2.Resize(detMat, detMat, new OpenCvSharp.Size(resizeWidth, resizeHeight), interpolation: InterpolationFlags.Linear);
            byte[] detData = new byte[detMat.Total() * 3];
            Marshal.Copy(detMat.Data, detData, 0, detData.Length);
            var imgTensor = torch.tensor(detData, torch.ScalarType.Byte).reshape(detMat.Height, detMat.Width, 3).permute(2, 0, 1);

            imgTensor = m_transformOperator.call(imgTensor);
            imgTensor = m_normalizeDetOperator.call(imgTensor.unsqueeze(0));
            imgTensor = imgTensor.to(m_device);
            var resultDetTensor = m_detModel.forward(imgTensor);

            imgTensor.Dispose();
            detMat.Dispose();
            resultDetTensor = resultDetTensor.cpu();
            var boxesBatch = PostProcess.DBPostprocrss(resultDetTensor, height, width, m_parameter.BoxThresh, m_parameter.BoxScoreThresh, m_parameter.UnClipRatio);
            resultDetTensor.Dispose();
            foreach (var box in boxesBatch[0])
            {
                var xSort = box.BoxPoints.OrderBy(p => p.X).ToList();
                var ySort = box.BoxPoints.OrderBy(p => p.Y).ToList();
                Rect rc = new Rect();
                rc.X = xSort[0].X;
                rc.Width = xSort[xSort.Count - 1].X - rc.X + 1;
                rc.Y = ySort[0].Y;
                rc.Height = ySort[ySort.Count - 1].Y - rc.Y + 1;
                var boxMat = Mat[rc];
  
                Cv2.CvtColor(boxMat, boxMat, ColorConversionCodes.BGR2RGB);
                if (boxMat.Height > (int)(boxMat.Width * 1.5))
                {
                    Cv2.Rotate(boxMat, boxMat, RotateFlags.Rotate90Counterclockwise);
                }

                var clsMat = boxMat.Clone();
                Cv2.Resize(boxMat, clsMat, new OpenCvSharp.Size(192, 48), interpolation: InterpolationFlags.Linear);
                byte[] clsData = new byte[clsMat.Total() * clsMat.ElemSize()];
                Marshal.Copy(clsMat.Data, clsData, 0, clsData.Length);
                var clsTensor = torch.tensor(clsData, torch.ScalarType.Byte).reshape(clsMat.Height, clsMat.Width, 3).permute(2, 0, 1).unsqueeze(0);
                clsTensor = m_transformOperator.call(clsTensor);
                clsTensor = clsTensor.to(m_device);

                var resultClsTensor = m_clsModel.forward(clsTensor);

                clsTensor.Dispose();
                var direction = PostProcess.ClsPostprocess(resultClsTensor);
                resultClsTensor.Dispose();
                if (direction == 1)
                {
                    Cv2.Rotate(clsMat, clsMat, RotateFlags.Rotate180);
                }

                Cv2.Resize(clsMat, clsMat, new OpenCvSharp.Size(320, 48), interpolation: InterpolationFlags.Linear);
                byte[] recData = new byte[clsMat.Total() * clsMat.ElemSize()];
                Marshal.Copy(clsMat.Data, recData, 0, recData.Length);

                var recTensor = torch.tensor(recData, torch.ScalarType.Byte).reshape(clsMat.Height, clsMat.Width, 3).permute(2, 0, 1).unsqueeze(0);
                recTensor = m_transformOperator.call(recTensor);
                recTensor = recTensor.to(m_device);

                var resultRecTensor = m_recModel.forward(recTensor);

                recTensor.Dispose();
                clsMat.Dispose();
                var resultList = m_converter.Decode(resultRecTensor);
                resultRecTensor.Dispose();
                if (resultList[0].Item2.Count > 0)
                {
                    TextBlock textBlock = new TextBlock();
                    textBlock.Text = resultList[0].Item1;
                    textBlock.BoxPoints = box.BoxPoints;
                    textBlock.Score = resultList[0].Item2.Average();
                    result.TextBlocks.Add(textBlock);
                }
            }
            return result;
        }

        public void ShowResult(OCRResult ocrResult)
        {
            Console.WriteLine(ocrResult.ToString());
        }

        public Mat DrawResult(OCRResult ocrResult, string imgPath)
        {
            var img = Cv2.ImRead(imgPath);
            List<List<Point>> polys = new List<List<Point>>();
            foreach (var textBlock in ocrResult.TextBlocks)
            {
                List<Point> points = new List<Point>();
                foreach (var point in textBlock.BoxPoints)
                {
                    var pt = new Point(point.X, point.Y);
                    points.Add(pt);
                }
                polys.Add(points);
            }
            Cv2.Polylines(img, polys, true, OpenCvSharp.Scalar.Chocolate, thickness: 2);
            return img;
        }
    }
}

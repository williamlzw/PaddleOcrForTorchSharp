using System;
using TorchSharp;
using OpenCvSharp;
using static TorchSharp.torch;
using System.Runtime.InteropServices;
using ClipperLib;
using OCREngineHelp;

namespace ProcessHelp
{
    public static class PreProcess
    {
        public static (int, int, double, double) GetSizeAndRatio(int height, int width)
        {
            var max = Math.Max(height, width);
            double ratio;
            if (max > 960)
            {
                if (height > width)
                {
                    ratio = (double)960 / height;
                }
                else
                {
                    ratio = (double)960 / width;
                }
            }
            else
            {
                ratio = 1;
            }
            int resize_h = (int)(height * ratio);
            int resize_w = (int)(width * ratio);
            resize_h = Math.Max((int)Math.Round((double)resize_h / 32) * 32, 32);
            resize_w = Math.Max((int)Math.Round((double)resize_w / 32) * 32, 32);
            double ratio_h = (double)resize_h / height;
            double ratio_w = (double)resize_w / width;
            return (resize_h, resize_w, ratio_h, ratio_w);
        }
    }

    public static class PostProcess
    {
        public static long ClsPostprocess(torch.Tensor pred)
        {
            var pred_idxs = pred.cpu().argmax(dim: 1);
            var idx = pred_idxs.ToInt64();
            return idx;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="pred"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="boxThresh">用于过滤DB预测的二值化图像，设置为0.-0.3对结果影响不明显；默认0.3</param>
        /// <param name="boxScoreThresh">DB后处理过滤box的阈值，如果检测存在漏框情况，可酌情减小；默认0.6</param>
        /// <param name="unclipRatio">表示文本框的紧致程度，越小则文本框更靠近文本;默认1.6</param>
        /// <returns></returns>
        public static List<List<TextBlock>> DBPostprocrss(torch.Tensor pred, int height, int width, double boxThresh = 0.3, double boxScoreThresh = 0.6, double unclipRatio = 1.6)
        {
            pred = pred[TensorIndex.Colon, TensorIndex.Single(0), TensorIndex.Colon, TensorIndex.Colon];
            List<List<TextBlock>> textBlocksBatch = new List<List<TextBlock>>();
            var segmentation = pred > boxThresh;
            for (int i = 0; i < pred.shape[0]; i++)
            {
                var pred_idx = pred[i];
                var mask = segmentation[i];
                var textBlocks = BoxesFromBitmap(pred_idx, mask, width, height, boxScoreThresh, unclipRatio);
                textBlocksBatch.Add(textBlocks);
            }
            return textBlocksBatch;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mat"></param>
        /// <returns>shape[C, H, W]</returns>
        public static torch.Tensor MatToTensor(Mat mat)
        {
            torch.Tensor tensor = null;
            if (mat.Type() == MatType.CV_8UC3)
            {
                byte[] data = new byte[mat.Total() * 3];
                Marshal.Copy(mat.Data, data, 0, data.Length);
                tensor = torch.tensor(data, ScalarType.Byte).reshape(mat.Height, mat.Width, 3).permute(2, 0, 1);
            }
            else if (mat.Type() == MatType.CV_8UC1)
            {
                byte[] data = new byte[mat.Total() * 1];
                Marshal.Copy(mat.Data, data, 0, data.Length);
                tensor = torch.tensor(data, ScalarType.Byte).reshape(mat.Height, mat.Width, 1).permute(2, 0, 1);
            }
            return tensor;
        }

        public static Mat TensorToMat(torch.Tensor tensor, MatType type)
        {
            var channel = tensor.shape[0];
            var height = tensor.shape[1];
            var width = tensor.shape[2];
            var img = tensor.permute(1, 2, 0);
            Mat mat = null;
            if (channel == 3)
            {
                mat = new Mat((int)height, (int)width, type);
            }
            else if (channel == 1)
            {
                mat = new Mat((int)height, (int)width, type);
            }
            var access = img.data<byte>();
            var data = access.ToArray();
            Marshal.Copy(data, 0, mat.Data, data.Length);
            return mat;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="tensor">shape[C, H, W]</param>
        /// <returns></returns>
        public static Mat TensorToMat(torch.Tensor tensor)
        {
            var channel = tensor.shape[0];
            var height = tensor.shape[1];
            var width = tensor.shape[2];
            var img = tensor.permute(1, 2, 0);
            Mat mat = null;
            if (channel == 3)
            {
                mat = new Mat((int)height, (int)width, MatType.CV_8UC3);
            }
            else if (channel == 1)
            {
                mat = new Mat((int)height, (int)width, MatType.CV_8UC1);
            }
            var access = img.data<byte>();
            var data = access.ToArray();
            Marshal.Copy(data, 0, mat.Data, data.Length);
            return mat;
        }

        static (List<Point2f>, float) GetMiniBoxes(Point[] contour)
        {
            var boundingBox = Cv2.MinAreaRect(contour);
            var boxPoints = Cv2.BoxPoints(boundingBox);
            var points = boxPoints.OrderBy(e => e.X).ToList<Point2f>();
            int index_1 = 0;
            int index_2 = 1;
            int index_3 = 2;
            int index_4 = 3;
            if (points[1].Y > points[0].Y)
            {
                index_1 = 0;
                index_4 = 1;
            }
            else
            {
                index_1 = 1;
                index_4 = 0;
            }
            if (points[3].Y > points[2].Y)
            {
                index_2 = 2;
                index_3 = 3;
            }
            else
            {
                index_2 = 3;
                index_3 = 2;
            }
            List<Point2f> box = new List<Point2f>();
            box.Add(points[index_1]);
            box.Add(points[index_2]);
            box.Add(points[index_3]);
            box.Add(points[index_4]);
            var minValue = Math.Min(boundingBox.Size.Width, boundingBox.Size.Height);
            return (box, minValue);
        }


        static double BoxScore(Mat pred, List<Point2f> box)
        {
            long height = pred.Height;
            long width = pred.Width;
            var boxArr = box.ToArray();
            Array.Sort(boxArr, new Comparison<Point2f>((p1, p2) => p1.X.CompareTo(p2.X)));
            int xmin = (int)Math.Floor(boxArr[0].X);
            int xmax = (int)Math.Floor(boxArr[boxArr.Length - 1].X);

            Array.Sort(boxArr, new Comparison<Point2f>((p1, p2) => p1.Y.CompareTo(p2.Y)));
            int ymin = (int)Math.Floor(boxArr[0].Y);
            int ymax = (int)Math.Floor(boxArr[boxArr.Length - 1].Y);

            xmin = (int)Math.Clamp(xmin, 0, width - 1);
            xmax = (int)Math.Clamp(xmax, 0, width - 1);
            ymin = (int)Math.Clamp(ymin, 0, height - 1);
            ymax = (int)Math.Clamp(ymax, 0, height - 1);

            List<Point> newBox = new List<Point>();
            for (int i = 0; i < boxArr.Length; i++)
            {
                var index = boxArr[i];
                Point point = new Point(index.X - xmin, index.Y - ymin);
                newBox.Add(point);
            }

            Mat mask = new Mat((ymax - ymin + 1), (xmax - xmin + 1), MatType.CV_8UC1, OpenCvSharp.Scalar.All(0));
            List<List<Point>> poly = new List<List<Point>>();
            poly.Add(newBox);
            Cv2.FillPoly(mask, poly, OpenCvSharp.Scalar.All(1));
            Rect area = new Rect();
            area.X = xmin;
            area.Width = xmax + 1 - xmin;
            area.Y = ymin;
            area.Height = ymax + 1 - ymin;
            var bitmap = pred[area];
            
            var value = Cv2.Mean(bitmap, mask)[0];
            return value;
        }


        static Point[] UnClip(List<Point2f> box, double unclipRatio = 1.6)
        {
            List<IntPoint> point = new List<IntPoint>();
            foreach (var index in box)
            {
                IntPoint pt3 = new IntPoint();
                pt3.X = (int)index.X;
                pt3.Y = (int)index.Y;
                point.Add(pt3);
            }

            var area = Clipper.Area(point);

            List<List<IntPoint>> polys = new List<List<IntPoint>>();
            polys.Add(point);

            var len = PolygonLength(point);
            double delta = area * unclipRatio / len;

            var expanded = Clipper.OffsetPolygons(polys, delta, JoinType.jtRound);

            List<Point> retBox = new List<Point>();
            foreach (var pathIndex in expanded)
            {
                foreach (var pt in pathIndex)
                {
                    Point pt2 = new Point();
                    pt2.X = (int)pt.X;
                    pt2.Y = (int)pt.Y;
                    retBox.Add(pt2);
                }
            }
            return retBox.ToArray();
        }

        static double PolygonLength(List<IntPoint> points)
        {
            List<IntPoint> newPoints = points;
            newPoints.Add(points[0]);
            double len = 0;
            double x1, x2, y1, y2;
            for (int i = 1; i < newPoints.Count; i++)
            {
                x1 = points[i - 1].X;
                x2 = points[i].X;
                y1 = points[i - 1].Y;
                y2 = points[i].Y;
                len += Math.Pow((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1), 0.5);
            }
            return len;
        }

        /// <summary>
        /// The positions of the geometric centroid of a planar non-self-intersecting polygon with vertices (x1,y1), ..., (xn,yn)
        /// </summary>
        /// <param name="points"></param>
        /// <returns></returns>
        static Point PolygonCentroid(List<IntPoint> points)
        {
            double area = PolygonArea(points);
            double x = 0;
            double y = 0;
            double x1, x2, y1, y2;
            for (int i = 1; i < points.Count; i++)
            {
                x1 = points[i - 1].X;
                x2 = points[i].X;
                y1 = points[i - 1].Y;
                y2 = points[i].Y;
                x += (x1 + x2) * Determinant(x1, y1, x2, y2);
                y += (y1 + y2) * Determinant(x1, y1, x2, y2);
            }
            x /= 6 * area;
            y /= 6 * area;
            return new Point(x, y);
        }

        static double Determinant(double x1, double y1, double x2, double y2)
        {
            return x1 * y2 - x2 * y1;
        }

        /// <summary>
        /// The (signed) area of a planar non-self-intersecting polygon with vertices (x1,y1), ..., (xn,yn)
        /// </summary>
        /// <param name="vertices"></param>
        /// <returns>Note that the area of a convex polygon is defined to be positive if the points are arranged in a counterclockwise order and negative if they are in clockwise order (Beyer 1987).</returns>
        static double PolygonArea(List<IntPoint> vertices)
        {
            if (vertices.Count < 3)
            {
                return 0;
            }
            double area = Determinant(vertices[vertices.Count - 1].X, vertices[vertices.Count - 1].Y, vertices[0].X, vertices[0].Y);
            for (int i = 1; i < vertices.Count; i++)
            {
                area += Determinant(vertices[i - 1].X, vertices[i - 1].Y, vertices[i].X, vertices[i].Y);
            }
            return area / 2;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="pred"></param>
        /// <param name="mask"></param>
        /// <param name="dstWidth"></param>
        /// <param name="dstHeight"></param>
        /// <param name="boxScoreThresh">DB后处理过滤box的阈值，如果检测存在漏框情况，可酌情减小；默认0.6</param>
        /// <param name="unclipRatio">表示文本框的紧致程度，越小则文本框更靠近文本;默认1.6</param>
        /// <returns></returns>
        static List<TextBlock> BoxesFromBitmap(torch.Tensor pred, torch.Tensor mask, int dstWidth, int dstHeight, double boxScoreThresh = 0.6, double unclipRatio = 1.6)
        {
            var bitmap = (mask * 255).to(torch.ScalarType.Byte).unsqueeze(0);
            var bitmapMat = TensorToMat(bitmap);
            var predMat = bitmapMat / 255;
            var height = bitmapMat.Height;
            var width = bitmapMat.Width;
            
            Cv2.FindContours(bitmapMat, out OpenCvSharp.Point[][] contours, out HierarchyIndex[] outputArray, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
            var numContours = Math.Min(contours.Length, 100);
            List<TextBlock> textBlocks = new List<TextBlock>();

            for (int i = 0; i < numContours; i++)
            {
                var contour = contours[i];

                var (box, sside) = GetMiniBoxes(contour);

                if (sside < 3)
                {
                    continue;
                }

                var score = BoxScore(predMat, box);

                if (boxScoreThresh > score)
                {
                    continue;
                }

                var clip_contour = UnClip(box, unclipRatio);

                var (clip_box, clip_sside) = GetMiniBoxes(clip_contour);

                if (clip_sside < 5)
                {
                    continue;
                }
                TextBlock textBlock = new TextBlock();
                foreach (var index in clip_box)
                {
                    Point pt = new();
                    pt.X = (int)Math.Clamp(Math.Round((double)(index.X / width) * dstWidth), 0, dstWidth);
                    pt.Y = (int)Math.Clamp(Math.Round((double)(index.Y / height) * dstHeight), 0, dstHeight);
                    textBlock.BoxPoints.Add(new OCRPoint(pt.X, pt.Y));
                }
                textBlocks.Add(textBlock);
            }
            return textBlocks;
        }
    }
}

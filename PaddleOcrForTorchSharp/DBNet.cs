using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace DBNetHelp
{
    public class Hswish : Module<Tensor, Tensor>
    {
        private readonly bool _inplace;
        public Hswish(bool inplace = true) : base("Hswish")
        {
            _inplace = inplace;
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            return input * nn.functional.relu6(input + 3, inplace: _inplace) / 6;
        }
    }

    public class Hsigmoid : Module<Tensor, Tensor>
    {
        private readonly bool _inplace;
        public Hsigmoid(bool inplace = true) : base("Hsigmoid")
        {
            _inplace = inplace;
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            return nn.functional.relu6(1.2 * input + 3, inplace: _inplace) / 6;
        }
    }

    public class ConvBNLayer : Module<Tensor, Tensor>
    {
        private readonly Conv2d conv;
        private readonly BatchNorm2d bn;
        private readonly Module<Tensor, Tensor> act = null;
        private readonly bool _if_act;
        public ConvBNLayer(long in_channels, long out_channels, (long, long) kernel_size, (long, long) stride, (long, long) padding, long groups = 1, string actType = "relu", bool if_act = true) : base("ConvBNLayer")
        {
            _if_act = if_act;
            conv = Conv2d(in_channels, out_channels, kernel_size, stride: stride, dilation: (1, 1), padding: padding, groups: groups, bias: false);
            bn = BatchNorm2d(out_channels);
            if (if_act)
            {
                if (actType == "relu")
                {
                    act = ReLU(inplace: true);
                }
                else if (actType == "hard_swish")
                {
                    act = new Hswish(inplace: true);
                }
                else if (actType == "hard_sigmoid")
                {
                    act = new Hsigmoid(inplace: true);
                }
            }
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = bn.forward(conv.forward(input));
            if (_if_act)
            {
                x = act.forward(x);
            }
            return x;
        }
    }

    public class SEModule : Module<Tensor, Tensor>
    {
        private readonly AdaptiveAvgPool2d avg_pool;
        private readonly Conv2d conv1;
        private readonly Conv2d conv2;
        private readonly ReLU relu1;
        private readonly Hardsigmoid hard_sigmoid;
        public SEModule(long channel, long reduction = 4) : base("SEModule")
        {
            avg_pool = AdaptiveAvgPool2d(1);
            conv1 = Conv2d(channel, channel / reduction, kernelSize: 1);
            conv2 = Conv2d(channel / reduction, channel, kernelSize: 1);
            relu1 = ReLU(inplace: true);
            hard_sigmoid = Hardsigmoid(inplace: true);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = avg_pool.forward(input);
            x = conv1.forward(x);
            x = relu1.forward(x);
            x = conv2.forward(x);
            x = hard_sigmoid.forward(x);
            x = input * x;
            return x;
        }
    }

    public class ResidualUnit : Module<Tensor, Tensor>
    {
        private readonly ConvBNLayer expand_conv;
        private readonly ConvBNLayer bottleneck_conv;
        private readonly SEModule mid_se = null;
        private readonly ConvBNLayer linear_conv;
        private readonly bool _use_se = false;
        private readonly bool _short_cut = false;
        public ResidualUnit(long in_channels, long mid_channels, long out_channels, (long, long) kernel_size, (long, long) stride, bool use_se, string actType) : base("ResidualUnit")
        {
            if (stride == (1, 1) && in_channels == out_channels)
            {
                _short_cut = true;
            }
            expand_conv = new ConvBNLayer(in_channels, mid_channels, kernel_size: (1, 1), stride: (1, 1), padding: (0, 0), actType: actType);
            var (k1, k2) = kernel_size;
            var k = (int)(k1 - 1) / 2;
            bottleneck_conv = new ConvBNLayer(mid_channels, mid_channels, kernel_size: kernel_size, stride: stride, padding: (k, k), groups: mid_channels, actType: actType);
            _use_se = use_se;
            if (use_se)
            {
                mid_se = new SEModule(mid_channels);
            }
            linear_conv = new ConvBNLayer(mid_channels, out_channels, kernel_size: (1, 1), stride: (1, 1), padding: (0, 0), actType: actType, if_act: false);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = expand_conv.forward(input);
            x = bottleneck_conv.forward(x);
            if (_use_se)
            {
                x = mid_se.forward(x);
            }
            x = linear_conv.forward(x);
            if (_short_cut)
            {
                x = input + x;
            }
            return x;
        }
    }

    public class MobileNetV3 : Module<Tensor, List<Tensor>>
    {
        private readonly ConvBNLayer conv;
        private readonly ModuleList<Module<Tensor, Tensor>> stages = new ModuleList<Module<Tensor, Tensor>>();
        public readonly List<long> _out_channels = new List<long>();
        private static long make_divisible(double v, long divisor = 8)
        {
            long min_value = divisor;
            long new_v = Math.Max(min_value, (long)(((long)(v + 4) / divisor) * divisor));
            if (new_v < (long)(0.9 * v))
            {
                new_v += divisor;
            }
            return new_v;
        }
        public MobileNetV3(long in_channels, double scale = 0.5) : base("MobileNetV3")
        {
            long inplanes = 16;
            conv = new ConvBNLayer(in_channels, make_divisible(inplanes * scale), kernel_size: (3, 3), stride: (2, 2), padding: (1, 1), actType: "hard_swish");

            inplanes = make_divisible(scale * inplanes);

            var block_list = new List<(string, Module<Tensor, Tensor>)>();
            block_list.Add(("0", new ResidualUnit(inplanes, make_divisible(scale * 16), make_divisible(scale * 16), kernel_size: (3, 3), stride: (1, 1), false, actType: "relu")));//i 0
            inplanes = make_divisible(16 * scale);

            block_list.Add(("1", new ResidualUnit(inplanes, make_divisible(scale * 64), make_divisible(scale * 24), kernel_size: (3, 3), stride: (2, 2), false, actType: "relu")));//i 1
            inplanes = make_divisible(24 * scale);

            block_list.Add(("2", new ResidualUnit(inplanes, make_divisible(scale * 72), make_divisible(scale * 24), kernel_size: (3, 3), stride: (1, 1), false, actType: "relu")));//i 2
            inplanes = make_divisible(24 * scale);

            _out_channels.Add(inplanes);
            stages.Add(Sequential(block_list));
            block_list.Clear();

            block_list.Add(("0", new ResidualUnit(inplanes, make_divisible(scale * 72), make_divisible(scale * 40), kernel_size: (5, 5), stride: (2, 2), false, actType: "relu")));//i 3
            inplanes = make_divisible(40 * scale);

            block_list.Add(("1", new ResidualUnit(inplanes, make_divisible(scale * 120), make_divisible(scale * 40), kernel_size: (5, 5), stride: (1, 1), false, actType: "relu")));//i 4
            inplanes = make_divisible(40 * scale);

            block_list.Add(("2", new ResidualUnit(inplanes, make_divisible(scale * 120), make_divisible(scale * 40), kernel_size: (5, 5), stride: (1, 1), false, actType: "relu")));//i 5
            inplanes = make_divisible(40 * scale);

            _out_channels.Add(inplanes);
            stages.Add(Sequential(block_list));
            block_list.Clear();


            block_list.Add(("0", new ResidualUnit(inplanes, make_divisible(scale * 240), make_divisible(scale * 80), kernel_size: (3, 3), stride: (2, 2), false, actType: "hard_swish")));//i 6
            inplanes = make_divisible(80 * scale);

            block_list.Add(("1", new ResidualUnit(inplanes, make_divisible(scale * 200), make_divisible(scale * 80), kernel_size: (3, 3), stride: (1, 1), false, actType: "hard_swish")));//i 7
            inplanes = make_divisible(80 * scale);

            block_list.Add(("2", new ResidualUnit(inplanes, make_divisible(scale * 184), make_divisible(scale * 80), kernel_size: (3, 3), stride: (1, 1), false, actType: "hard_swish")));//i 8
            inplanes = make_divisible(80 * scale);

            block_list.Add(("3", new ResidualUnit(inplanes, make_divisible(scale * 184), make_divisible(scale * 80), kernel_size: (3, 3), stride: (1, 1), false, actType: "hard_swish")));//i 9
            inplanes = make_divisible(80 * scale);

            block_list.Add(("4", new ResidualUnit(inplanes, make_divisible(scale * 480), make_divisible(scale * 112), kernel_size: (3, 3), stride: (1, 1), false, actType: "hard_swish")));//i 10
            inplanes = make_divisible(112 * scale);

            block_list.Add(("5", new ResidualUnit(inplanes, make_divisible(scale * 672), make_divisible(scale * 112), kernel_size: (3, 3), stride: (1, 1), false, actType: "hard_swish")));//i 11
            inplanes = make_divisible(112 * scale);

            _out_channels.Add(inplanes);
            stages.Add(Sequential(block_list));
            block_list.Clear();

            block_list.Add(("0", new ResidualUnit(inplanes, make_divisible(scale * 672), make_divisible(scale * 160), kernel_size: (5, 5), stride: (2, 2), false, actType: "hard_swish")));//i 12
            inplanes = make_divisible(160 * scale);

            block_list.Add(("1", new ResidualUnit(inplanes, make_divisible(scale * 960), make_divisible(scale * 160), kernel_size: (5, 5), stride: (1, 1), false, actType: "hard_swish")));//i 13
            inplanes = make_divisible(160 * scale);

            block_list.Add(("2", new ResidualUnit(inplanes, make_divisible(scale * 960), make_divisible(scale * 160), kernel_size: (5, 5), stride: (1, 1), false, actType: "hard_swish")));//i 14
            inplanes = make_divisible(160 * scale);

            block_list.Add(("3", new ConvBNLayer(inplanes, make_divisible(scale * 960), kernel_size: (1, 1), stride: (1, 1), padding: (0, 0), actType: "hard_swish")));

            _out_channels.Add(make_divisible(scale * 960));
            stages.Add(Sequential(block_list));
            block_list.Clear();

            RegisterComponents();
        }

        public override List<Tensor> forward(Tensor input)
        {
            var x = conv.forward(input);
            List<Tensor> out_list = new List<Tensor>();
            foreach (var stage in stages)
            {
                x = stage.forward(x);
                out_list.Add(x);
            }
            return out_list;
        }
    }

    public class RSELayer : Module<Tensor, Tensor>
    {
        public readonly long _out_channels;
        private readonly Conv2d in_conv;
        private readonly SEModule se_block;
        private readonly bool _shortcut;
        public RSELayer(long in_channels, long out_channels, long kernel_size, bool shortcut = true) : base("RSELayer")
        {
            _out_channels = out_channels;
            var padding = (int)(kernel_size / 2);
            in_conv = Conv2d(in_channels, out_channels, kernelSize: kernel_size, padding: padding, bias: false);
            se_block = new SEModule(out_channels);
            _shortcut = shortcut;
            RegisterComponents();
        }
        public override Tensor forward(Tensor input)
        {
            var x = in_conv.forward(input);
            torch.Tensor output;
            if (_shortcut)
            {
                output = x + se_block.forward(x);
            }
            else
            {
                output = se_block.forward(x);
            }
            return output;
        }
    }

    public class RSEFPN : Module<List<Tensor>, Tensor>
    {
        private readonly ModuleList<Module<Tensor, Tensor>> ins_conv = new ModuleList<Module<Tensor, Tensor>>();
        private readonly ModuleList<Module<Tensor, Tensor>> inp_conv = new ModuleList<Module<Tensor, Tensor>>();
        public readonly long _out_channels;
        public RSEFPN(List<long> in_channels, long out_channels, bool shortcut = true) : base("RSEFPN")
        {
            _out_channels = out_channels;
            for (int i = 0; i < in_channels.Count; i++)
            {
                ins_conv.Add(new RSELayer(in_channels[i], out_channels, 1, shortcut));
                inp_conv.Add(new RSELayer(out_channels, out_channels / 4, 3, shortcut));
            }

            RegisterComponents();
        }

        public override Tensor forward(List<Tensor> input)
        {
            var c2 = input[0];
            var c3 = input[1];
            var c4 = input[2];
            var c5 = input[3];
            var in5 = ins_conv[3].forward(c5);
            var in4 = ins_conv[2].forward(c4);
            var in3 = ins_conv[1].forward(c3);
            var in2 = ins_conv[0].forward(c2);
            double[] scale_factor = { 2, 2 };
            var out4 = in4 + nn.functional.interpolate(in5, scale_factor: scale_factor, mode: InterpolationMode.Nearest);
            var out3 = in3 + nn.functional.interpolate(out4, scale_factor: scale_factor, mode: InterpolationMode.Nearest);
            var out2 = in2 + nn.functional.interpolate(out3, scale_factor: scale_factor, mode: InterpolationMode.Nearest);
            var p5 = inp_conv[3].forward(in5);
            var p4 = inp_conv[2].forward(out4);
            var p3 = inp_conv[1].forward(out3);
            var p2 = inp_conv[0].forward(out2);
            double[] scale_factor_p5 = { 8, 8 };
            p5 = nn.functional.interpolate(p5, scale_factor: scale_factor_p5, mode: InterpolationMode.Nearest);
            double[] scale_factor_p4 = { 4, 4 };
            p4 = nn.functional.interpolate(p4, scale_factor: scale_factor_p4, mode: InterpolationMode.Nearest);
            double[] scale_factor_p3 = { 2, 2 };
            p3 = nn.functional.interpolate(p3, scale_factor: scale_factor_p3, mode: InterpolationMode.Nearest);
            List<Tensor> cat = new List<Tensor>();
            cat.Add(p5);
            cat.Add(p4);
            cat.Add(p3);
            cat.Add(p2);
            var fuse = torch.cat(cat, 1);
            return fuse;
        }
    }

    public class Head : Module<Tensor, Tensor>
    {
        private readonly Conv2d conv1;
        private readonly BatchNorm2d conv_bn1;
        private readonly ReLU relu1;
        private readonly ConvTranspose2d conv2;
        private readonly BatchNorm2d conv_bn2;
        private readonly ReLU relu2;
        private readonly ConvTranspose2d conv3;
        public Head(long in_channels, long k = 50) : base("Head")
        {
            conv1 = Conv2d(in_channels, in_channels / 4, kernelSize: 3, padding: 1, bias: false);
            conv_bn1 = BatchNorm2d(in_channels / 4);
            relu1 = ReLU(inplace: true);
            conv2 = ConvTranspose2d(in_channels / 4, in_channels / 4, kernelSize: 2, stride: 2);
            conv_bn2 = BatchNorm2d(in_channels / 4);
            relu2 = ReLU(inplace: true);
            conv3 = ConvTranspose2d(in_channels / 4, 1, kernelSize: 2, stride: 2);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = conv1.forward(input);
            x = conv_bn1.forward(x);
            x = relu1.forward(x);
            x = conv2.forward(x);
            x = conv_bn2.forward(x);
            x = relu2.forward(x);
            x = conv3.forward(x);
            x = torch.sigmoid(x);
            return x;
        }
    }

    public class DBHead : Module<Tensor, Tensor>
    {
        private readonly long _k;
        private readonly Head binarize;
        private readonly Head thresh;
        public DBHead(long in_channels, long k = 50) : base("DBHead")
        {
            _k = k;
            binarize = new Head(in_channels);
            thresh = new Head(in_channels);
            RegisterComponents();
        }

        private Tensor step_function(Tensor x, Tensor y)
        {
            return torch.reciprocal(1 + torch.exp(-_k * (x - y)));
        }

        public override Tensor forward(Tensor input)
        {
            var shrink_maps = binarize.forward(input);
            if (training == false)
            {
                return shrink_maps;
            }
            else
            {
                var threshold_maps = thresh.forward(input);
                var binary_maps = step_function(shrink_maps, threshold_maps);
                List<Tensor> cat = new List<Tensor>();
                cat.Add(shrink_maps);
                cat.Add(threshold_maps);
                cat.Add(binary_maps);
                var y = torch.cat(cat, 1);
                return y;
            }
        }
    }

    public class DBNet : Module<Tensor, Tensor>
    {
        private readonly MobileNetV3 backbone;
        private readonly RSEFPN neck;
        private readonly DBHead head;
        public DBNet(long in_channels) : base("DBNet")
        {
            backbone = new MobileNetV3(in_channels);
            neck = new RSEFPN(backbone._out_channels, 96);
            head = new DBHead(neck._out_channels);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var _ = NewDisposeScope();
            var backbone_out = backbone.forward(input);
            var neck_out = neck.forward(backbone_out);
            var head_out = head.forward(neck_out);
            return head_out.MoveToOuterDisposeScope();
        }
    }
}

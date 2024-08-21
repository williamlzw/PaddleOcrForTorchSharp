using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace ClsNetHelp
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

    public class MobileNetV3 : Module<Tensor, Tensor>
    {
        private readonly ConvBNLayer conv1;
        private readonly ConvBNLayer conv2;
        private readonly Sequential blocks;
        private readonly MaxPool2d pool;
        public readonly long _out_channels;
        
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
        public MobileNetV3(long in_channels, double scale = 0.35) : base("MobileNetV3")
        {
            long inplanes = 16;
            conv1 = new ConvBNLayer(in_channels, make_divisible(inplanes * scale), kernel_size: (3, 3), stride: (2, 2), padding: (1, 1), actType: "hard_swish");

            inplanes = make_divisible(scale * inplanes);

            var block_list = new List<(string, Module<Tensor, Tensor>)>();
            block_list.Add(("0", new ResidualUnit(inplanes, make_divisible(scale * 16), make_divisible(scale * 16), kernel_size: (3, 3), stride: (2, 1), true, actType: "relu")));//i 0
            inplanes = make_divisible(16 * scale);

            block_list.Add(("1", new ResidualUnit(inplanes, make_divisible(scale * 72), make_divisible(scale * 24), kernel_size: (3, 3), stride: (2, 1), false, actType: "relu")));//i 1
            inplanes = make_divisible(24 * scale);

            block_list.Add(("2", new ResidualUnit(inplanes, make_divisible(scale * 88), make_divisible(scale * 24), kernel_size: (3, 3), stride: (1, 1), false, actType: "relu")));//i 2
            inplanes = make_divisible(24 * scale);

            block_list.Add(("3", new ResidualUnit(inplanes, make_divisible(scale * 96), make_divisible(scale * 40), kernel_size: (5, 5), stride: (2, 1), true, actType: "hard_swish")));//i 3
            inplanes = make_divisible(40 * scale);

            block_list.Add(("4", new ResidualUnit(inplanes, make_divisible(scale * 240), make_divisible(scale * 40), kernel_size: (5, 5), stride: (1, 1), true, actType: "hard_swish")));//i 4
            inplanes = make_divisible(40 * scale);

            block_list.Add(("5", new ResidualUnit(inplanes, make_divisible(scale * 240), make_divisible(scale * 40), kernel_size: (5, 5), stride: (1, 1), true, actType: "hard_swish")));//i 4
            inplanes = make_divisible(40 * scale);

            block_list.Add(("6", new ResidualUnit(inplanes, make_divisible(scale * 120), make_divisible(scale * 48), kernel_size: (5, 5), stride: (1, 1), true, actType: "hard_swish")));//i 5
            inplanes = make_divisible(48 * scale);

            block_list.Add(("7", new ResidualUnit(inplanes, make_divisible(scale * 144), make_divisible(scale * 48), kernel_size: (5, 5), stride: (1, 1), true, actType: "hard_swish")));//i 6
            inplanes = make_divisible(48 * scale);

            block_list.Add(("8", new ResidualUnit(inplanes, make_divisible(scale * 288), make_divisible(scale * 96), kernel_size: (5, 5), stride: (2, 1), true, actType: "hard_swish")));//i 7
            inplanes = make_divisible(96 * scale);

            block_list.Add(("9", new ResidualUnit(inplanes, make_divisible(scale * 576), make_divisible(scale * 96), kernel_size: (5, 5), stride: (1, 1), true, actType: "hard_swish")));//i 8
            inplanes = make_divisible(96 * scale);

            block_list.Add(("10", new ResidualUnit(inplanes, make_divisible(scale * 576), make_divisible(scale * 96), kernel_size: (5, 5), stride: (1, 1), true, actType: "hard_swish")));//i 9
            inplanes = make_divisible(96 * scale);

            blocks = Sequential(block_list);
            conv2 = new ConvBNLayer(inplanes, make_divisible(scale * 576), kernel_size: (1, 1), stride: (1, 1), padding: (0, 0), actType: "hard_swish");
            _out_channels = make_divisible(scale * 576);
            pool = MaxPool2d(2, 2);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = conv1.forward(input);
            x = blocks.forward(x);
            x = conv2.forward(x);
            x = pool.forward(x);
            return x;
        }
    }

    public class ClsHead : Module<Tensor, Tensor>
    {
        private readonly AdaptiveAvgPool2d pool;
        private readonly Linear fc;
        public ClsHead(long in_channels, long class_dim) : base("ClsHead")
        {
            pool = AdaptiveAvgPool2d(1);
            fc = Linear(in_channels, class_dim);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = pool.forward(input);
            x = torch.reshape(x, x.shape[0], x.shape[1]);
            x = fc.forward(x);
            if(training == false)
            {
                x = torch.nn.functional.softmax(x, 1);
            }
            return x;
        }
    }

    public class ClsNet : Module<Tensor, Tensor>
    {
        private readonly MobileNetV3 backbone;
        private readonly ClsHead head;
        public ClsNet(long in_channels) : base("ClsNet")
        {
            backbone = new MobileNetV3(in_channels);
            head = new ClsHead(backbone._out_channels, 2);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var _ = NewDisposeScope();
            var backbone_out = backbone.forward(input);
            var head_out = head.forward(backbone_out);
            return head_out.MoveToOuterDisposeScope();
        }
    }
}

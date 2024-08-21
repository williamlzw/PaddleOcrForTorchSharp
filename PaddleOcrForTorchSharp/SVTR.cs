using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SVTRNetHelp
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

    public class ConvBNBackBoneLayer : Module<Tensor, Tensor>
    {
        private readonly Conv2d _conv;
        private readonly BatchNorm2d _batch_norm;
        private readonly Hswish _act;
        public ConvBNBackBoneLayer(long num_channels, long num_filters, (long, long) filter_size, (long, long) stride, (long, long) padding, long groups = 1) : base("ConvBNBackBoneLayer")
        {
            _conv = Conv2d(num_channels, num_filters, filter_size, stride: stride, dilation: (1, 1), padding: padding, groups: groups, bias: false);
            _batch_norm = BatchNorm2d(num_filters);
            _act = new Hswish();
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = _conv.forward(input);
            x = _batch_norm.forward(x);
            x = _act.forward(x);
            return x;
        }
    }

    public class ConvBNLayer : Module<Tensor, Tensor>
    {
        private readonly Conv2d conv;
        private readonly BatchNorm2d norm;
        private readonly Hswish act;
        public ConvBNLayer(long num_channels, long num_filters, (long, long) filter_size, (long, long) stride, (long, long) padding, long groups = 1) : base("ConvBNLayer")
        {
            conv = Conv2d(num_channels, num_filters, filter_size, stride: stride, dilation: (1, 1), padding: padding, groups: groups, bias: false);
            norm = BatchNorm2d(num_filters);
            act = new Hswish();
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = conv.forward(input);
            x = norm.forward(x);
            x = act.forward(x);
            return x;
        }
    }

    public class SEModule : Module<Tensor, Tensor>
    {
        private readonly AdaptiveAvgPool2d avg_pool;
        private readonly Conv2d conv1;
        private readonly Conv2d conv2;
        public SEModule(long channel, long reduction = 4) : base("SEModule")
        {
            avg_pool = AdaptiveAvgPool2d(1);
            conv1 = Conv2d(channel, channel / reduction, kernelSize: 1);
            conv2 = Conv2d(channel / reduction, channel, kernelSize: 1);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = avg_pool.forward(input);
            x = conv1.forward(x);
            x = nn.functional.relu(x);
            x = conv2.forward(x);
            x = nn.functional.hardsigmoid(x);
            x = torch.mul(input, x);
            return x;
        }
    }

    public class DepthwiseSeparable : Module<Tensor, Tensor>
    {
        private readonly ConvBNBackBoneLayer _depthwise_conv;
        private readonly ConvBNBackBoneLayer _pointwise_conv;
        private readonly SEModule _se = null;
        private readonly bool _use_se;
        public DepthwiseSeparable(long num_channels, long num_filters1, long num_filters2, long num_groups, (long, long) stride, (long, long) dw_size, (long, long) padding, double scale = 0.5, bool use_se = false) : base("DepthwiseSeparable")
        {
            _depthwise_conv = new ConvBNBackBoneLayer(num_channels: num_channels, filter_size: dw_size, num_filters: (int)(num_filters1 * scale), stride: stride, padding: padding, groups: (int)(num_groups * scale));
            _pointwise_conv = new ConvBNBackBoneLayer(num_channels: (int)(num_filters1 * scale), filter_size: (1, 1), num_filters: (int)(num_filters2 * scale), stride: (1, 1), padding: (0, 0));
            if (use_se)
            {
                _se = new SEModule((int)(num_filters1 * scale));
            }
            _use_se = use_se;
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = _depthwise_conv.forward(input);
            if (_use_se)
            {
                x = _se.forward(x);
            }
            x = _pointwise_conv.forward(x);
            return x;
        }
    }

    public class MobileNetV1Enhance : Module<Tensor, Tensor>
    {
        private readonly ConvBNBackBoneLayer conv1;
        private readonly Sequential block_list;
        private readonly AvgPool2d pool;
        public readonly int out_channels;
        public MobileNetV1Enhance(long in_channels, double scale, (long, long) last_conv_stride) : base("MobileNetV1Enhance")
        {
            conv1 = new ConvBNBackBoneLayer(num_channels: in_channels, filter_size: (3, 3), num_filters: (int)(32 * scale), stride: (2, 2), padding: (1, 1));
            var modules = new List<(string, Module<Tensor, Tensor>)>();
            modules.Add(("0", new DepthwiseSeparable(num_channels: (int)(32 * scale), num_filters1: 32, num_filters2: 64, dw_size: (3, 3), padding: (1, 1), num_groups: 32, stride: (1, 1), scale: scale)));
            modules.Add(("1", new DepthwiseSeparable(num_channels: (int)(64 * scale), num_filters1: 64, num_filters2: 128, dw_size: (3, 3), padding: (1, 1), num_groups: 64, stride: (1, 1), scale: scale)));
            modules.Add(("2", new DepthwiseSeparable(num_channels: (int)(128 * scale), num_filters1: 128, num_filters2: 128, dw_size: (3, 3), padding: (1, 1), num_groups: 128, stride: (1, 1), scale: scale)));
            modules.Add(("3", new DepthwiseSeparable(num_channels: (int)(128 * scale), num_filters1: 128, num_filters2: 256, dw_size: (3, 3), padding: (1, 1), num_groups: 128, stride: (2, 1), scale: scale)));
            modules.Add(("4", new DepthwiseSeparable(num_channels: (int)(256 * scale), num_filters1: 256, num_filters2: 256, dw_size: (3, 3), padding: (1, 1), num_groups: 256, stride: (1, 1), scale: scale)));
            modules.Add(("5", new DepthwiseSeparable(num_channels: (int)(256 * scale), num_filters1: 256, num_filters2: 512, dw_size: (3, 3), padding: (1, 1), num_groups: 256, stride: (2, 1), scale: scale)));
            foreach (var i in Enumerable.Range(0, 5))
            {
                modules.Add(((6 + i).ToString(), new DepthwiseSeparable(num_channels: (int)(512 * scale), num_filters1: 512, num_filters2: 512, dw_size: (5, 5), padding: (2, 2), num_groups: 512, stride: (1, 1), scale: scale)));
            }

            modules.Add(("11", new DepthwiseSeparable(num_channels: (int)(512 * scale), num_filters1: 512, num_filters2: 1024, dw_size: (5, 5), padding: (2, 2), num_groups: 512, stride: (2, 1), scale: scale, use_se: true)));
            modules.Add(("12", new DepthwiseSeparable(num_channels: (int)(1024 * scale), num_filters1: 1024, num_filters2: 1024, dw_size: (5, 5), padding: (2, 2), num_groups: 1024, stride: last_conv_stride, scale: scale, use_se: true)));
            block_list = Sequential(modules);
            pool = AvgPool2d(2, stride: 2);
            out_channels = (int)(1024 * scale);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = conv1.forward(input);
            x = block_list.forward(x);
            x = pool.forward(x);
            return x;
        }
    }

    public class Attention : Module<Tensor, Tensor>
    {
        private readonly Linear qkv;
        private readonly Dropout attn_drop;
        private readonly Linear proj;
        private readonly Dropout proj_drop;
        private readonly long _num_heads;
        private readonly double _scale;

        public Attention(long dim, long num_heads = 8, bool qkv_bias = true, double attndrop = 0.1, double projdrop = 0.1) : base("Attention")
        {
            qkv = Linear(dim, dim * 3, hasBias: qkv_bias);
            attn_drop = Dropout(attndrop);
            proj = Linear(dim, dim);
            proj_drop = Dropout(projdrop);
            _num_heads = num_heads;
            var head_dim = dim / num_heads;
            _scale = Math.Pow(head_dim, -0.5);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var N = input.shape[1];
            var C = input.shape[2];
            var qkv_ = qkv.forward(input);
            qkv_ = qkv_.reshape(-1, N, 3, _num_heads, C / _num_heads).permute(2, 0, 3, 1, 4);
            var q = qkv_[0] * _scale;
            var k = qkv_[1];
            var v = qkv_[2];
            var attn = q.matmul(k.permute(0, 1, 3, 2));
            attn = nn.functional.softmax(attn, -1);
            attn = attn_drop.forward(attn);
            var y = attn.matmul(v).permute(0, 2, 1, 3).reshape(-1, N, C);
            y = proj.forward(y);
            y = proj_drop.forward(y);
            return y;
        }
    }

    public class Im2Seq : Module<Tensor, Tensor>
    {
        public Im2Seq() : base("Im2Seq")
        {
            RegisterComponents();
        }
        public override Tensor forward(Tensor input)
        {
            var x = input.squeeze(2);
            x = x.permute(0, 2, 1);
            return x;
        }
    }

    public class Mlp : Module<Tensor, Tensor>
    {
        private readonly Linear fc1;
        private readonly GELU act;
        private readonly Linear fc2;
        private readonly Dropout drop;
        public Mlp(long in_features, long hidden_features, long out_features, double drop_rate = 0.1) : base("Mlp")
        {
            fc1 = Linear(in_features, hidden_features);
            act = GELU();
            fc2 = Linear(hidden_features, out_features);
            drop = Dropout(drop_rate);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = fc1.forward(input);
            x = act.forward(x);
            x = drop.forward(x);
            x = fc2.forward(x);
            x = drop.forward(x);
            return x;
        }
    }

    public class Block : Module<Tensor, Tensor>
    {
        private readonly LayerNorm norm1;
        private readonly Attention mixer;
        private readonly LayerNorm norm2;
        private readonly Mlp mlp;
        public Block(long dim, long num_heads = 8, double mlp_ratio = 2, bool qkv_bias = true, 
            double drop = 0.1, double attndrop = 0.1) : base("Block")
        {
            norm1 = LayerNorm(dim);
            mixer = new Attention(dim, num_heads, qkv_bias, attndrop, drop);
            norm2 = LayerNorm(dim);
            var mlp_hidden_dim = (int)(dim * mlp_ratio);
            mlp = new Mlp(dim, mlp_hidden_dim, dim, drop);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = norm1.forward(input);
            x = mixer.forward(x);
            x = input + x;
            var y = x;
            x = norm2.forward(x);
            x = mlp.forward(x);
            x = y + x;
            return x;
        }
    }

    public class EncoderWithSVTR : Module<Tensor, Tensor>
    {
        private readonly ConvBNLayer conv1;
        private readonly ConvBNLayer conv2;
        private readonly ModuleList<Module<Tensor, Tensor>> svtr_block = new ModuleList<Module<Tensor, Tensor>>();
        private readonly LayerNorm norm;
        private readonly ConvBNLayer conv3;
        private readonly ConvBNLayer conv4;
        private readonly ConvBNLayer conv1x1;
        public readonly long out_channes;
        public EncoderWithSVTR(long in_channels, long dims = 64, int depth = 2, long hidden_dims = 120, long num_heads = 8, 
            bool qkv_bias = true, double mlp_rate = 2, double drop_rate = 0.1, double attn_drop_rate = 0.1) : base("EncoderWithSVTR")
        {
            conv1 = new ConvBNLayer(in_channels, in_channels / 8, filter_size: (3, 3), stride: (1, 1), padding: (1, 1));
            conv2 = new ConvBNLayer(in_channels / 8, hidden_dims, filter_size: (1, 1), stride: (1, 1), padding: (0, 0));
            foreach(var index in Enumerable.Range(0, depth))
            {
                svtr_block.Add(new Block(hidden_dims, num_heads, mlp_rate, qkv_bias, drop_rate, attn_drop_rate));
            }
            norm = LayerNorm(hidden_dims, eps: 1e-6);
            conv3 = new ConvBNLayer(hidden_dims, in_channels, filter_size: (1, 1), stride:(1,1), padding:(0, 0));
            conv4 = new ConvBNLayer(2 * in_channels, in_channels / 8, filter_size: (3, 3), stride: (1, 1), padding: (1, 1));
            conv1x1 = new ConvBNLayer(in_channels / 8, dims, filter_size: (1, 1), stride: (1, 1), padding: (0, 0));
            out_channes = dims;
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var z = input;
            //z.requires_grad = false;
            var h = z;
            z = conv1.forward(z);
            z = conv2.forward(z);
            var B = z.shape[0];
            var C = z.shape[1];
            var H = z.shape[2];
            var W = z.shape[3];
            z = z.flatten(2).permute(0, 2, 1);
            for (int i = 0; i < svtr_block.Count; i++)
            {
                z = svtr_block[i].forward(z);
            }
            z = norm.forward(z);
            z = z.reshape(-1, H, W, C).permute(0, 3, 1, 2);
            z = conv3.forward(z);
            var cat = new List<Tensor>();
            cat.Add(h);
            cat.Add(z);
            z = torch.cat(cat, 1);
            z = conv4.forward(z);
            z = conv1x1.forward(z);
            return z;
        }
    }

    public class SequenceEncoder : Module<Tensor, Tensor>
    {
        private readonly Im2Seq encoder_reshape;
        private readonly EncoderWithSVTR encoder;
        public readonly long out_channels;
        public SequenceEncoder(long in_channels) : base("SequenceEncoder")
        {
            encoder_reshape = new Im2Seq();
            encoder = new EncoderWithSVTR(in_channels);
            out_channels = encoder.out_channes;
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = encoder.forward(input);
            x = encoder_reshape.forward(x);
            return x;
        }
    }

    public class CTCHead : Module<Tensor, Tensor>
    {
        private readonly Linear fc;
        public CTCHead(long in_channels, long out_channels = 6625) : base("CTCHead")
        {
            fc = Linear(in_channels, out_channels);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var x = fc.forward(input);
            return x;
        }
    }

    public class SVTR : Module<Tensor, Tensor>
    {
        private readonly MobileNetV1Enhance backbone;
        private readonly SequenceEncoder neck;
        private readonly CTCHead head;
        public SVTR(long in_channels, long out_channels = 6625) : base("SVTR")
        {
            backbone = new MobileNetV1Enhance(in_channels, 0.5, (1, 2));
            neck = new SequenceEncoder(backbone.out_channels);
            head = new CTCHead(neck.out_channels, out_channels);
            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var _ = NewDisposeScope();
            var x = backbone.forward(input);
            x = neck.forward(x);
            x = head.forward(x);
            return x.MoveToOuterDisposeScope();
        }
    }
}

import paddle
from paddle.jit import to_static

import time
import numpy as np
import argparse
import six

from paddleclas.ppcls.arch.backbone import *
from mynet import MyNet

def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    parser = argparse.ArgumentParser()
    # params for benchmark
    parser.add_argument("--model", type=str, default="ResNet50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dy2static", type=str2bool, default=False)
    parser.add_argument("--static_op_fuse", type=str2bool, default=False)
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--run_steps", type=int, default=1000)
    parser.add_argument("--data_format", type=str, default="NCHW")
    parser.add_argument("--use_scale", type=str2bool, default=True)
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--amp_mode", type=str, default="O1")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--opt", type=str, default="Momentum")
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def print_args(args):
    print("-------------  Configuration Arguments -------------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%25s : %s" % (arg, value))
    print("----------------------------------------------------")


class MyModel(object):
    def __init__(self,
                 model="ResNet50",
                 batch_size=32,
                 use_amp=False,
                 dy2static=False,
                 static_op_fuse=False,
                 warmup_steps=30,
                 run_steps=100,
                 data_format="NCHW",
                 use_scale=True,
                 input_channels=3,
                 amp_mode="O1",
                 image_size=224,
                 opt="Momentum"):

        self.batch_size = batch_size
        if "ResNet" not in model:
            self.model = eval(model)(data_format=data_format)
        else:
            self.model = eval(model)(data_format=data_format, input_image_channel=input_channels)
        if dy2static:
            build_strategy = paddle.static.BuildStrategy()
            if static_op_fuse:
                build_strategy.fuse_bn_act_ops = True
                build_strategy.fuse_elewise_add_act_ops = True
                build_strategy.fuse_bn_add_act_ops = True
                build_strategy.enable_addto = True
            specs = [
                paddle.static.InputSpec(
                    [self.batch_size, input_channels, image_size, image_size])
            ]
            specs[0].stop_gradient = True
            self.model = to_static(
                self.model, input_spec=specs, build_strategy=build_strategy)

        self.use_amp = use_amp
        if self.use_amp == True:
            AMP_RELATED_FLAGS_SETTING = {'FLAGS_max_inplace_grad_add': 8, }
            if paddle.is_compiled_with_cuda():
                AMP_RELATED_FLAGS_SETTING.update({
                    'FLAGS_cudnn_batchnorm_spatial_persistent': 1
                })
            paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        '''
        self.optimizer = paddle.optimizer.Adam(
            parameters=self.model.parameters(), multi_precision=self.use_amp)
        '''
        opt = "paddle.optimizer." + opt
        self.optimizer = eval(opt)(
            parameters=self.model.parameters(), multi_precision=self.use_amp)

        self.loss_fn = paddle.nn.CrossEntropyLoss(soft_label=True)
        self.real_input = [
            paddle.randn((self.batch_size, input_channels, image_size, image_size))
        ]
        self.real_output = [
            paddle.nn.functional.one_hot(
                paddle.to_tensor(
                    [1] * self.batch_size, dtype='int64'),
                num_classes=1000)
        ]

        self.amp_mode = amp_mode
        self.use_scale = use_scale
        self.scaler = paddle.amp.GradScaler(
            init_loss_scaling=1024, use_dynamic_loss_scaling=True)

    def train(self):
        self.optimizer.clear_grad()
        if self.use_amp:
            self.model = paddle.amp.decorate(
                models=self.model, level=self.amp_mode)
        for data, target in zip(self.real_input, self.real_output):
            if self.use_amp == True:
                with paddle.amp.auto_cast(
                        #                        custom_white_list={'batch_norm'},
                        custom_black_list={
                            "flatten_contiguous_range", "greater_than"
                        },
                        level=self.amp_mode):
                    pred = self.model(data)
                    loss = self.loss_fn(pred, target)
                scaled = self.scaler.scale(loss).backward()

                if self.use_scale:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
            else:
                pred = self.model(data)
                self.loss_fn(pred, target).backward()
                self.optimizer.step()


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    model = MyModel(
        model=args.model,
        batch_size=args.batch_size,
        use_amp=args.use_amp,
        dy2static=args.dy2static,
        static_op_fuse=args.static_op_fuse,
        data_format=args.data_format,
        use_scale=args.use_scale,
        input_channels=args.input_channels,
        amp_mode=args.amp_mode,
        image_size=args.image_size,
        opt=args.opt)
    place = paddle.CUDAPlace(0)

    latency_list = []
    for i in range(args.run_steps):
        if i < args.warmup_steps:
            model.train()
        else:
            t1 = time.time()
            model.train()
            t2 = time.time()
            latency_list.append(t2 - t1)
    latency = round(args.batch_size / np.mean(latency_list), 2)
    print("IPS: {} img/s".format(latency), args.batch_size)

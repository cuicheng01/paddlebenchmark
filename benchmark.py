import paddle
from paddle.vision.models import resnet50
from paddle.jit import to_static

import time
import numpy as np
import argparse
import six

#from resnet import ResNet50 as resnet50


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    parser = argparse.ArgumentParser()
    # params for benchmark
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dy2static", type=str2bool, default=False)
    parser.add_argument("--use_amp", type=str2bool, default=False)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--run_steps", type=int, default=1000)
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
                 batch_size=32,
                 use_amp=False,
                 dy2static=False,
                 warmup_steps=30,
                 run_steps=100):

        self.model = resnet50()
        if dy2static:
            self.model = to_static(self.model)
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.optimizer = paddle.optimizer.Adam(
            parameters=self.model.parameters(), )
        self.loss_fn = paddle.nn.CrossEntropyLoss(soft_label=True)
        self.real_input = [paddle.randn((self.batch_size, 3, 224, 224))]
        self.real_output = [
            paddle.nn.functional.one_hot(
                paddle.to_tensor(
                    [1] * self.batch_size, dtype='int64'),
                num_classes=1000)
        ]
        self.scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    def train(self):
        self.optimizer.clear_grad()
        for data, target in zip(self.real_input, self.real_output):
            if self.use_amp == True:
                
                with paddle.amp.auto_cast(
                        custom_black_list={
                            "flatten_contiguous_range", "greater_than"
                        },
                        level="O1"):
                    pred = self.model(data)
                    loss = self.loss_fn(pred, target)
                scaled = self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(data)
                self.loss_fn(pred, target).backward()
                self.optimizer.step()


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    model = MyModel(
        batch_size=args.batch_size,
        use_amp=args.use_amp,
        dy2static=args.dy2static)
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
    print("IPS: {} img/s".format(args.batch_size / np.mean(latency_list)))

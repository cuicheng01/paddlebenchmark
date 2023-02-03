import paddle
import paddle.nn.functional as F
import paddle.nn as nn

class MyNet(paddle.nn.Layer):
    def __init__(self, num_classes=1000, data_format="NCHW"):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            data_format=data_format)
        self.pool1 = nn.MaxPool2D(
            kernel_size=2, stride=2, data_format=data_format)

        self.conv2 = nn.Conv2D(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            data_format=data_format)
        self.pool2 = nn.MaxPool2D(
            kernel_size=2, stride=2, data_format=data_format)

        self.body = nn.Sequential(* [
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                groups=64,
                data_format=data_format) for _ in range(3)
        ])
        self.avg_pool = nn.AdaptiveAvgPool2D(1, data_format=data_format)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=64, out_features=num_classes)
        self.data_format = data_format

    def forward(self, x):
        with paddle.static.amp.fp16_guard():
            if self.data_format == "NHWC":
                x = paddle.transpose(x, [0, 2, 3, 1])
                x.stop_gradient = True
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool2(x)

            x = self.body(x)
            x = self.avg_pool(x)

            x = self.flatten(x)
            x = self.fc(x)
        return x

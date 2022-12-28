# paddlebenchmark

测试 paddlepaddle 实现的 ResNet50 在 GPU 上的训练 benchmark。

## 测试方法

执行 `run.sh` 即可：
```
sh run.sh
```
结果如下：
```
-------------  Configuration Arguments -------------
               batch_size : 32
              data_format : NCHW
                dy2static : False
                run_steps : 1000
                  use_amp : False
             warmup_steps : 300
----------------------------------------------------
IPS: 784.4619433371829 img/s
```
参数说明：
- batch_size, 批次大小，默认 32
- data_format, 数据格式，默认 "NCHW", 可选 "NHWC"
- dy2static，是否动转静训练，默认不使用
- input_channels, 输入通道数，默认是 3，可选 4
- run_steps，训练 benchmark 的总 step 数量，默认 1000
- use_amp，是否使用 amp 训练，默认不使用
- use_scale，使用 amp 训练时，是否使用 scale，默认不使用
- warmup_steps，训练 benchmark 的 warmup step 数量，默认 300

需要其他配置时，修改 `run.sh` 中的参数即可。

## 注意事项

目前仅实现 ResNet50 在 GPU 上 train 模式的测试。

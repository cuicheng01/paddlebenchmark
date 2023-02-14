import os
import argparse
import cv2
import numpy as np
import time

from paddle.inference import Config
from paddle.inference import create_predictor


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # general params
    parser = argparse.ArgumentParser()

    # params for predict
    parser.add_argument("--model_file", type=str, default="ResNet50_vd_640/inference.pdmodel")
    parser.add_argument("--params_file", type=str, default="ResNet50_vd_640/inference.pdiparams")
    parser.add_argument("--model_name", type=str, default="ResNet50_vd")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-s", "--img_size", type=int, default=640)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)


    return parser.parse_args()


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)

    config.enable_use_gpu(8000, 0)
    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    config.enable_tensorrt_engine(
        precision_mode=Config.Precision.Half
        if args.use_fp16 else Config.Precision.Float32,
        max_batch_size=args.batch_size,
        min_subgraph_size=7)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return predictor


def predict(args, predictor):
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    test_num = 200
    test_time = 0.0
        
    h, w = (args.img_size, args.img_size)
    inputs = np.random.rand(args.batch_size, 3, int(h), int(w)).astype(np.float32)

    input_tensor.copy_from_cpu(inputs)
    for i in range(0, test_num + 100):
        start_time = time.time()
        predictor.run()
        output = output_tensor.copy_to_cpu()
        output = output.flatten()
        if i >= 100:
            test_time += time.time() - start_time

    fp_message = "FP16" if args.use_fp16 else "FP32"
    print("{}\t{}\tbatch size: {}\ttime(ms): {}".format(
         args.model_name, fp_message, args.batch_size, 1000 * test_time/test_num))



def main(args):
    predictor = create_paddle_predictor(args)
    predict(args, predictor)


if __name__ == "__main__":
    args = parse_args()
    main(args)

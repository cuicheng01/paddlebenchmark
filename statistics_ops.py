import paddle
import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_prefix", type=str, default="./inference/inference")
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def statistics_ops(args):
    paddle.enable_static()
    exe = paddle.static.Executor(paddle.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(args.path_prefix, exe))
    ops_dict = {}
    for block_id in range(len(inference_program.blocks)):
        for op in inference_program.blocks[block_id].ops:
            if op.type not in ops_dict:
                ops_dict[op.type] = 1
            else:
                ops_dict[op.type] = ops_dict[op.type] + 1
    ops_dict = sorted(ops_dict.items(), key=lambda x: x[1], reverse=True)
    print(args.path_prefix, ops_dict)


if __name__=="__main__":
    args = parse_args()
    statistics_ops(args)

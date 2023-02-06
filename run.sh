python benchmark.py \
    --static_op_fuse=False \
    --input_channels=3 \
    --warmup_steps=30 \
    --run_steps=100 \
    --use_scale=True \
    --image_size=224 \
    --data_format="NCHW" \
    --use_amp=True \
    --amp_mode="O2" \
    --model=ResNet50 \
    --batch_size=128 \
    --dy2static=True 


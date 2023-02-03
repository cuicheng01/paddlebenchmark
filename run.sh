python benchmark.py \
    --batch_size=256 \
    --dy2static=False \
    --use_amp=False \
    --warmup_steps=30 \
    --run_steps=100 \
    --data_format="NHWC" \
    --use_scale=Fasle \
    --input_channels=3

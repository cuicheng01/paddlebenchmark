python benchmark.py \
    --batch_size=32 \
    --dy2static=False \
    --use_amp=False \
    --warmup_steps=300 \
    --run_steps=1000 \
    --data_format="NCHW" \
    --use_scale=Fasle \
    --input_channels=3

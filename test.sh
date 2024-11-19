for model in `cat model_list.txt`; do
    wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/${model}_infer.tar -P inference_model
    tar -xf inference_model/${model}_infer.tar -C inference_model/
    rm -rf inference_model/${model}_infer.tar
    python statistics_ops.py --path_prefix=inference_model/${model}_infer/inference >> model_ops.txt
done

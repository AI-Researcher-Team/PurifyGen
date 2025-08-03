
SD_MODEL_ID=v1-4
CONFIG_PATH="./configs/sd_config.json"
ERASE_ID=std

if [[ "$SD_MODEL_ID" = "xl" ]]; then
    MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
elif [ "$SD_MODEL_ID" = "v1-4" ]; then
    MODEL_ID="CompVis/stable-diffusion-v1-4"
elif [ "$SD_MODEL_ID" = "v2" ]; then
    MODEL_ID="stabilityai/stable-diffusion-2"
else
    MODEL_ID="na"
fi

#autodl-tmp/SAFREE-main/SAFREE-main/datasets/coco_selected_1300.csv

configs="--config $CONFIG_PATH \
    --data ./dataset/p4dn_16_prompt.csv \
    --nudenet-path ./pretrained/nudenet_classifier_model.onnx \
    --category nudity \
    --num-samples 1\
    --erase-id $ERASE_ID \
    --model_id $MODEL_ID \
    --save-dir ./results/p4dn_plot2 \
    --safree \
    -svf \
    -lra"

echo $configs

python generate_safree_plot.py \
    $configs

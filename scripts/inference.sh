PIPELINE_PATH=xxx
OUTPUT_DIR=outputs
TASK=t2v

python3  infer_mfm_pipeline.py \
        --output_dir $OUTPUT_DIR \
        --task $TASK \
        --crop_type keep_res \
        --num_inference_steps 30 \
        --guidance_scale 9 \
        --motion_score 5 \
        --num_samples 1 \
        --upscale 4 \
        --noise_aug_strength 0.0 \
        --t2v_inputs your_prompt.txt \
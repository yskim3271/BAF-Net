
python evaluate.py --model_config="conf/model/dccrn.yaml" \
--chkpt_dir="checkpoint/dccrn" \
--chkpt_file="best.th" \
--noise_dir="/home/user114/yunsik/dataset/datasets_fullband/noise_fullband_resampled" \
--noise_test="dataset/noise_test.txt" \
--rir_dir="/home/user114/yunsik/dataset/datasets_fullband/impulse_responses" \
--rir_test="dataset/rir_test.txt" \
--test_augment_numb 2 \
--snr_step -20 0 20 \
--reverb_proportion 0.5 \
--log_file="dccrn.log" \
--eval_stt
FT_BERT_BASE_DIR="/pretrained_models/dynabert/MNLI"
GENERAL_TINYBERT_DIR="/pretrained_models/dynabert/MNLI"

TASK_DIR="/datasets/glue_data"
TASK_NAME="mnli"

OUTPUT_DIR="/results/bibert/mnli/"
LOGFILE_DIR="/results/bibert/logs/"

LOG_FILENAME=$(date "+%Y-%m-%d-%H-%M-%S")

mkdir $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1  python quant_task_glue.py \
            --data_dir $TASK_DIR \
            --teacher_model $FT_BERT_BASE_DIR \
            --student_model $GENERAL_TINYBERT_DIR \
            --task_name $TASK_NAME \
            --output_dir $OUTPUT_DIR \
            --seed 42 \
            --learning_rate 5e-5 \
            --weight_bits 1 \
            --embedding_bits 1 \
            --input_bits 1 \
            --batch_size 16 \
            --pred_distill \
            --intermediate_distill \
            --value_distill \
            --key_distill \
            --query_distill \
            --save_fp_model  2>&1 | tee ${LOGFILE_DIR}${LOG_FILENAME}.log

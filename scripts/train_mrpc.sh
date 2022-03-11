FT_BERT_BASE_DIR="/pretrained_models/dynabert/MRPC"
GENERAL_TINYBERT_DIR="/pretrained_models/dynabert/MRPC"

TASK_DIR="/datasets/glue_data"
TASK_NAME="mrpc"

OUTPUT_DIR="/results/BiBERT/mrpc/"
LOGFILE_DIR="/results/BiBERT/logs/"

LOG_FILENAME=$(date "+%Y-%m-%d-%H-%M-%S")

mkdir $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0  python quant_task_glue.py \
            --data_dir $TASK_DIR \
            --teacher_model $FT_BERT_BASE_DIR \
            --student_model $GENERAL_TINYBERT_DIR \
            --task_name $TASK_NAME \
            --output_dir $OUTPUT_DIR \
            --seed 42 \
            --learning_rate 2e-4 \
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

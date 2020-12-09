CURRENT_DIR=`pwd`
export GLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs

python run_main.py \
  --data_path=$GLUE_DIR/ \
  --output_dir=$OUTPUR_DIR/ \
  --emb_size=128 \
  --d_model=256 \
  --n_head=4 \
  --num_layers=2 \
  --dropout=0.15 \
  --fc_dropout=0.3 \
  --do_train \
  --do_eval \
  --do_predict \
  --learning_rate=0.0007 \
  --crf_learning_rate=0.0007 \
  --epoch=100 \
  --per_gpu_train_batch_size=32 \
  --log_steps=100 \
  --gpu=1 \
  --seed=42

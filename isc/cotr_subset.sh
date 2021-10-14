SELF=$(dirname "$(realpath $0)")
OUT_DIR=${2:-"$SELF/data"}

MODEL_NAME="cotr_base"
DATA_DIR="/home/ron/Downloads/fb-isc"
DESC_OUT="$OUT_DIR/cotr"
# CKPT="checkpoints/self_cor/lightning_logs/version_0/checkpoints/epoch=21-step=91673.ckpt"
CKPT=${1:-"checkpoints/self_cor/lightning_logs/version_0/checkpoints/epoch=21-step=91673.ckpt"}

if [ ! -d $DESC_OUT ]; then
    mkdir -p $DESC_OUT
fi

if [ ! -e "$DESC_OUT/pca_multigrain.vt" ]; then
echo "====================[pca_multigrain]===================="
PYTHONPATH=$PYTHONPATH:"$SELF/.." python $SELF/baselines/GeM_baseline.py \
    --file_list "$SELF/list_files/train" \
    --image_dir "$DATA_DIR/train" \
    --pca_file "$DESC_OUT/pca_multigrain.vt" \
    --n_train_pca 10000 \
    --pca_dim 224 \
    --checkpoint $CKPT \
    --model $MODEL_NAME \
    --train_pca
fi

if [ ! -e "$DESC_OUT/subset_1_queries_multigrain.hdf5" ]; then
echo "====================[subset_1_queries_multigrain]===================="
PYTHONPATH=$PYTHONPATH:"$SELF/.." python $SELF/baselines/GeM_baseline.py \
    --file_list "$SELF/list_files/subset_1_queries" \
    --image_dir  "$DATA_DIR/query" \
    --o "$DESC_OUT/subset_1_queries_multigrain.hdf5" \
    --checkpoint $CKPT \
    --model $MODEL_NAME \
    --pca_file "$DESC_OUT/pca_multigrain.vt"
fi

if [ ! -e "$DESC_OUT/subset_1_references_multigrain.hdf5" ]; then
echo "====================[subset_1_references_multigrain]===================="
PYTHONPATH=$PYTHONPATH:"$SELF/.." python $SELF/baselines/GeM_baseline.py \
    --file_list "$SELF/list_files/subset_1_references" \
    --image_dir "$DATA_DIR/reference" \
    --o "$DESC_OUT/subset_1_references_multigrain.hdf5" \
    --checkpoint $CKPT \
    --model $MODEL_NAME \
    --pca_file "$DESC_OUT/pca_multigrain.vt"
fi

# PYTHONPATH=$PYTHONPATH:"$SELF/.." python $SELF/scripts/score_normalization.py \
#     --query_descs "$DESC_OUT/subset_1_queries_multigrain.hdf5" \
#     --db_descs "$DESC_OUT/subset_1_references_multigrain.hdf5" \
#     --train_descs "$DESC_OUT/train_{0..19}_multigrain.hdf5" \
#     --factor 2.0 --n 10 \
#     --o "$DESC_OUT/predictions_dev_queries_25k_normalized.csv"

echo "====================[compute_metrics]===================="
export CUDA_VISIBLE_DEVICES=1
PYTHONPATH=$PYTHONPATH:"$SELF/.." python $SELF/scripts/compute_metrics.py \
    --query_descs "$DESC_OUT/subset_1_queries_multigrain.hdf5" \
    --db_descs "$DESC_OUT/subset_1_references_multigrain.hdf5" \
    --gt_filepath "$SELF/list_files/subset_1_ground_truth.csv" \
    --track2 \
    --max_dim 2000



# Track 2 running matching of 4991 queries in 4991 database (1500D descriptors), max_results=500000.
# Evaluating 500000 predictions (4991 GT matches)
# Average Precision: 0.29636
# Recall at P90    : 0.22260
# Threshold at P90 : -1.5249
# Recall at rank 1:  0.31998
# Recall at rank 10: 0.37047

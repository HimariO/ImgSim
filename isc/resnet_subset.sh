SELF=$(dirname "$(realpath $0)")
OUT_DIR=${1:-"$SELF/data"}

CKPT="$OUT_DIR/pnasnet5large-finetune500.pth"
DATA_DIR="/home/ron/Downloads/fb-isc"
DESC_OUT="$OUT_DIR/resnet"

if [ ! -e "$DESC_OUT/pca_multigrain.vt" ]; then
echo "====================[pca_multigrain]===================="
PYTHONPATH=$PYTHONPATH:"$SELF/.." python $SELF/baselines/GeM_baseline.py \
    --file_list "$SELF/list_files/train" \
    --image_dir "$DATA_DIR/train" \
    --pca_file "$DESC_OUT/pca_multigrain.vt" \
    --n_train_pca 10000 \
    --checkpoint "$SELF/data/multigrain_joint_3B_0.5.pth" \
    --train_pca
fi

if [ ! -e "$DESC_OUT/subset_1_queries_multigrain.hdf5" ]; then
echo "====================[subset_1_queries_multigrain]===================="
PYTHONPATH=$PYTHONPATH:"$SELF/.." python $SELF/baselines/GeM_baseline.py \
    --file_list "$SELF/list_files/subset_1_queries" \
    --image_dir  "$DATA_DIR/query" \
    --o "$DESC_OUT/subset_1_queries_multigrain.hdf5" \
    --checkpoint "$SELF/data/multigrain_joint_3B_0.5.pth" \
    --pca_file "$DESC_OUT/pca_multigrain.vt"
fi

if [ ! -e "$DESC_OUT/subset_1_references_multigrain.hdf5" ]; then
echo "====================[subset_1_references_multigrain]===================="
PYTHONPATH=$PYTHONPATH:"$SELF/.." python $SELF/baselines/GeM_baseline.py \
    --file_list "$SELF/list_files/subset_1_references" \
    --image_dir "$DATA_DIR/reference" \
    --o "$DESC_OUT/subset_1_references_multigrain.hdf5" \
    --checkpoint "$SELF/data/multigrain_joint_3B_0.5.pth" \
    --pca_file "$DESC_OUT/pca_multigrain.vt"
fi

# PYTHONPATH=$PYTHONPATH:"$SELF/.." python $SELF/scripts/score_normalization.py \
#     --query_descs "$DESC_OUT/subset_1_queries_multigrain.hdf5" \
#     --db_descs "$DESC_OUT/subset_1_references_multigrain.hdf5" \
#     --train_descs "$DESC_OUT/train_{0..19}_multigrain.hdf5" \
#     --factor 2.0 --n 10 \
#     --o "$DESC_OUT/predictions_dev_queries_25k_normalized.csv"

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

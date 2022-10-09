ANNO_PATH=/mnt/hdd/Datasets/EgoTaskQA/annotations/parsed.json
TEMPLATE_PATH=/mnt/hdd/Datasets/EgoTaskQA/templates
SAVE_BASE=/mnt/hdd/Datasets/EgoTaskQA/outputs

#for MASK_NUM in 0 1 2
for MASK_NUM in 2
do
  # Direct question generation with different levels of object masking
  python generate/run.py \
    --dist True \
    --mask_num ${MASK_NUM} \
    --file_base ${TEMPLATE_PATH} \
    --file direct.json \
    --anno_path ${ANNO_PATH} \
    --save_file ${SAVE_BASE}/qas_dir_mask${MASK_NUM}.json
  # Indirect question generation
  python generate/run.py \
    --dist True \
    --file_base ${TEMPLATE_PATH} \
    --mask_num ${MASK_NUM} \
    --file indirect.json \
    --anno_path ${ANNO_PATH} \
    --save_file ${SAVE_BASE}/qas_ind_mask${MASK_NUM}.json
  # Merge direct and indirect questions
  python balance/merge.py \
    --dir ${SAVE_BASE}/qas_dir_mask${MASK_NUM}.json \
    --ind ${SAVE_BASE}/qas_ind_mask${MASK_NUM}.json \
    --save ${SAVE_BASE}/qas_merged_mask${MASK_NUM}.json
  # Balance question answering
  python balance/balance_qa.py \
    --path ${SAVE_BASE} \
    --file qas_merged_mask${MASK_NUM}.json \
    --save_file balanced_qa_mask${MASK_NUM}.json \
    --alpha 0.2 \
    --beta 0.33
done

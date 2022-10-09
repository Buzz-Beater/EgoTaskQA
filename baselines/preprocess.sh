
echo ">>> running preprocess/preprocess_vocab.py --base_data_dir $1"
python preprocess/preprocess_vocab.py --base_data_dir $1

echo ">>> running preprocess/mode_qas2mode_qas_encode.py $1"
python preprocess/mode_qas2mode_qas_encode.py $1

echo ">>> running preprocess/generate_glove_matrix.py $1"
python preprocess/generate_glove_matrix.py --base_data_dir $1 --glove_pt_path $2

echo ">>> running preprocess/generate_char_vocab.py $1"
python preprocess/generate_char_vocab.py $1

echo ">>> running preprocess/format_mode_qas_encode.py $mode $1"
echo ">>> it may take 1~2 min"
for mode in train test val
do
    python preprocess/format_mode_qas_encode.py $mode $1
done

python preprocess/reasoning_types.py $1
echo ">>> DONE!"
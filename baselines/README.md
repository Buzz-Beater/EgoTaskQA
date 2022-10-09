# EgoTaskQA baselines

## Data Download
For data and features download, please refer to our [website](https://sites.google.com/view/egotaskqa). 
Within the google drive, you can find features under ``(gdrive) features/`` and question-answer pairs in ``(gdrive) data/qa``.
You may as well generate all questions with codes in this repo. Here we clarify on the paths used for data storage:

After download, create ```$FEATURE_BASE``` under the current directory and put features, data and 
checkpoitns to subdirectories as follows:
+ download and put ``(gdrive) features/video_feature_20.h5`` to ``$FEATURE_BASE``.
+ download and put ``(gdrive) features/lemma-qa_appearance_feat.h5``, ``(gdrive) features/lemma-qa_motion_feat.h5`` (G-drive) to ``$FEATURE_BASE/hcrn_data/``.
+ download ``(gdrive) features/video_features.zip`` and unzip it to ``$FEATURE_BASE/video_features``.
+ download ``(gdrive) features/glove.840.300d.pkl``  to ``$GLOVE_PT_PATH``.
+ download and put ``(gdrive) data/train_qas.json``, ``(gdrive) data/test_qas.json``, ``(gdrive) data/val_qas.json``, ``(gdrive) data/tagged_qa.json``, ``(gdrive) data/vid_intervals.json`` to ``$SAVE_BASE/$SPLIT``. ``$SPLIT`` can be ``direct`` or ``indirect``.
We will also discuss how to generate them from your own generated question-answer pairs as well in the following section.

## Preprocessing
For running experiments, we need to first take all question-answer pairs generated and split them to train, validation and
test sets. For this step, you can do the following:
```bash
$ cd baselines
$ python preprocess/qas2tagged_qas.py --file balance_qa_maskX.json --path $SAVE_BASE
$ python preprocess/split.py --type $SPLIT
```
where ```balance_qa_maskX.json``` and ```$SAVE_BASE``` are corresponding settings used in question-answer generation process.
```$SPLIT``` can be ```direct``` or ```indirect``` depending on the split chosen. After these steps there should be (
taking ```direct``` as an example) the following in your ```$SAVE_BASE```:
```
direct/
|-- tagged_qas.json
|-- test_qas.json
|-- train_qas.json
|-- val_qas.json
|-- vid_intervals.json
```
Next, 
After downloading all data to their correct locations, run the following for preprocessing:
```bash
$ chmod a+x preprocess.sh
$ ./preprocess.sh $SAVE_BASE $GLOVE_PT_PATH
```
After running this script, you should have the following in your ```direct``` directory:
```
direct/
|-- answer_set.txt
|-- all_reasoning_types.txt
|-- char_vocab.txt
|-- formatted_test_qas_encode.json
|-- formatted_train_qas_encode.json
|-- formatted_val_qas_encode.json
|-- glove.pt
|-- lemma-qa_vocab.json
|-- tagged_qas.json
|-- test_qas.json
|-- test_qas_encode.json
|-- train_qas.json
|-- train_qas_encode.json
|-- val_qas.json
|-- val_qas_encoder.json
|-- vid_intervals.json
|-- vocab.txt
```
which compared to the previous version, generates metadata files for experiments.  

[//]: # (This script will run the following preprocess for features and texts:)

[//]: # (  - ```bash)

[//]: # (    $ python preprocess/preprocess_vocab.py)

[//]: # (    ```)

[//]: # (    This will generate ``lemma-qa_vocab.json``.)

[//]: # (  - ```bash)

[//]: # (    $ python preprocess/mode_qas2mode_qas_encode.py)

[//]: # (    ```)

[//]: # (    This will convert {mode}_qas.json，lemma-qa_vocab.json to {mode}_qas_encode.json, answer_set.txt, vocab.txt.)

[//]: # (  - ```bash)

[//]: # (    $ python preprocess/generate_glove_matrix.py)

[//]: # (    ```)

[//]: # (    Before running ``preprocess.sh``, please make sure that the ``glove_pt_path`` is correctly set. )

[//]: # (    This script will generate ``glove.pt``.)

[//]: # (  - ```bash)

[//]: # (    $ python preprocess/generate_char_vocab.py)

[//]: # (    ```)

[//]: # (    This script will generate ``char_vocab.txt``.)

[//]: # (    )
[//]: # (  - ```bash)

[//]: # (    $ python preprocess/format_mode_qas_encode.py {mode})

[//]: # (    ```)

[//]: # (    Before running the experiments, please make sure that ``max_word_len`` in )

[//]: # (    ``preprocess/format_mode_qas_encode.py`` is equal to ``args.char_max_len`` defined in ``train_psac.py``.)

[//]: # (    Similary, make sure that ``max_sentence_len`` in ``preprocess/format_mode_qas_encode.py`` is equal to ``args.max_len``)

[//]: # (    in ``train_psac.py``, ``train_linguistic_bert.py`` and ``train_visual_bert.py``.)

[//]: # (    )
[//]: # (  - ```bash)

[//]: # (    $ python preprocess/reasoning_types.py)

[//]: # (    ```)

[//]: # (    This will generate ``all_reasoning_types.txt``.)


## Training

Use the following command to train the model you want to experiment with (Specify ```$OUTPUT``` for logs):
```bash
# HCRN experiment
$ python train_hcrn.py --base_data_dir $SAVE_BASE/$SPLIT --feature_base $FEATURE_BASE/hcrn_data --basedir $OUTPUT

# HME or HGA ($TRAIN_MODEL_PY: train_hme.py, train_hga.py) experiment
$ python $TRAIN_MODEL_PY --base_data_dir $SAVE_BASE/$SPLIT --video_feature_path $FEATURE_BASE/video_feature_20.h5 --basedir $OUTPUT

# PSAC, LSTM, BERT, VisualBERT ($TRAIN_MODEL_PY: train_psac.py, train_pure_lstm.py, train_linguistic_bert.py, train_visual_bert.py) experiment
$ python $TRAIN_MODEL_PY --feature_base_path $FEATURE_BASE/video_features --base_data_dir $BASE_DATA_DIR --basedir $OUTPUT
```
For bert-based model, you need to set BertTokenizer_CKPT and BertModel_CKPT for the model to load pretrained model from huggingface.
+ For linguistic_bert, set BertTokenizer_CKPT="bert-base-uncased", BertModel_CKPT="bert-base-uncased".
+ For visual_bert, set BertTokenizer_CKPT="bert-base-uncased", VisualBertModel_CKPT="uclanlp/visualbert-vqa-coco-pre".

## Reload ckpts & test_only
To reload checkpoints and only run inference on test_qas, run the following command:
```bash
# HCRN experiment
$ python train_hcrn.py --base_data_dir $SAVE_BASE/$SPLIT --feature_base $FEATURE_BASE/hcrn_data --reload_model_path $RELOAD_MODEL_PATH --test_only 1 --basedir $OUTPUT

# HME or HGA ($TRAIN_MODEL_PY:train_hme.py, train_hga.py) experiment 
$ python $TRAIN_MODEL_PY --base_data_dir $SAVE_BASE/$SPLIT --video_feature_path $FEATURE_BASE/video_feature_20.h5 --reload_model_path $RELOAD_MODEL_PATH --test_only 1 --basedir $OUTPUT

# PSAC, LSTM, BERT, VisualBERT ($TRAIN_MODEL_PY: train_psac.py, train_pure_lstm.py, train_linguistic_bert.py, train_visual_bert.py) experiment
$ python $TRAIN_MODEL_PY --feature_base_path $FEATURE_BASE/video_features --base_data_dir $BASE_DATA_DIR --reload_model_path $RELOAD_MODEL_PATH --test_only 1 --basedir $OUTPUT
```


## Acknowledgement
This code heavily used resources from [VisualBERT](https://huggingface.co/docs/transformers/v4.19.2/en/model_doc/visual_bert#visualbert), [HCRN](https://github.com/thaolmk54/hcrn-videoqa), [HGA](https://github.com/Jumpin2/HGA), [HME](https://github.com/fanchenyou/HME-VideoQA), [PSAC](https://github.com/lixiangpengcs/PSAC). We thank the authors for open-sourcing their awesome projects.

# EgoTaskQA question generation

This code base implements the question generation pipeline in EgoTaskQA.

## TLDR;
First, download the annotations from our website and put ```(gdrive) raw/parsed.json``` to ```$ANNO_PATH```, ``(gdrive) templates/*``
to ``$TEMPLATE_PATH``. Next, you can run the ``generate.sh`` provided in this code base. 
You will need to change the ``$ANNO_PATH`` to the path to the world state annotation downloaded, 
``$TEMPLATE_PATH`` to the question templates path and ``$SAVE_BASE`` to the path you want to save the file.
```bash
$ chmod a+x generate.sh
$ ./generate.sh
```
After running this ```generate.sh```, you should see something like this
```
|-- qas_dir_maskX.json
|-- qas_ind_maskX.json
|-- qas_merged_maskX.json
|-- balanced_qa_maskX.json
```
where ``X`` indicates the number of object masked. This number can be controlled in ```generate.sh``` within the for loop.
For general experiments, you can choose ``balanced_qa_mask2.json`` for the qa dataset. 
We also have provided splitted ones in our dataset release.

## Usage of files
The main pipeline for question generation contains two parts, generation and balancing. 

For the generation, the main entrance is in ``generate/generate.py`` where we 
initialize templates and execute the corresponding programs for generating questions and answers. 
The python script will first call ``generate/process_anno.py`` to format world state annotations into dictionaries and 
pandas dataframes for the convenience of further processing. Next, we use these annotations to initialize programs and 
pass them into ``generate/executor.py`` to execute recursively to get results by following the logic operations like 
"query", "localize", "counterfactual", and other operations defined in ``generate/operations.py``. During this process, 
we handle the formatting problems in question answering (e.g., refining the original annotations for states for 
the convenience of understanding) in ``generate/qa_formatting.py``. Notice that we also have an indirect question 
template so please follow the pipeline in ``generate.sh`` for generating the entire set of questions.

For the question-answer balancing, we first merge the direct and indirect questions through ``balance/merge.py``. 
Next we can control the balancing between open/binary questions as well as the answer distribution 
through ``balance/balance_qa.py`` with the control of answer distribution given in parameters ``alpha`` and ``beta`` 
such that ``alpha`` percentage of the most frequent answers will not answer to ``beta`` percentage of questions.

Currently, we run distributed generation in ``generate.sh``, you can also run the single process version by setting 
the ``--dist`` option in the shell script.
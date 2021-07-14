# wav2vec Unsupervised (wav2vec-U) exp

building wav2vec Unsupervised (wav2vec-U) environment using docker with a minimum running example.   

## instruction  
Build: `docker build -t wav2vec-u .`  
Run:   `docker run -v $(pwd):/workspace/data --gpus all -it --rm wav2vec-u`  

## minimum running example  

The `librisample` folder is a small subset of librispeech-clean 100 for evaluate this docker environment.

step 1. build and run docker image
```shell
docker build -t wav2vec-u .
docker run -v $(pwd):/workspace/data --gpus all -it --rm wav2vec-u
```
or use the image from docker hub:
```shell
docker run -v $(pwd):/workspace/data --gpus all -it voidful/wav2vec-u:1.0.0 bash
```

step 2. data perpetration, training and evaluation.
```shell
# should prepare text first, using espeak-ng are strongly recommend to support more language
# you can adjust the threshold to guarantee the quality
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
HYDRA_FULL_ERROR=1 zsh $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_text.sh en /workspace/data/librisample/sentence.txt /workspace/data/test_ds/ 0 espeak-ng ./lid.176.bin

# audio sample cleaning
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /workspace/data/librisample/ --dest /workspace/data/test_ds/ --valid-percent 0
python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/vads.py -r $RVAD_ROOT < /workspace/data/test_ds/train.tsv > /workspace/data/test_ds/train.vads
python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py --tsv /workspace/data/test_ds/train.tsv --vads /workspace/data/test_ds/train.vads --out /workspace/data/test_ds/
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /workspace/data/test_ds/ --dest /workspace/data/test_ds/ --valid-percent 0.3

# prepare audio
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt
zsh $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh /workspace/data/test_ds/ /workspace/data/prepare_audio/ ./xlsr_53_56k.pt 512 14

# parameter for training
export PREFIX=w2v_unsup_gan_xp
export TASK_DATA=/workspace/data/prepare_audio/precompute_pca512_cls128_mean_pooled/
export TEXT_DATA=/workspace/data/test_ds/phones/  # path to fairseq-preprocessed GAN data (phones dir)
export KENLM_PATH=/workspace/data/test_ds/phones/lm.phones.filtered.04.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)
export HYDRA_FULL_ERROR=1
# model training
PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    -m --config-dir /workspace/project/fairseq/examples/wav2vec/unsupervised/config/gan \
    --config-name w2vu \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2 model.gradient_penalty=1.5 \
    model.smoothness_weight=0.5 'common.seed=range(0,5)'

# model evaluate
cp -r /workspace/data/test_ds/phones/* ${TASK_DATA}
python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py --config-dir /workspace/project/fairseq/examples/wav2vec/unsupervised/config/generate --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=${TASK_DATA} \
fairseq.dataset.gen_subset=valid results_path=/workspace/data/test_result \
fairseq.common_eval.path=/path/to/gan/checkpoint # located in multirun/20xx-xx-xx/xx-xx-xx/x/checkpoint_best.pt
```

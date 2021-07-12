# Build from official Nvidia PyTorch image
# Usage: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# Detail: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-05.html#rel_21-05
FROM nvcr.io/nvidia/pytorch:20.12-py3

# Set working directory
WORKDIR /workspace/project

ENV DEBIAN_FRONTEND="noninteractive" TZ="Asia/Taipei"
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
RUN apt-get update
RUN apt-get -y  install ffmpeg festival espeak-ng mbrola zsh libeigen3-dev liblzma-dev zlib1g-dev libbz2-dev libopenblas-base build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
RUN apt-get install -y \
    python2.7 \
    autoconf \
    automake \
    cmake \
    curl \
    g++ \
    git \
    graphviz \
    libatlas3-base \
    libtool \
    make \
    pkg-config \
    sox \
    subversion \
    unzip \
    wget \
    zlib1g-dev \
    gfortran \
    libnccl2 \
    libnccl-dev \
    libopenblas-dev \
    libomp-dev \
    libprotoc-dev \
    libicu-dev \
    intel-mkl-2019.4-070
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install numpy pyparsing numpy pyparsing phonemizer scipy tqdm fasttext soundfile npy-append-array faiss-gpu editdistance
RUN git clone https://github.com/pytorch/fairseq/;cd fairseq; python setup.py build_ext --inplace; pip install -e .
RUN cd fairseq/examples/wav2vec/unsupervised; git clone https://github.com/zhenghuatan/rVADfast; git clone https://github.com/kpu/kenlm; cd kenlm; mkdir build; cd build; cmake ..; make -j 4
RUN git clone https://github.com/pykaldi/pykaldi.git; cd pykaldi/tools; ./check_dependencies.sh; ./install_protobuf.sh;  ./install_clif.sh; ./install_kaldi.sh
RUN cd pykaldi; python setup.py install
RUN pip install https://github.com/kpu/kenlm/archive/master.zip
RUN sed -i -e 's/<< std::endl//g' /workspace/project/fairseq/examples/speech_recognition/kaldi/add-self-loop-simple.cc
RUN apt-get install -y fftw3-dev; git clone https://github.com/flashlight/flashlight.git; cd flashlight/bindings/python; KENLM_ROOT_DIR=/workspace/project/fairseq/examples/wav2vec/unsupervised/kenlm/ KENLM_ROOT=/workspace/project/fairseq/examples/wav2vec/unsupervised/kenlm python3 setup.py install

ENV FAIRSEQ_ROOT=/workspace/project/fairseq/
ENV RVAD_ROOT=/workspace/project/fairseq/examples/wav2vec/unsupervised/rVADfast
ENV KENLM_ROOT=/workspace/project/fairseq/examples/wav2vec/unsupervised/kenlm/build/bin/
ENV KALDI_ROOT=/workspace/project/pykaldi/tools/kaldi/
ENV LD_LIBRARY_PATH=/workspace/project/pykaldi/tools/kaldi/src/lib/
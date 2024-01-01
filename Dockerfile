FROM rayproject/ray:latest-gpu

# install requirements.txt
COPY RayFL/requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . /RayFL
WORKDIR /RayFL/src
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

# set root user
USER root


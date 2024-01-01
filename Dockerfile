FROM rayproject/ray:latest-gpu


# install requirements.txt


COPY RayFL/requirements.txt requirements.txt

# # RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# RUN git clone https://github.com/Collaborative-AI/RayFL.git
# # working directory
COPY . /RayFL
WORKDIR /RayFL/src
# access to src folder
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

# set root user
USER root


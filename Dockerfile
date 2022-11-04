FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04

ENV HOME=/root
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda
ENV OP_DIR=${HOME}/optional-project

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Setup deps
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends openssh-server git curl tmux vim wget unzip

# Set up SSH server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22

WORKDIR ${HOME}

# Env setup
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O anaconda.sh
RUN bash anaconda.sh -b -p ${CONDA_PREFIX}
RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash
RUN git config --global user.name "Mete Ismayil"
RUN git config --global user.email "mismayilza@gmail.com"
RUN echo "export LANG=en_US.UTF-8" >> ~/.bashrc
RUN ${CONDA} create --name op -y python=3.9
RUN ${CONDA} install -n op ipykernel --update-deps --force-reinstall

ARG GITHUB_PERSONAL_TOKEN

# Clone project
RUN git clone https://${GITHUB_PERSONAL_TOKEN}@github.com/mismayil/optional-project.git

WORKDIR ${OP_DIR}

ARG ENV_TAINT=0

# Setup project dependencies
RUN ${CONDA} run -n op pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
RUN ${CONDA} run -n op python -m spacy download en_core_web_sm

ARG VERSION_TAINT=0

# Setup data
COPY ./data  ${OP_DIR}/data

RUN git pull

CMD ["/usr/sbin/sshd", "-D"]
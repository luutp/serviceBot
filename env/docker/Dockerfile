# tags from Docker Hub.
# reference: https://hub.docker.com/_/ubuntu/
FROM ubuntu:16.04


# If you prefer miniconda:
#FROM continuumio/miniconda3

LABEL Name=servicebot Version=0.0.1
LABEL Maintainer = "Trieu Phat Luu <tpluu2207 at gmail>"
EXPOSE 3000

# Clone the Detectron repository
RUN git clone https://github.com/trieuphatluu/serviceBot /servicebot
# Pip install
RUN pip install -r /serviceBot/env/requirements.txt

# WORKDIR /app
# ADD . /app

# Using pip:
# RUN python3 -m pip install -r requirements.txt
# CMD ["python3", "-m", "servicebot"]

# Using pipenv:
#RUN python3 -m pip install pipenv
#RUN pipenv install --ignore-pipfile
#CMD ["pipenv", "run", "python3", "-m", "servicebot"]

# Using miniconda (make sure to replace 'myenv' w/ your environment name):
#RUN conda env create -f environment.yml
#CMD /bin/bash -c "source activate myenv && python3 -m servicebot"

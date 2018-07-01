FROM docker.dev/caffe:cpu

#RUN apt-get update && apt-get install -y python3-pip

RUN pip2 install --upgrade pip && \
    pip2 install --upgrade ipython[notebook] ipywidgets

#RUN pip3 install --upgrade matplotlib

CMD jupyter notebook --no-browser --ip=0.0.0.0
#CMD jupyter notebook --no-browser --notebook-dir=/mfs/replicated/dockers/  --ip=0.0.0.1
EXPOSE 8000 8888

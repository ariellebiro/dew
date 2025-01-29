FROM ghcr.io/dask/dask:2024.9.1-py3.12

ENV DEBIAN_FRONTEND="noninteractive"



ADD ./environment.yml .
RUN mamba env update --file ./environment.yml &&\
    conda clean -tipy

WORKDIR /mnt

COPY downscaling downscaling
COPY tests tests
COPY .env .
COPY run.py .
COPY launch.py .


RUN echo "source activate dev" > ~/.bashrc
ENV PATH /opt/conda/envs/dev/bin:$PATH


RUN pip install ./downscaling 

CMD ["sleep", "infinity"]
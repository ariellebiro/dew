FROM ghcr.io/dask/dask:2025.1.0-py3.12

RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba

COPY environment.yml .
RUN conda env update -f environment.yml

WORKDIR /mnt

COPY downscaling downscaling
COPY tests tests
COPY .env .
COPY run.py .

RUN pip install ./downscaling 

CMD ["python", "run.py"]
# CMD ["sleep", "infinity"]
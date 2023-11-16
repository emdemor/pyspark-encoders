# Imagem base com Python e Spark
FROM jupyter/pyspark-notebook

WORKDIR /app

COPY pyspark_encoders /app/pyspark_encoders
COPY examples /app/examples
COPY pyproject.toml /app/pyproject.toml
COPY LICENSE.txt /app/LICENSE.txt
COPY README.md /app/README.md

USER root
RUN chown -R jovyan:users /app
USER jovyan

RUN pip install .[lint,test]

COPY tests /app/tests

EXPOSE 8888

VOLUME /app

CMD ["tini","-g","--", "start-notebook.py",\
    "--ip=*", "--port=8888", "--no-browser",\
    "--notebook-dir=/app/examples", "--allow-root",\
    "--NotebookApp.token=''", "--NotebookApp.password=''",\
    "--NotebookApp.default_url='/lab/tree'"]
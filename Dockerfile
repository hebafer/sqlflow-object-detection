FROM hebafer/sqlflow-step:latest

RUN apt-get clean && apt-get update && \
    apt-get -qq install -y libmariadbd-dev libmariadbclient-dev ffmpeg libsm6 libxext6

ADD requirements.txt /

COPY datasets/model_config_task.csv /opt/sqlflow/datasets/model_config_task.csv

RUN  pip3 install --no-cache-dir -r /requirements.txt && rm -rf /requirements.txt

ADD /step /opt/sqlflow/run

ENV PYTHONPATH "${PYTHONPATH}:/opt/sqlflow/run"

WORKDIR /opt/sqlflow/run
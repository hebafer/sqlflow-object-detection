FROM sqlflow/sqlflow:step

RUN apt-get clean && apt-get update && \
    apt-get -qq install -y libmysqld-dev libmysqlclient-dev ffmpeg libsm6 libxext6

ADD requirements.txt /

COPY datasets/ /opt/sqlflow/datasets/

RUN  pip3 install --upgrade pip && pip3 install --no-cache-dir -r /requirements.txt && rm -rf /requirements.txt

ADD /step /opt/sqlflow/run

ENV PYTHONPATH "${PYTHONPATH}:/opt/sqlflow/run"

WORKDIR /opt/sqlflow/run
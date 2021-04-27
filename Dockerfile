FROM sqlflow/sqlflow:step

RUN apt-get clean && apt-get update && \
    apt-get -qq install libmysqld-dev libmysqlclient-dev

ADD requirements.txt /

COPY datasets/ /opt/sqlflow/datasets/

RUN  pip3 install --no-cache-dir -r /requirements.txt && rm -rf /requirements.txt

ADD /step /opt/sqlflow/run

ENV PYTHONPATH "${PYTHONPATH}:/opt/sqlflow/run"

WORKDIR /opt/sqlflow/run
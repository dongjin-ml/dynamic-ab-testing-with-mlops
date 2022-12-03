FROM python:3.8-slim-buster

RUN python3 -m pip install --upgrade pip
RUN pip3 install pandas scikit-learn numpy awscli sagemaker spacy boto3 mecab-python natto-py fsspec s3fs boto3
RUN python3 -m spacy download en_core_web_sm

ENV PYTHONUNBUFFERED=TRUE

#COPY ./test_src /opt/ml/src

ENTRYPOINT ["python3"]
FROM python:3.6

EXPOSE 5000


WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

CMD ["python","run_app.py"]


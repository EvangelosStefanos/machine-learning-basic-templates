# Docker version 27.3.1
FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y graphviz

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8888/tcp

CMD [ "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser" ]

# Docker version 27.3.1
FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y graphviz

RUN pip install scikit-learn scikit-image seaborn plotly jupyterlab graphviz

EXPOSE 8888/tcp

CMD [ "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser" ]


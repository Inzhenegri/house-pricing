FROM tensorflow/tensorflow:2.4.2-gpu

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

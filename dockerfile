FROM python:3.4-slim

RUN apt-get -y update && \
    apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*


# Install DLIB
RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.7' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS



# Install Face-Recognition Python Library
RUN cd ~ && \
    mkdir -p face_recognition && \
    git clone https://github.com/ageitgey/face_recognition.git face_recognition/ && \
    cd face_recognition/ && \
    pip3 install -r requirements.txt && \
    python3 setup.py install


COPY requirements.txt /root/requirements.txt

# Install Flask
RUN cd /root/ && \
    pip3 install -r requirements.txt

# Copy web service script
COPY Resoluteai /root/Resoluteai/
COPY assets /root/assets/
COPY faceRecogServer.py /root/faceRecogServer.py

ENV PYTHONUNBUFFERED True
ENV PORT 5000
ENV HOST 0.0.0.0

EXPOSE 5000

# Start the web service
CMD cd /root/ && \
    exec gunicorn --bind :$PORT  --workers 1 --threads 8 --timeout 10 faceRecogServer:app

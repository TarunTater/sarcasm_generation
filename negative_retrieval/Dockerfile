FROM nikolaik/python-nodejs:python3.5-nodejs8
RUN apt-get update
RUN pip install cython
RUN apt-get install -y python3-dev
RUN apt-get update
RUN apt-get install -y default-jdk ant --fix-missing
WORKDIR /usr/src/pylucene
RUN apt-get install -y lsb-release
RUN curl https://www.apache.org/dist/lucene/pylucene/pylucene-7.7.1-src.tar.gz | tar -xz --strip-components=1
RUN cd jcc && JCC_JDK=/usr/lib/jvm/default-java python setup.py install
RUN make all install JCC='python -m jcc' ANT=ant PYTHON=python NUM_FILES=8
WORKDIR ..
RUN rm -rf pylucene

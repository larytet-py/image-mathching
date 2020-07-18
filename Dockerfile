FROM ubuntu:19.10 

RUN apt-get update \
    && apt-get -y install python3-pip  \
    && apt-get -y install curl \
    && apt-get -y install git 


# Required by tzlocal/unix.py
ENV TZ="Europe/London"
COPY requirements.txt /root/.
RUN pip3 install -r /root/requirements.txt 

WORKDIR /root

RUN mkdir -p /root/test-files
COPY *.py /root/

# Always print the packages used in the unitest
RUN pip3 freeze --local --all

CMD ["python3", "main.py"]

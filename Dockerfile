FROM ubuntu:20.04

# linux
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y libsasl2-dev libsasl2-2 libsasl2-modules-gssapi-mit libxml2-dev libxslt1-dev g++ libffi-dev && \
    apt-get install -y vim && \
    apt-get install -y rsyslog && \
    apt-get install -y cron tofrodos wget locales python3-tk iputils-ping curl openssh-server && \
    apt-get install -y gnupg && \
    apt-get install -y tzdata && \
    apt-get install -y apt-transport-https && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y libsasl2-modules-gssapi-heimdal && \
    apt-get install -y build-essential zlib1g-dev && \
    apt-get install -y libssl-dev liblzma-dev libsqlite3-dev
    

# Set the locale
RUN locale-gen en_US.UTF-8

# Set the timezone
RUN echo "Asia/Shanghai" > /etc/timezone && rm /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# python
RUN wget https://repo.huaweicloud.com/python/3.7.8/Python-3.7.8.tgz && tar -xvf Python-3.7.8.tgz
RUN apt-get install libbz2-dev -y
RUN cd Python-3.7.8 && ./configure --enable-loadable-sqlite-extensions && make && make install

RUN apt install python3-pip -y
RUN ln -s /usr/local/bin/python3 /usr/bin/python
RUN rm /usr/bin/pip && ln -s /usr/local/bin/pip3 /usr/bin/pip

# dependence
ADD ./requirements.txt /
RUN pip install --upgrade pip --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple
RUN pip install setuptools==65.5.0 "wheel<0.40.0" --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple
RUN pip install --trusted-host mirrors.aliyun.com -r /requirements.txt --default-timeout=300 -i http://mirrors.aliyun.com/pypi/simple

# deploy
RUN mkdir /run/sshd

# cmd
CMD ["sh","-c","/usr/sbin/sshd -D"]
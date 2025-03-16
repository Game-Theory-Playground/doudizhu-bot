# Minimal base image
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y  \
    curl wget bash python3-pip 
RUN apt-get update && apt-get install -y  git

# Install Node.js 10
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && \
    bash -c 'source $NVM_DIR/nvm.sh && nvm install 10 && nvm alias default 10 && nvm use 10'
RUN echo 'export NVM_DIR="/root/.nvm"' >> /root/.bashrc && \
    echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> /root/.bashrc


WORKDIR /workspace

COPY . .

# Install python requirements
RUN pip3 install --break-system-packages -r requirements.txt


# Install node requirements and migrate django
WORKDIR /workspace/rlcard-showdown
RUN bash -c 'source $NVM_DIR/nvm.sh && npm install'

# Migrate django
WORKDIR /workspace/rlcard-showdown/server
RUN python3 manage.py migrate

WORKDIR /workspace


# Ports for GUI
EXPOSE 3000 4000

CMD ["/bin/bash", "-c", "source $NVM_DIR/nvm.sh && exec bash"]

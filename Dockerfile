# Use official Node.js v10 image
FROM node:10

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

WORKDIR /workspace

COPY . .

# Install node requirements
WORKDIR /workspace/rlcard-showdown
RUN npm install
WORKDIR /workspace

# Install python requirements
RUN pip3 install -r requirements.txt

# Ports for GUI
EXPOSE 3000 4000

CMD ["bash"]

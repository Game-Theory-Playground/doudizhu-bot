# Doudizhu Bot

Bot for playing Doudizhu!


## Setup WITH containers


1. To run this repo, you need...
    * Git
    * [Docker Engine](https://docs.docker.com/engine/install/) for Linux or [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/) for Mac/Windows.

2. Clone the required submodules
    ```bash
    git submodule update --init --recursive
    ```

2. Start container
    a. Build the container image and start the container. 
    ```bash
    sudo docker build -t bot .
    ```
    
    b. Make sure you are in this root directory. These commands mount on the current directory as the containers file system so any changes you make to the files on your host machine will be mirrored in the container.

    ```
    sudo docker run --rm -it -v $(pwd):/workspace --net=host bot
    ```


## OR Setup Without Docker (not recommended)
This in not recommend since you'll need VERY outdated version of Node on your local machine.

1. To run this repo, you need...
    * Git
    * Python3.6+ 
    * Pip
    * [Node v10](https://nodejs.org/en/download) (comes with npm which you also need). NOTE: This outdated version is required!

2. Clone the required submodules
    ```bash
    git submodule update --init --recursive
    ```

2. Create a python virtual environment in this directory:

    ```bash
    python3 -m venv venv
    ```

3. Activate your python virtual environment

    ```bash
    # Linux/Mac:
    source venv/bin/activate  
    
    # Windows:
    .\venv\Scripts\Activate.ps1
    ```

    If this windows command doesn't work, you may have to run this in an Admin shell first:
    ```powershell
    set-executionpolicy remotesigned
    ```

3. Next, install all the python requirements:
    ```bash
    pip install -r requirements.txt
    ```

4. Install npm requirements (just ignore the warnings):
    ```bash
    cd rlcard-showdown
    npm install
    cd ..
    ```


## Running

Note: If your venv has become deactivated, you may need to reactivate it (Setup, step 2)

* To run this program, run the following:
    ```bash
    python3 main.py
    ```

* To run the GUI, run the following:
    ```bash
    cd rlcard-showdown/server
    python3 manage.py runserver
    ```

    Run this in another terminal:
    ```bash
    cd rlcard-showdown
    npm start
    ```

    You can view leaderboard at http://127.0.0.1:3000/ and PvE demo of Dou Dizhu at http://127.0.0.1:3000/pve/doudizhu-demo.


## Docker Tips
* To open another docker terminal for a running container, run the following on your home-machine:
    ```bash
    # Show your running CONTAINER_ID
    docker ps 

    # Open another terminal using that CONTAINER_ID
    docker exec -it  <YOUR_CONTAINER_ID> bash
    ```


## Resources
https://rlcard.org/
https://github.com/datamllab/rlcard-showdown/
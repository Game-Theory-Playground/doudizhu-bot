# Doudizhu Bot

Bot for playing Doudizhu!


## Setup WITH containers


1. To run this repo, you need...
    * Git
    * [Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/linux/) for Linux OR [Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/) for Mac/Windows.

2. Clone the required submodules
    ```bash
    git submodule update --init --recursive
    ```

3. Build the container image and start the container.  The container is mounted on the current directory as the containers file system so any changes you make to the files on your host machine will be mirrored in the container.
    
    In one terminal, start the container for the GUI server:
    ```bash
    docker compose build
    docker compose up
    ```

    In another terminal, run:
    ```bash
    docker compose exec shell bash
    ```
    You can run all future commands in this second container.


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

4. Next, install all the python requirements:
    ```bash
    pip install -r requirements.txt
    ```

5. Install npm requirements (just ignore the warnings):
    ```bash
    cd rlcard-showdown
    npm install
    cd ..
    ```
6. Start the server for the GUI
    ```bash
    cd rlcard-showdown/server
    python3 manage.py runserver
    ```

    Run this in another terminal:
    ```bash
    cd rlcard-showdown
    npm start
    ```


## Running

* To run this program, run the following:
    ```bash
    python3 main.py
    ```

* You can view the GUI at http://127.0.0.1:3000/.

## Docker Tips
* To open another docker terminal for a running container, run the following on your home-machine:
    ```bash
    docker compose exec shell bash
    ```


## Resources
https://rlcard.org/
https://github.com/datamllab/rlcard-showdown/

# Doudizhu Bot

Bot for playing Doudizhu!

## For training/running the model:
This in not recommend since you'll need VERY outdated version of Node on your local machine.

1. To run this repo, you need...
    * Git
    * Python3.6+ 
    * Pip

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

4. Below are your options:
* To train dmc bot bot, run:
    ```bash
    python3 train.py --algorithm dmc --num_actor_devices 1 --num_actors 5 --save_interval 1
    ```

* To evaluate your trained model with 2 rule-based strategies, run the following **after replacing the .pth file with the path to your model**
    ```bash
    python3 evaluator.py --models results/doudizhu/0_0.pth doudizhu-rule-v1 doudizhu-rule-v1 --cuda '' --num_games 100
    ```
* To create a dodizhu.zip of your model that is formatted for the GUI, run:
    ```bash
    python3 zip_model.py
    ```

* To train a rarsms bot bot, run:
    ```bash
    python3 train.py --douzerox_path ./trained_models/douzerox/0_0.pth --algorithm rarsms --num_actor_devices 1 --num_actors 5 --save_interval 1

    ```


## For Running the RL-Card Showdown GUI:
The RL-Card Showdown GUI runs on a VERY outdated version of node, so we recommend running it in containers (as instructed below).

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
4.  * To test the trained model against other bots in the GUI, open http://127.0.0.1:3000/, upload the zip of your model in /results, and Launch a tournament.


## Docker Tips
* To open another docker terminal for a running container, run the following on your home-machine:
    ```bash
    docker compose exec shell bash
    ```


## Resources
* https://rlcard.org/
* https://github.com/datamllab/rlcard-showdown/

# Doudizhu Bot

Bot for playing Doudizhu

## Prerequisites
To run this repo, you need...
* Python3.6+ 
* Pip
* [Node](https://nodejs.org/) (comes with npm which you also need)


## Setup
1. Clone the required submodules
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

* TO run this program, run the following:
    ```bash
    python3 main.py
    ```


## Resources
https://rlcard.org/
https://github.com/datamllab/rlcard-showdown/
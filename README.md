For further insights on design choices and explanation of the different parts please read the Implementation part of the thesis (can be found here: https://git-ce.rwth-aachen.de/smith-project/smith-abschlussarbeiten/ba-pieper/-/tree/main/)
## Installation
This thesis has been done using python 3.10, please ensure correct python version as it uses some features introduced with python 3.10
1. Install all needed libraries: `pip install -r requirements.txt` requirements.txt can be found in 06_Implementation\Implementation
2. Edit `options.json` with your own credetials in the database_connection part:
   
    ```json
    {
      "user": DB_USERNAME,
      "password": DB_PASSWORT,
      "database": "SMITH_MIMIC/3",
      "ssh": {
        "host": "137.226.78.84",
        "username": SSH_USERNAME,
        "password": SSH_PASSWORT,
        "forward_host": "137.226.191.195",
        "forward_port": 3306
      }
    }
    ```


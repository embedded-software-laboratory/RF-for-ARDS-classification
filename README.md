For further insights on design choices and explaination of the different parts please read the Implemtation part of the thesis
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


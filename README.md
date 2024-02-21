# Random Forest for the retrospective classification of ARDS in ICU time-series data

This repository contains the implementation of a random forest algorithm for the retrospecitve classification of patients with Acute Respiratory Distress Syndrome (ARDS) in intensive care time-series data. For detailed information on how to use this software, please see the documentation folder (Documentation is still in progress). 

This software is still in development and new features may be added in the future. 

# License 
Code licensed under the GPL License. See LICENSE.txt file for terms.

# Important information on the GitLab / GitHub repositories
The repository is maintained at the GitLab of the Embedded Software Laboratory and mirrored to this GitHub repository. That means, changes on the GitLab repository are automatically, regulary pushed to the GitHub repository. **During this process, the content of the GitHub repository gets overwritten completely, so please don't push any changes into the GitHub repository directly. Otherwise, your work may get lost.**

# Installation
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


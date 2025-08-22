## Miniforge Setup Instructions

1. **Uninstall previous installations**  
   If you have *Anaconda* or *Miniconda* previously installed, uninstall them first.

2. **Rename the `.condarc` file**  
   Locate the `.condarc` file under the user profile directory and rename it to `.old`.  
   This ensures a new one is created in the new Miniforge environment.

3. **Add Miniforge to PATH**  
   Add the following path variable for the user:  
```
C:\Users**username**\AppData\Local\miniforge3
```
4. **Configure SSL verification**  
From the Miniforge prompt, set the SSL verify to the certificate:  
```
conda config --set ssl_verify C:\SoftwareSupport\cacert_org.pem
```
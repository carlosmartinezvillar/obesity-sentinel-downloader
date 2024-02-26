# cfgs (configs)

This dir has the files used to set up everything:

* `Dockerfile`: the setup for the ocker container used to run [`download.py`](../download.py)
* `pod.yml`: Main file that sets up (everything really) the kubernetes pod used in NRP's nautilus. `DATA_DIR` is set here as an environment variable pointing to the persistent volume claim (PVC) used to store the images. The access keys stored in the secret (`secret_config.yml`) are also set to env variables, and sets the correct pre-built container.
* `pvc.yml`: The persistent volume claim.
* `secret_config.yml`: The secret to store s3 keys.

Hence, things were done in the following order (after container was built):

1. `kubernetes create -f secret_config.yml`
2. `kubernetes create -f pvc.yml`
3. `kubernetes create -f pod.yml`
4. `kubernetes exec -it obesity-pod -- /bin/bash`
5. `git clone https://github.com/carlosmartinezvillar/obesity-sentinel-downloader.git` (inside container)
6. `python3 download.py` (inside the container)
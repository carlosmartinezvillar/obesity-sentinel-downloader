# obesity-sentinel-downloader

The [download.py](./download.py) script searches, filters, and downloads the Sentinel-2 images used as inputs in the upcoming paper ? [(link to paper)](https://). This was done by using [final.csv](./final.csv) as a set of Missouri census tract polygons that can be reliably linked to obesity rates in the state (work done in [link to previous paper](https://)). Files are downloaded from ESA's Copernicus Dataspace Ecosystem via S3 storage.

## Prerequisites

### Libraries

In order to run [download.py](./download.py) the following are needed: 

* `rclone`: A copy of rclone installed in the system to download the products via S3.

In addition to the following Python libraries:

* `geopandas`
* `matplotlib`
* `numpy`
* `pandas`
* `rasterio`
* `shapely==2.0.1`
* `requests`

### Docker image

A ready-to-use docker image with all the previously mentioned libraries pre-installed is available [here]
(https://hub.docker.com/repository/docker/cimartinezvillar/obesity-downloader/general).

### ESA Dataspace Access Keys

Products are downloaded via S3 from ESA's Dataspace ecosystem, which requires access keys. Instructions on how to do this can be found [here]
(https://documentation.dataspace.copernicus.eu/APIs/S3.html).

By default `rclone` will read the credentials set in `/root/.config/rclone/rclone.conf`. If you're using this file to set up the credentials, then the settings for your remote called `esa` should look like this:

```
[esa]
type = s3
provider = Other
access_key_id = <my_access_key_id>
secret_access_key = <my_secret_access_key>
endpoint = s3.dataspace.copernicus.eu
```

Otherwise `rclone` will use env variables following the convention `RCLONE_CONFIG_REMOTE_VARIABLE`. Where `REMOTE` is the name of the remote (`ESA` for above) and `VARIABLE` each variable needed (e.g.: `RCLONE_CONFIG_ESA_TYPE=S3`, `RCLONE_CONFIG_ESA_SECRET_ACCESS_KEY=<access_key>`).


## How-to
To run, do

```
python3 download.py
```
Figures will plotted and saved under the `figs/` directory. The images will saved in a directory named after the enviroment variable `DATA_DIR`.

## Directory/Downloaded files

The downloaded files are placed in `DATA_DIR`, with each .SAFE subdirectory corresponding to a sentinel product. Each of these folders will contain the three RGB bands, the scene classification 20m resolution band, and the metadata xml file of the product. Unless something goes horribly wrong, it should look like this:

```
/data
├── S2B_MSIL2A_20230531T183919_N0509_R070_T10SFG_20230531T225032.SAFE
│   ├── MTD_MSIL2A.xml
│   ├── T10SFG_20230531T183919_B02_10m.jp2
│   ├── T10SFG_20230531T183919_B03_10m.jp2
│   ├── T10SFG_20230531T183919_B04_10m.jp2
│   └── T10SFG_20230531T183919_SCL_20m.jp2
└── S2B_MSIL2A_20230601T180919_N0509_R084_T11SQU_20230601T224315.SAFE
    ├── MTD_MSIL2A.xml
    ├── T11SQU_20230601T180919_B02_10m.jp2
    ├── T11SQU_20230601T180919_B03_10m.jp2
    ├── T11SQU_20230601T180919_B04_10m.jp2
    └── T11SQU_20230601T180919_SCL_20m.jp2
```

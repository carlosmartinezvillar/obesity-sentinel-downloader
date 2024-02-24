# obesity-sentinel-downloader

## Prerequisites



**docker image**

A docker image with all the previously mentioned packages pre-installed is available [here](https://hub.docker.com/repository/docker/cimartinezvillar/obesity-downloader/general).

## How-to

```
python3 download.py -d
```

```
python3 download.py -p
```

## Directory/Downloaded files

Once the files are downloaded, the directory should end up structured so that each .SAFE subdir will contain the three RGB bands, the scene classification, and the metadata xml file of the product. Unless something goes horribly wrong, it should look like this:

```
/data
├── S2B_MSIL2A_20230531T183919_N0509_R070_T10SFG_20230531T225032.SAFE
│   ├── MTD.xml
│   ├── T10SFG_20230531T183919_B02_10m.jp2
│   ├── T10SFG_20230531T183919_B03_10m.jp2
│   ├── T10SFG_20230531T183919_B04_10m.jp2
│   └── T10SFG_20230531T183919_SCL_20m.jp2
└── S2B_MSIL2A_20230601T180919_N0509_R084_T11SQU_20230601T224315.SAFE
    ├── MTD.xml
    ├── T11SQU_20230601T180919_B02_10m.jp2
    ├── T11SQU_20230601T180919_B03_10m.jp2
    ├── T11SQU_20230601T180919_B04_10m.jp2
    └── T11SQU_20230601T180919_SCL_20m.jp2
```
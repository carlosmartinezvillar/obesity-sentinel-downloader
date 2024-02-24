################################################################################
# base image
################################################################################
FROM ubuntu:jammy

RUN apt-get update && apt-get install -y \
	python3.10 \
	python3-pip \
	git \
	gdal-bin \
	libgdal-dev \
	libopenjp2-7 \
	libopenjp2-7-dev \
	rclone

RUN pip install -U pip \
	&& pip install geopandas matplotlib numpy pandas rasterio shapely==2.0.1 requests

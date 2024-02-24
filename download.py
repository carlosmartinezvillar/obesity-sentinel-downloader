import requests
import pandas as pd
import geopandas as gpd
import shapely
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import subprocess as sp
import argparse

# DATA_DIR = '/sentinel-images'
DATA_DIR = './data'
REMOTE   = 'esa:'
################################################################################
# HELPER FUNCTIONS
################################################################################
def search_and_parse(geometry_wkt=None):	

	if geometry_wkt is None:
		with open('query_wkt.txt','r') as fp:
			geometry_wkt = fp.read()

	base_url = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"

	#07-10 to 08-15 is okay...
	payload = {
		'productType':'S2MSI2A',
		'cloudCover':'[0,5]',
		'startDate':'2022-07-01T00:00:00Z',
		'completionDate':'2022-08-31T23:59:59Z',
		'geometry':geometry_wkt,
		'sortParam':'startDate',
		'sortOrder':'ascending',
		'maxRecords':500,
		'page':1
	}

	print("Searching within: %s--%s" % (payload['startDate'],payload['completionDate']))
	resp = requests.get(base_url,params=payload) #HTTPs request

	if resp.status_code != 200: #check correct response
		print("HTTP request error. Return code different from 200.")
		return None

	resp_json = resp.json() #https string to json

	if len(resp_json['features']) == 0: #nr of product >0 ?
		print("HTTP response 200, but empty list of products.")
		return None
	else:
		print("%i products found." % len(resp_json['features']))

	polygons,s3_urls,cloudcov = [],[],[]
	for f in resp_json['features']:
		polygons += [shapely.geometry.shape(f['geometry'])]
		s3_urls  += [f['properties']['productIdentifier']]
		cloudcov += [f['properties']['cloudCover']]

	return gpd.GeoDataFrame({'s3':s3_urls,'clouds':cloudcov,'geometry':polygons},geometry='geometry')


def filter_overlapping(results):
	'''
	Select the best product of those overlapping. For each tile a "best" product
	is chosen by:
		1. Selecting the product largest area
		2. If two products have the same area, select the one with less clouds
	'''
	tiles    = np.array([str(i).split('_')[5] for i in results['s3']])
	filtered = []
	for u in np.unique(tiles):
		subset = results[u == tiles]
		top_two_area = subset.area.nlargest(2)
		if len(top_two_area) > 1:
			if top_two_area.iloc[0]==top_two_area.iloc[1]:
				subsubset = subset.loc[top_two_area.index].reset_index(drop=True)
				chosen = subsubset.iloc[subsubset['clouds'].argmin()]
			else:
				chosen = subset.loc[top_two_area.index[0]]
		else:	
			chosen = subset.iloc[0]

		filtered.append(chosen)

	return gpd.GeoDataFrame(filtered,geometry='geometry').reset_index(drop=True)


def download_images(gdf):
	'''
	Takes the geopandas GeoDataFrame 'gdf', iterates through the rows downloading
	the RGB bands + xml metadata files of each product.
	'''

	# Fix .SAFE file paths (id est "eodata/" to "EODATA/")
	source_paths = ["EODATA"+_[7:] for _ in gdf['s3']]
	N = len(source_paths)

	for i,safe_src in enumerate(source_paths):
		# 0. Path strings and such
		safe_src = REMOTE + source_paths[i]
		safe_dir = safe_src.split('/')[-1]
		out_dir  = '/'.join([DATA_DIR,safe_dir])
		print("\n[%i/%i] Downloading %s" % (i,N,safe_dir))
		print('_'*80)

		# 1. Get middle folder (L2A_AGGGGGG_DDDDDDTDDDDDD/) D:datastrip, G:granule
		proc1     = sp.run(["rclone","lsd",safe_src+"/GRANULE"],stdout=sp.PIPE) #BANDS
		subdir    = proc1.stdout.decode().split()[-1]

		# 2. Get RGB bands
		print("Getting RGB bands...")
		bands_dir = '/'.join([safe_src,"GRANULE",subdir,"IMG_DATA","R10m"])
		proc2     = sp.run([
			"rclone","copy","-P","--dry-run",bands_dir,out_dir,
			"--include","*02_10m.jp2",
			"--include","*03_10m.jp2",
			"--include","*04_10m.jp2"
			])

		# 3. Get SCL 20m band
		print("Getting SCL band (scene classification)...")
		date,tile = safe_src.split('_')[2:6:3]
		scl_file  = '_'.join([tile,date,'SCL_20m.jp2'])
		scl_src   = '/'.join([safe_src,"GRANULE",subdir,"IMG_DATA","R20m",scl_file])
		proc3     = sp.run(["rclone","copy","-P",scl_src,out_dir])

		# 4. Download metadata/xml file
		print("\nGetting XML/Metadata file...")
		xml_src = '/'.join([safe_src,"MTD_MSIL2A.xml"])
		proc4   = sp.run(["rclone","copy","-P",xml_src,out_dir])


def get_chips_worker(src_reader):
	pass


def get_chips():
	pass

################################################################################
# MAIN #
################################################################################
if __name__ == '__main__':

	plot     = False
	download = True

	############################################################
	# I. SET SEARCH AREA
	############################################################
	# 1.1 GET POLYGONS IN DAHU ET AL. 2024
	df             = pd.read_csv('final.csv')
	df['geometry'] = df['geometry'].apply(shapely.wkt.loads)
	census_tracts  = gpd.GeoDataFrame(df,geometry='geometry').set_crs(crs="EPSG:4326")
	mo_union_poly  = census_tracts.unary_union #Polygon
	mo_union_poly  = shapely.geometry.Polygon(mo_union_poly.exterior.coords) #remove holes

	# 1.2 SIMPLIFY SEARCH STRING
	mo_simple     = mo_union_poly.simplify(tolerance=0.05,preserve_topology=False) #Polygon
	mo_simple_str = mo_simple.wkt #string
	n_v           = len(mo_simple_str.split(','))
	print("Search area set to polygon of %i vertices." % n_v)
	with open('query_wkt.txt','w') as fp: # Save to wkt
		fp.write(mo_simple_str)
	print("WKT simple search area polygon written to 'query_wkt.txt'.")

	# 1.3 PLOT THE SIMPLIFIED POLYGON
	if plot:
		fig, ax = plt.subplots()
		ax.set_aspect('equal')
		mo_simple_series = gpd.GeoSeries(mo_simple).set_crs(crs="EPSG:4326")
		mo_simple_series.plot(ax=ax,color='white',edgecolor='black',linewidth=0.2)
		gpd.GeoSeries(mo_union_poly).set_crs(crs="EPSG:4326").boundary.plot(ax=ax,color='red',linewidth=0.2)
		ax.set_title("MO with %i vertices" % n_v)
		plt.savefig('./figs/mo_simplified.png')
		plt.close()

	############################################################
	# II. SEARCH
	############################################################
	# 2.1 SEARCH (ESA OpenSearch)
	print("-"*60)
	results = search_and_parse(mo_simple_str)
	print("-"*60)	

	# 2.2 PLOT PRODUCTS RETURNED
	if plot:
		fig, ax = plt.subplots(1,1,figsize=(20,20))
		results['geometry'].plot(ax=ax,color='green',alpha=0.25,linewidth=0.2,edgecolor='black')
		census_tracts.boundary.plot(ax=ax,linewidth=0.2,color='black')
		ax.set_title("N=%i" % len(results))
		for i,c in enumerate(results.centroid):
			rot,tile  = results['s3'].iloc[i].split('_')[4:6]
			s = '_'.join([rot,tile])
			ax.text(x=c.x,y=c.y,s=s,ha='center',va='center',fontsize=7,color='black')
		plt.savefig('./figs/raster_overlay.png')
		plt.close()
		print("Plot written to ./figs/raster_overlay.png")

	############################################################
	# III. FILTER RESULTS
	############################################################
	# 3.1 FILTER TO NON-OVERLAPPING PRODUCTS
	filtered_gdf = filter_overlapping(results)
	print("* Filtered to %i (keeping laregest area image in tile)" % len(filtered_gdf))

	# 3.2 PLOT FILTERED
	if plot:
		fig, ax = plt.subplots(1,1,figsize=(20,20))
		census_tracts.boundary.plot(ax=ax,linewidth=0.2,color='black')
		for i,c in enumerate(filtered_gdf.centroid): #tile names
			rot,tile = filtered_gdf['s3'].iloc[i].split('_')[4:6]
			date     = filtered_gdf['s3'].iloc[i].split('_')[2].split('T')[0]
			s = '\n'.join([rot,tile,date])
			ax.text(x=c.x,y=c.y,s=s,ha='center',va='center',fontsize=7,color='black')
		filtered_gdf.set_crs("EPSG:4326").plot(ax=ax,linewidth=0.2,color='g',alpha=0.25,edgecolor='g')
		max_clouds = filtered_gdf['clouds'].max()
		avg_clouds = filtered_gdf['clouds'].mean()
		ax.set_title("N=%i, mean clouds=%.3f, max clouds=%.3f" % (len(filtered_gdf),avg_clouds,max_clouds))
		plt.savefig("./figs/raster_filtered.png")
		plt.close()
		print("Plot written to './figs/raster_filtered.png'.")

	# 3.3 REMOVE FUNNY TILES (UTM 14, 16)
	remove      = ['T14TQK','T14TQL','T16SBJ','T16SBH','T16SBG','T16SBF','T16SBE']
	tiles       = np.array([s3.split('_')[5] for s3 in filtered_gdf['s3']])
	remove_mask = np.array([(_==tiles) for _ in remove]).any(axis=0)
	final_gdf   = filtered_gdf[~remove_mask]
	print("* Filtered to %i (removed tiles in UTM 14 & 16)" % len(final_gdf))

	# 3.4 PLOT WITH 6 TILES REMOVED
	if plot:
		fig, ax = plt.subplots(1,1,figsize=(20,20))
		census_tracts.boundary.plot(ax=ax,linewidth=0.2,color='black')
		for i,c in enumerate(final_gdf.centroid):
			rot,tile = final_gdf['s3'].iloc[i].split('_')[4:6]
			date     = final_gdf['s3'].iloc[i].split('_')[2].split('T')[0]
			s = '\n'.join([rot,tile,date])
			ax.text(x=c.x,y=c.y,s=s,ha='center',va='center',fontsize=7,color='black')	
		final_gdf.set_crs("EPSG:4326").plot(ax=ax,linewidth=0.2,color='g',alpha=0.25,edgecolor='g')
		max_clouds = filtered_gdf['clouds'].max()
		avg_clouds = filtered_gdf['clouds'].mean()
		N          = len(final_gdf)
		ax.set_title("N=%i, mean clouds=%.3f, max clouds=%.3f" % (N,avg_clouds,max_clouds))			
		plt.savefig("./figs/cleaned_final.png")
		plt.close()
		print("Plot written to './figs/cleaned_final.png'.")	


	############################################################
	# IV. DOWNLOAD
	############################################################
	if download:
		download_images(final_gdf)

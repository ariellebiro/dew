# import pandas as pd
import logging
import sys
import time

from dask import config 
from dask.distributed import LocalCluster
from dotenv import load_dotenv

from downscaling import run_downscaling_workflow,initialize_logger


# ----------------------TEST SCRIPT----------------------- #
if __name__=="__main__":
    client = LocalCluster().get_client()
    print(client.dashboard_link)

    import logging
    import os
    import geopandas as gpd
    import s3fs


    from dotenv import load_dotenv
    from shapely.geometry import box

    load_dotenv()
    config.set({"logging.distributed": "error"})
    initialize_logger()

    time.sleep(5)
    logging.info(f"Starting test run {client.dashboard_link}")

    # Set paths and parameters
    shapefile = "/home/ubuntu/dask-sst-sandbox/tests/indian-creek.json"
    # shapefile = "/mnt/tests/indian-creek.json"
    aorc_path_template = "noaa-nws-aorc-v1-1-1km/{year}.zarr"
    nasa_historical_path_template = (
        "nex-gddp-cmip6/NEX-GDDP-CMIP6/{model}/historical/r1i1p1f1/pr/pr_day_{model}_historical_r1i1p1f1_gn_{year}.nc"
    )
    nasa_future_path_template = (
        "nex-gddp-cmip6/NEX-GDDP-CMIP6/{model}/{ssp}/r1i1p1f1/pr/pr_day_{model}_{ssp}_r1i1p1f1_gn_{year}.nc"
    )
    output_dir = "s3://wejo-xfer/downscaled_future"  # Update this for your output location

    aorc_variable_name = "APCP_surface"
    nasa_variable_name = "pr"

    # Set testing parameters
    historical_years = [1980]  # [1980]  # Historical year for testing #list(range(1980, 2015)) for all historical years
    future_years = [2015]  # Future year for testing
    models = ["CanESM5"]
    ssps = ["ssp245"]
    buffer = 0.1
    doys = [1]  # Example DOYs for testing, list(range(1, 366)) for all DOYs

    # Read shapefile and create buffered bounds
    shape = gpd.read_file(shapefile)
    projected_shape = shape.to_crs(epsg=4326)  # Replace with a projected CRS for your region
    # Convert buffered bounds to a bounding box
    minx, miny, maxx, maxy = shape.total_bounds
    bounding_box = box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
    buffered_bounds = gpd.GeoDataFrame({"geometry": [bounding_box]}, crs=shape.crs)

    # set up s3  connection
    s3 = s3fs.S3FileSystem(key=os.getenv("AWS_ACCESS_KEY_ID"), secret=os.getenv("AWS_SECRET_ACCESS_KEY"))

    future = run_downscaling_workflow(
        shapefile=shape,
        aorc_path_template=aorc_path_template,
        nasa_historical_path_template=nasa_historical_path_template,
        future_path_template=nasa_future_path_template,
        historical_years=historical_years,
        future_years=future_years,
        models=models,
        ssps=ssps,
        buffered_bounds=buffered_bounds,
        doys=doys,
        s3=s3,
        output_dir=output_dir,
    )
    logging.info("Test run completed successfully.")

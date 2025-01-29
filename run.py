# import pandas as pd
import logging
import os
import sys
import time
import warnings

import s3fs
from dask import config
from dask.distributed import LocalCluster
from dotenv import load_dotenv


from downscaling import (
    AORC_PATH_TEMPLATE,
    NASA_FUTURE_PATH_TEMPLATE,
    NASA_HISTORICAL_PATH_TEMPLATE,
    initialize_logger,
    run_downscaling_workflow,
    init
)

# ----------------------TEST SCRIPT----------------------- #
if __name__ == "__main__":

    initialize_logger()
    client = LocalCluster().get_client()
    time.sleep(5)
    print(client.dashboard_link)
    
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    config.set({"logging.distributed": "error"})
    

    # Set paths and parameters
    aoi_file = "/home/ubuntu/dask-sst-sandbox/tests/indian-creek.json"

    output_dir = "s3://wejo-xfer/downscaled_future/indian-creek-test"

    aorc_variable_name = "APCP_surface"
    nasa_variable_name = "pr"

    historical_years = range(1980, 1981)
    future_years = range(2015, 2016)

    models = ["CanESM5"]

    ssps = ["ssp245"]

    buffer = .1

    doys = range(6,7)  # Example DOYs for testing, list(range(1, 366)) for all DOYs

    aoi_gdf, buffered_bounds = init(aoi_file, buffer=buffer)

    # set up s3  connection
    s3 = s3fs.S3FileSystem(key=os.getenv("AWS_ACCESS_KEY_ID"), secret=os.getenv("AWS_SECRET_ACCESS_KEY"))

    logging.info(f"Starting test run {client.dashboard_link}")

    future = run_downscaling_workflow(
        aoi_gdf=aoi_gdf,
        aorc_path_template=AORC_PATH_TEMPLATE,
        nasa_historical_path_template=NASA_HISTORICAL_PATH_TEMPLATE,
        future_path_template=NASA_FUTURE_PATH_TEMPLATE,
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

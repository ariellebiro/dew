# import pandas as pd
import logging
import os
import sys
import time
import warnings
import fsspec
import s3fs
from dask import config, compute
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


    #keep memory safe
    cluster = LocalCluster(
        n_workers=6,
        threads_per_worker=1,
        memory_limit="8GB",
        dashboard_address=":8787",
    )

    client = cluster.get_client()
    time.sleep(5)
    print(client.dashboard_link)
    
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")
    config.set({"logging.distributed": "error"})
    

    # Set paths and parameters
    #aoi_file = "/mnt/tests/indian-creek.json"
    aoi_file = "/home/ubuntu/dask-sst-sandbox/tests/indian-creek.json"

    output_dir = "s3://hydromet/downscaled_future/duwamish"

    aorc_variable_name = "APCP_surface"
    nasa_variable_name = "pr"

    historical_years = range(1980, 1986)
    future_years = range(2050, 2051)

    models = ["CESM2"]

    ssps = ["ssp245"]

    buffer = 1

    doys = range(1, 90) 
    
     # Example DOYs for testing, list(range(1, 366)) for all DOYs

    aoi_gdf, buffered_bounds = init(aoi_file, buffer=buffer)

    # set up s3  connection
    s3_private = s3fs.S3FileSystem(key=os.getenv("AWS_ACCESS_KEY_ID"), secret=os.getenv("AWS_SECRET_ACCESS_KEY"))
    s3_public = fsspec.filesystem("s3", anon=True)

    logging.info(f"Starting test run {client.dashboard_link}")

    tasks = run_downscaling_workflow(
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
        output_dir=output_dir,
        s3_private=s3_private,
        s3_public=s3_public,
    )

    results = compute(*tasks) #trigger execution
    logging.info("Test run completed successfully.")
    client.close()

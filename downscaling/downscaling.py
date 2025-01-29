"""
This script orchestrates the downscaling workflow for climate data, including the following phases:

1. **Phase 1: Processing Historical and Future Data**
   - Reads, organizes, and sorts historical data by day of year (DOY).
   - Loads AORC and NASA historical data, clips to the buffered bounds of the region of interest, organizes by daily values, and regrids AORC data to match the NASA grid for quantile mapping.
   - Reads, organizes, and sorts future data by DOY.
   - Loads NASA future data, clips to the buffered bounds, and resamples to daily values.

2. **Phase 2: Quantile Mapping**
   - Fits quantile maps to historical data and applies them to future data for each DOY.

3. **Phase 3: Compiling Year Data, Regridding and Saving the Downscaled Data**
   - Compiles the transformed future data for all DOYs of a specific year into one dataset.
   - Regrids the compiled year data to match the original AORC grid and saves the result to S3.

The script uses Dask for parallel processing and delayed execution, xarray for handling multi-dimensional arrays, and pyresample for regridding. The data is read from and saved to S3 using s3fs.

Functions:
- `process_historical_data`: Processes historical data by reading, organizing, and sorting data by DOY.
- `process_future_data`: Processes future data by reading, organizing, and sorting data by DOY.
- `fit_and_apply_quantile_map`: Fits quantile maps to historical data and applies them to future data for each DOY.
- `compile_year_data`: Compiles the transformed future data for all DOYs of a specific year into one dataset.
- `regrid_and_save`: Regrids the compiled year data to match the original AORC grid and saves the result to S3.
- `run_downscaling_workflow`: Orchestrates the entire downscaling workflow, including processing historical and future data, fitting and applying quantile maps, compiling year data, regridding, and saving results.

Usage:
- Set the paths and parameters in the main section of the script.
- Call the `run_downscaling_workflow` function with the appropriate arguments.

Dependencies:
- dask
- xarray
- pandas
- cftime
- numpy
- skdownscale
- s3fs
- pyresample
- fsspec
- zarr
- dotenv
- geopandas
- shapely

"""

import logging
import os
from datetime import datetime, timedelta

import cftime
import fsspec
import numpy as np
import pandas as pd
import rioxarray
import s3fs
import xarray as xr
import zarr
from dask import delayed
from dotenv import load_dotenv
from pyresample import geometry, kd_tree
from skdownscale.pointwise_models import QuantileMapper


#############
###PHASE 1###
# reading, organizing, and sorting data by DOY
# separate functions for each dataset
@delayed
def process_historical_data(
    aorc_path_template,
    nasa_historical_path_template,
    shape,
    buffered_bounds,
    aorc_variable_name,
    nasa_variable_name,
    start_year,
    end_year,
    s3,
    model,
    doys,
):
    """
    Processes historical data by reading, organizing, and sorting data by the day of year (DOY).
    This includes loading AORC and NASA historical data, clipping to the buffered_bounds of Kanawha, organizing by daily values, and regridding AORC data to match the NASA grid for quantile mapping.

    Parameters:
        aorc_path_template (str): Template path for AORC data on S3.
        nasa_historical_path_template (str): Template path for NASA historical data on S3.
        shape (GeoDataFrame): Shapefile of the region of interest.
        buffered_bounds (GeoDataFrame): Buffered bounds of the region of interest.
        aorc_variable_name (str): Variable name for AORC data.
        nasa_variable_name (str): Variable name for NASA data.
        start_year (int): Start year for processing historical data.
        end_year (int): End year for processing historical data.
        s3 (S3FileSystem): S3FileSystem object for reading data from S3.
        model (str): Model name for NASA data.
        doys: list of days of the year to process.

    Returns:
        nasa_combined (xarray.DataArray): Combined NASA historical data organized by DOY.
        aorc_combined (xarray.DataArray): Combined AORC historical data organized by DOY.
        original_aorc_grid (xarray.DataArray): Original AORC grid reference for regridding future data.
    """
    logging.info(f"Processing historical data for model: {model}, years {start_year}-{end_year}")
    nasa_data_list = []
    aorc_data_list = []

    # Initialize original AORC grid reference (to regrid future data)
    original_aorc_grid = None

    for year in range(start_year, end_year + 1):
        logging.info(f"Processing historical data for year: {year}")

        for doy in doys:
            # skip february 29
            date = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)
            if date.month == 2 and date.day == 29:
                logging.info(f"Skipping February 29 for year {year} (DOY {doy})")
                continue

            logging.info(f"Processing AORC data for DOY {doy} in year {year}")

            # generate time slice for DOY
            start_time = date
            end_time = date + timedelta(days=1) - timedelta(seconds=1)

            # Load AORC data
            try:
                aorc_data = xr.open_zarr(s3.get_mapper(aorc_path_template.format(year=year)), chunks={"time": 24})[
                    aorc_variable_name
                ]
                aorc_data = aorc_data.sel(time=slice(start_time, end_time))

                if len(aorc_data["time"]) != 24:
                    logging.warning(
                        f"Expected 24 hourly slice for DOY {doy} in {year}, but found {len(aorc_data['time'])}. Skipping."
                    )

                # rename spatial dims if necessary
                if "latitude" in aorc_data.coords and "longitude" in aorc_data.coords:
                    aorc_data = aorc_data.rename({"latitude": "lat", "longitude": "lon"})

                # assign spatial dims and CRS
                aorc_data = aorc_data.rio.write_crs("EPSG:4326", inplace=True)
                aorc_data = aorc_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

                aorc_clipped = aorc_data.rio.clip(buffered_bounds.geometry, shape.crs)
                aorc_daily = aorc_clipped.resample(time="1D").sum()

                # save the AORC grid once for reference
                if original_aorc_grid is None:
                    original_aorc_grid = aorc_clipped

                # append AORC data for this doy
                aorc_data_list.append(aorc_daily)
            except Exception as e:
                logging.error(f"Error processing AORC data for DOY {doy} in year {year}: {e}")
                continue

        try:
            # Load NASA historical data
            nasa_data = xr.open_dataset(s3.open(nasa_historical_path_template.format(model=model, year=year)))[
                nasa_variable_name
            ]
            nasa_data = nasa_data.assign_coords(lon=(((nasa_data.lon + 180) % 360) - 180)).sortby("lon")
            nasa_data = nasa_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            nasa_data = nasa_data.rio.write_crs("EPSG:4326", inplace=True)
            nasa_clipped = nasa_data.rio.clip(buffered_bounds.geometry, shape.crs)
            nasa_daily = nasa_clipped * 86400
            nasa_daily.attrs["units"] = "kg/m^2"

            # Convert time to datetime64 within the function itself
            if isinstance(nasa_daily["time"].values[0], cftime.datetime):
                print(f"Converting time to datetime64 for {nasa_variable_name} in year {year}")
                nasa_daily["time"] = pd.to_datetime([t.strftime("%Y-%m-%d") for t in nasa_daily["time"].values])

            # Sort NASA data by DOY and filter by the specified DOYs
            nasa_daily = nasa_daily.sortby("time")
            nasa_daily = nasa_daily.sel(time=nasa_daily.time.dt.dayofyear.isin(doys))

            nasa_data_list.append(nasa_daily)
        except Exception as e:
            logging.error(f"Error processing NASA data for year {year}: {e}")
            continue

        # Define Pyresample source and target geometries
        source_lons = aorc_daily["lon"].values
        source_lats = aorc_daily["lat"].values
        target_lons = nasa_daily["lon"].values
        target_lats = nasa_daily["lat"].values

        if source_lons.ndim == 1 and source_lats.ndim == 1:
            source_lons, source_lats = np.meshgrid(source_lons, source_lats)
        if target_lons.ndim == 1 and target_lats.ndim == 1:
            target_lons, target_lats = np.meshgrid(target_lons, target_lats)

        source_grid = geometry.SwathDefinition(lons=source_lons, lats=source_lats)
        target_grid = geometry.SwathDefinition(lons=target_lons, lats=target_lats)

        # Resample AORC data to match NASA grid using Pyresample
        regridded_aorc_data = []
        for time_step in aorc_daily["time"]:
            time_slice = aorc_daily.sel(time=time_step).values
            regridded_slice = kd_tree.resample_nearest(
                source_grid,
                time_slice,
                target_grid,
                radius_of_influence=25000,  # Adjust based on your spatial resolution needs
                fill_value=np.nan,
            )
            regridded_aorc_data.append(regridded_slice)

        # Stack regridded slices into a DataArray
        regridded_aorc_daily = xr.DataArray(
            np.stack(regridded_aorc_data, axis=0),
            dims=("time", "lat", "lon"),
            coords={"time": aorc_daily["time"], "lat": nasa_daily["lat"], "lon": nasa_daily["lon"]},
            attrs=aorc_daily.attrs,
        )

        aorc_data_list[-1] = regridded_aorc_daily

    # Combine all years
    nasa_combined = xr.concat(nasa_data_list, dim="time")
    aorc_combined = xr.concat(aorc_data_list, dim="time")
    logging.info(f"Completed processing historical data for model: {model}")

    return nasa_combined, aorc_combined, original_aorc_grid


@delayed
def process_future_data(
    nasa_future_path_template,
    buffered_bounds,
    shape,
    nasa_variable_name,
    quantile_mappers,
    aorc_combined,
    original_aorc_grid,
    start_year,
    end_year,
    output_dir,
    s3,
    model,
    ssp,
    doys,
):
    """
    Processes future NASA data by reading, organizing, and sorting data by DOY (day of year).
    This includes loading NASA future data (specifically 2015-2100), clipping to the buffered bounds, sorting by daily values.

    Parameters:
        nasa_future_path_template (str): Template path for NASA future data on S3.
        buffered_bounds (GeoDataFrame): Buffered bounds of the region of interest.
        shape (GeoDataFrame): Shapefile of the region of interest.
        nasa_variable_name (str): Variable name for NASA data.
        quantile_mappers (dict): Dictionary of QuantileMapper objects for each DOY (not needed for this, set to None)
        aorc_combined (xarray.DataArray): Combined AORC historical data organized by DOY.
        original_aorc_grid (xarray.DataArray): Original AORC grid reference for regridding future data.
        start_year (int): Start year for processing future data.
        end_year (int): End year for processing future data.
        output_dir (str): Output directory for saving regridded data.
        s3 (S3FileSystem): S3FileSystem object for reading and writing data to S3.
        model (str): Model name for NASA data.
        ssp (str): Shared Socioeconomic Pathway (SSP) for future data.
        doys: list of days of the year to process.

    Returns:
        nasa_future (xarray.DataArray): NASA future data organized by DOY.
    """
    logging.info(f"Processing future data for model: {model}, SSP: {ssp}, years {start_year}-{end_year}")

    nasa_future_list = []

    for year in range(start_year, end_year + 1):
        logging.info(f"Processing future data for year: {year}")

        # Load NASA future data
        nasa_future = xr.open_dataset(s3.open(nasa_future_path_template.format(model=model, ssp=ssp, year=year)))[
            nasa_variable_name
        ]
        nasa_future = nasa_future.assign_coords(lon=(((nasa_future.lon + 180) % 360) - 180)).sortby("lon")
        nasa_future = nasa_future.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        nasa_future = nasa_future.rio.write_crs("EPSG:4326", inplace=True)
        nasa_future_clipped = nasa_future.rio.clip(buffered_bounds.geometry, shape.crs)
        nasa_future_daily = nasa_future_clipped * 86400
        nasa_future_daily.attrs["units"] = "kg/m^2"

        # Convert time to datetime64 directly in the function
        if isinstance(nasa_future_daily["time"].values[0], cftime.datetime):
            print(f"Converting time to datetime64 for {nasa_variable_name} in year {year}")
            nasa_future_daily["time"] = pd.to_datetime(
                [t.strftime("%Y-%m-%d") for t in nasa_future_daily["time"].values]
            )

        # Sort and filter future data by DOY
        nasa_future_daily = nasa_future_daily.sortby("time")
        nasa_future_daily = nasa_future_daily.sel(time=nasa_future_daily.time.dt.dayofyear.isin(doys))

        nasa_future_list.append(nasa_future_daily)
        nasa_future = xr.concat(nasa_future_list, dim="time")

        # You can now process future data here, like fitting quantile mapping, etc
        logging.info(f"Completed processing future data for model: {model}, SSP: {ssp}")
    return nasa_future


#############
###PHASE 2###
# fitting quantile maps and applying to future data for DOY
@delayed
def fit_and_apply_quantile_map(aorc_data, nasa_data, nasa_future_data, doys):
    """
    Fits quantile maps to historical data (quantile maps between AORC and NASA for historical period) and then applies them to future data for each day of year (DOY) for the entire period of interest. So quantile maps are fit for DOY 1 across all historical years (e.g. 1980-2014) and then applied to DOY 1 for all future years (e.g. 2015-2100).

    Parameters:
        aorc_data (xarray.DataArray): Combined AORC historical data organized by DOY.
        nasa_data (xarray.DataArray): Combined NASA historical data organized by DOY.
        nasa_future_data (xarray.DataArray): NASA future data organized by DOY.
        doys: list of days of the year to process.

    Returns:
        transformed_future_combined (xarray.DataArray): Transformed future data organized by DOY.
    """
    logging.info(f"Fitting and applying quantile mapping for {len(doys)} DOYs")

    quantile_mappers = {}

    # Fit quantile mappers by DOY for historical data
    for doy, nasa_group in nasa_data.groupby("time.dayofyear"):
        if doy not in aorc_data.groupby("time.dayofyear").groups:
            logging.warning(f"Skipping DOY {doy} as it is missing in AORC data.")
            continue

        # Flatten and mask out NaN values
        nasa_flat = nasa_group.values.flatten().reshape(-1, 1)
        aorc_flat = aorc_data.sel(time=nasa_group.time).values.flatten().reshape(-1, 1)
        mask = ~np.isnan(nasa_flat) & ~np.isnan(aorc_flat)

        # If no valid data for this DOY, skip it
        if nasa_flat[mask].size == 0 or aorc_flat[mask].size == 0:
            logging.warning(f"Insufficient data for DOY {doy}. Skipping.")
            continue

        # Fit QuantileMapper
        qm = QuantileMapper()
        qm.fit(aorc_flat[mask].reshape(-1, 1), nasa_flat[mask].reshape(-1, 1))
        quantile_mappers[doy] = qm

    logging.info(f"Completed fitting quantile mappers for {len(quantile_mappers)} DOYs")

    # Apply quantile mapping to future data
    transformed_future = []
    for doy, future_group in nasa_future_data.groupby("time.dayofyear"):
        if doy not in quantile_mappers:
            logging.warning(f"No QuantileMapper available for DOY {doy}. Skipping transformation.")
            continue

        # Flatten future data and apply quantile mapping
        future_flat = future_group.values.flatten()
        mask = ~np.isnan(future_flat)

        # If all values are NaN for this DOY, skip it
        if np.sum(mask) == 0:
            logging.warning(f"All values are NaN for DOY {doy}. Skipping transformation.")
            continue

        # Apply the transformation to the valid data points
        transformed_flat = np.full_like(future_flat, np.nan)
        transformed_flat[mask] = quantile_mappers[doy].transform(future_flat[mask].reshape(-1, 1)).flatten()

        # Reshape and append the transformed group
        transformed_group = transformed_flat.reshape(future_group.shape)
        transformed_future.append(
            xr.DataArray(
                transformed_group,
                dims=future_group.dims,
                coords=future_group.coords,
                attrs=future_group.attrs,
            )
        )

    # Combine all the transformed DOYs into one dataset
    transformed_future_combined = xr.concat(transformed_future, dim="time")
    logging.info(f"Completed applying quantile mapping and transforming future data.")

    return transformed_future_combined


#############
###PHASE 3###
# compiling data by future year, regridding, and save to s3
@delayed
def compile_year_data(future_data_for_doys):
    """
    Compiles the transformed future data for all DOYs of a specific year into one dataset.

    Parameters:
        future_data_for_doys (list): List of transformed future data organized by DOY.

    Returns:
        compiled_year_data (xarray.DataArray): Compiled future data for a specific year.
    """
    # Concatenate along the "time" dimension
    return xr.concat(future_data_for_doys, dim="time")


@delayed
def regrid_and_save(compiled_year_data, original_aorc_grid, output_dir, year, model, ssp, buffered_bounds):
    """
    Regrids the compiled year data to match the original AORC grid and saves the results to S3. After regridding is the final downscaled product.

    Parameters:
        compiled_year_data (xarray.DataArray): Compiled future data for a specific year.
        original_aorc_grid (xarray.DataArray): Original AORC grid reference for regridding future data.
        output_dir (str): Output directory for saving regridded data.
        year (int): Year of the future data being processed.
        model (str): Model name for NASA data.
        ssp (str): Shared Socioeconomic Pathway (SSP) for future data.
        buffered_bounds (GeoDataFrame): Buffered bounds of the region of interest.

    Returns:
        regridded_data_da (xarray.DataArray): Regridded future data for a specific year - this is the final downscaled product.
    """

    load_dotenv()

    s3 = s3fs.S3FileSystem(key=os.getenv("AWS_ACCESS_KEY_ID"), secret=os.getenv("AWS_SECRET_ACCESS_KEY"))
    # Reproject to align CRS
    compiled_year_data = compiled_year_data.rio.write_crs("EPSG:4326")
    original_aorc_grid = original_aorc_grid.rio.write_crs("EPSG:4326")

    # extract lat/lon
    target_lons = original_aorc_grid["lon"].values
    target_lats = original_aorc_grid["lat"].values

    # if lat/lon are 1d, create 2d meshgrids
    if target_lons.ndim == 1 and target_lats.ndim == 1:
        target_lons, target_lats = np.meshgrid(target_lons, target_lats)

    # define source and target pyresample geometries
    target_grid = geometry.SwathDefinition(lons=target_lons, lats=target_lats)

    regridded_slices = []

    # loop through time steps and regrid each slice
    for time_step in compiled_year_data["time"]:
        # select data for current time step
        time_slice = compiled_year_data.sel(time=time_step).values
        source_lons = compiled_year_data["lon"].values
        source_lats = compiled_year_data["lat"].values
        if source_lons.ndim == 1 and source_lats.ndim == 1:
            source_lons, source_lats = np.meshgrid(source_lons, source_lats)
        source_grid = geometry.SwathDefinition(lons=source_lons, lats=source_lats)

        # resample the current time slice
        regridded_slice = kd_tree.resample_gauss(
            source_grid, time_slice, target_grid, radius_of_influence=25000, sigmas=25000, fill_value=np.nan
        )
        regridded_slices.append(regridded_slice)

    # combine regridded slices along the time dimension
    regridded_data = np.stack(regridded_slices, axis=0)

    regridded_data_da = xr.DataArray(
        regridded_data,
        dims=("time", "lat", "lon"),
        coords={
            "time": compiled_year_data["time"],
            "lat": original_aorc_grid["lat"],
            "lon": original_aorc_grid["lon"],
        },
        attrs=compiled_year_data.attrs,
    )

    # rename the variable to pr
    regridded_data_da = regridded_data_da.rename("pr")

    # Save the regrided data to S3
    output_file = f"{output_dir}/{model}_{ssp}_{year}_regridded.zarr"

    s3_store = s3fs.S3Map(root=output_file, s3=s3, check=False)
    zarr_store = zarr.storage.KVStore(s3_store)
    regridded_data_da.to_zarr(store=zarr_store, mode="w", consolidated=True)

    logging.info(f"Saved regridded data for {year}, model {model}, SSP {ssp} to {output_file}")
    return regridded_data_da


def run_downscaling_workflow(
    shapefile,
    aorc_path_template,
    nasa_historical_path_template,
    future_path_template,
    historical_years,
    future_years,
    models,
    ssps,
    buffered_bounds,
    doys,
    s3,
    output_dir,
):
    import dask
    """
    Orchestrates the entire downscaling workflow, including processing historical and future data, fitting and applying quantile mpas, compiling year data, regridding, and saving results.

    Parameters:
        shapefile (GeoDataFrame): Shapefile of the region of interest.
        aorc_path_template (str): Template path for AORC data on S3.
        nasa_historical_path_template (str): Template path for NASA historical data on S3.
        future_path_template (str): Template path for NASA future data on S3.
        historical_years (list): List of historical years to process.
        future_years (list): List of future years to process.
        models (list): List of model names for NASA data.
        ssps (list): List of Shared Socioeconomic Pathways (SSPs) for future data.
        buffered_bounds (GeoDataFrame): Buffered bounds of the region of interest.
        doys (list): List of days of the year to process.
        s3 (S3FileSystem): S3FileSystem object for reading and writing data to S3.
        output_dir (str): Output directory for saving regridded data.
    """

    # Phase 1: Process historical data
    logging.info("Phase 1: Processing historical data")
    historical_tasks = [
        {
            "task": process_historical_data(
                aorc_path_template,
                nasa_historical_path_template,
                shape=shapefile,
                buffered_bounds=buffered_bounds,
                aorc_variable_name="APCP_surface",
                nasa_variable_name="pr",
                start_year=year,
                end_year=year,
                s3=s3,
                model=model,
                doys=[doy],
            ),
            "doy": doy,
        }
        for year in historical_years
        for doy in doys
        for model in models
    ]

    # Extract original AORC grid from the first historical task
    original_aorc_grid = historical_tasks[0]["task"][2]

    # Phase 1: Process future data (runs concurrently)
    logging.info("Phase 1: Processing future data")
    future_tasks = [
        {
            "task": process_future_data(
                future_path_template,
                buffered_bounds,
                shape=shapefile,
                nasa_variable_name="pr",
                quantile_mappers=None,
                aorc_combined=None,
                original_aorc_grid=None,
                start_year=year,
                end_year=year,
                output_dir=output_dir,
                s3=s3,
                model=model,
                ssp=ssp,
                doys=[doy],
            ),
            "doy": doy,
        }
        for year in future_years
        for doy in doys
        for model in models
        for ssp in ssps
    ]

    # Phase 2: Quantile Mapping
    logging.info("Phase 2: Quantile mapping")
    transformed_tasks = [
        {
            "task": fit_and_apply_quantile_map(
                aorc_data=historical_task["task"][1],  # aorc_combined from historical task
                nasa_data=historical_task["task"][0],  # nasa_combined from historical task
                nasa_future_data=future_task["task"],  # nasa_future from future task
                doys=[doy],
            ),
            "year": year,
        }
        for doy in doys
        for year in future_years
        for historical_task in historical_tasks
        for future_task in future_tasks
        if historical_task["doy"] == doy and future_task["doy"] == doy
    ]

    # Phase 3: Year-wise compilation and regridding
    logging.info("Phase 3: Regridding and saving")
    regridded_tasks = [
        regrid_and_save(
            compiled_year_data=compile_year_data([task["task"] for task in transformed_tasks if task["year"] == year]),
            original_aorc_grid=original_aorc_grid,
            output_dir=output_dir,
            year=year,
            model=model,
            ssp=ssp,
            buffered_bounds=buffered_bounds,
        )
        for year in future_years
        for model in models
        for ssp in ssps
    ]

    # Visualize workflow
    logging.info("Generating Dask workflow visualization")
    # all_tasks = (
    #     [task["task"] for task in historical_tasks]
    #     + [task["task"] for task in future_tasks]
    #     + [task["task"] for task in transformed_tasks]
    #     + regridded_tasks
    # )
    # local_viz_path = "downscaling_workflow.png"
    # dask.visualize(all_tasks, filename=local_viz_path, format="png")

    # Trigger computation
    logging.info("Starting computation of downscaling workflow")
    dask.compute(*regridded_tasks)

SUPPRESS_LOGS = ["boto3", "botocore", "geopandas", "fiona", "rasterio", "pyogrio", "xarray", "shapely"]


def initialize_logger(json_logging: bool = False, level: int = logging.INFO):
    datefmt = "%Y-%m-%dT%H:%M:%SZ"
    if json_logging:
        for module in SUPPRESS_LOGS:
            logging.getLogger(module).setLevel(logging.WARNING)

        class FlushStreamHandler(logging.StreamHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        handler = FlushStreamHandler(sys.stdout)

        logging.basicConfig(
            level=level,
            handlers=[handler],
            format="""{"time": "%(asctime)s" , "level": "%(levelname)s", "msg": "%(message)s"}""",
            datefmt=datefmt,
        )
    else:
        for package in SUPPRESS_LOGS:
            logging.getLogger(package).setLevel(logging.ERROR)
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt=datefmt)
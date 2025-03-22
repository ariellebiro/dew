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

"""

import logging
import os
from datetime import datetime, timedelta
from shapely.geometry import box
import cftime
import fsspec
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import s3fs
import xarray as xr
import zarr
from dask import delayed
from dotenv import load_dotenv
from pyresample import geometry, kd_tree
from skdownscale.pointwise_models import QuantileMapper
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=UserWarning, module = "distributed.client")
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
SUPPRESS_LOGS = ["boto3", "botocore", "geopandas", "fiona", "rasterio", "pyogrio", "xarray", "shapely", "zarr", "dask", "distributed"]

AORC_PATH_TEMPLATE = "noaa-nws-aorc-v1-1-1km/{year}.zarr"
NASA_HISTORICAL_PATH_TEMPLATE = (
    "s3://hydromet/nex-gddp-cmip6/NEX-GDDP-CMIP6/{model}/historical/r4i1p1f1/pr/pr_day_{model}_historical_r4i1p1f1_gn_{year}.json"
)
NASA_FUTURE_PATH_TEMPLATE = (
    "s3://hydromet/nex-gddp-cmip6/NEX-GDDP-CMIP6/{model}/{ssp}/r4i1p1f1/pr/pr_day_{model}_{ssp}_r4i1p1f1_gn_{year}.json"
)

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


def init(aoi_file: str, log_level: int = logging.INFO, buffer: int = 0.1):
    initialize_logger()
    logging.info("Initialized environment variables")

    # Read shapefile and create buffered bounds
    aoi_gdf = gpd.read_file(aoi_file)
    # projected_shape = aoi_gdf.to_crs(epsg=4326)  # Replace with a projected CRS for your region
    # Convert buffered bounds to a bounding box
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    bounding_box = box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
    buffered_bounds = gpd.GeoDataFrame({"geometry": [bounding_box]}, crs=aoi_gdf.crs)

    return aoi_gdf, buffered_bounds

#############
###PHASE 1###
# reading, organizing, and sorting data by DOY
# separate functions for each dataset
@delayed
def process_historical_data(
    year,
    aorc_path_template,
    nasa_historical_path_template,
    aoi_gdf,
    buffered_bounds,
    aorc_variable_name,
    nasa_variable_name,
    s3_private,
    s3_public,
    model,
    doys,
):
    """
    Processes historical data by reading, organizing, and sorting data by the day of year (DOY).
    This includes loading AORC and NASA historical data, clipping to the buffered_bounds of Kanawha, organizing by daily values, and regridding AORC data to match the NASA grid for quantile mapping.

    Parameters:
        aorc_path_template (str): Template path for AORC data on S3.
        nasa_historical_path_template (str): Template path for NASA historical data on S3.
        aoi_gdf (GeoDataFrame): aoi_gdf of the region of interest.
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
    initialize_logger()
    nasa_data_list = []
    aorc_data_list = []

    # Initialize original AORC grid reference (to regrid future data)
    original_aorc_grid = None

    logging.info(f"Processing historical data for model: {model}, year: {year} (DOYS {sorted(doys)})")

    #determine if it's a leap year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    yearly_aorc_list=[]

    for doy in doys:
        #compute time range for this DOY
        start_time = datetime(year, 1, 1) + timedelta(days=doy - 1)
        end_time = start_time + timedelta(days=1) - timedelta(seconds=1)

        try:
            # Load AORC data
            aorc_data = xr.open_zarr(
                s3_public.get_mapper(aorc_path_template.format(year=year)), chunks={"time": "auto"}
            )[aorc_variable_name].sel(time=slice(start_time, end_time))

            # Rename lat/lon for consistency
            if "latitude" in aorc_data.coords and "longitude" in aorc_data.coords:
                aorc_data = aorc_data.rename({"latitude": "lat", "longitude": "lon"})

            aorc_data = aorc_data.rio.write_crs("EPSG:4326", inplace=True)
            aorc_data = aorc_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

            aorc_clipped = aorc_data.rio.clip(buffered_bounds.geometry, buffered_bounds.crs)

            if aorc_clipped.size == 0:
                logging.warning(f"AORC data for DOY {doy} in year {year} is empty after clipping.")
                continue  # Skip empty data

            # Resample to daily
            aorc_daily = aorc_clipped.resample(time="1D").sum()

            #concatenate per DOY, only at the end)
            yearly_aorc_list.append(aorc_daily.chunk({'time': 1}))

            # Save the AORC grid once for reference
            if original_aorc_grid is None:
                original_aorc_grid = aorc_clipped

        except Exception as e:
            logging.error(f"Error processing AORC data for DOY {doy} in year {year}: {e}")
            continue

    #concatenate all AORC data after collecting full year
    if yearly_aorc_list:
        aorc_data_list = xr.concat(yearly_aorc_list, dim="time")

    try:
        with s3_private.open(nasa_historical_path_template.format(model=model, year=year), mode="r") as f:
            nasa_data_refs = json.load(f)

        fs = fsspec.filesystem(
            "reference",
            fo=nasa_data_refs,
            target_options={"anon": True},
            remote_protocol="s3",
            remote_options={"anon": True}
        )
        fs_mapper = fs.get_mapper()
        nasa_data = xr.open_zarr(fs_mapper, consolidated=False)[nasa_variable_name]

        nasa_data = nasa_data.sel(time=nasa_data.time.dt.dayofyear.isin(doys))
        nasa_data = nasa_data.assign_coords(lon=(((nasa_data.lon + 180) % 360) - 180)).sortby("lon")
        nasa_data = nasa_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        nasa_data = nasa_data.rio.write_crs("EPSG:4326", inplace=True)
        nasa_clipped = nasa_data.rio.clip(buffered_bounds.geometry, aoi_gdf.crs)
        nasa_daily = nasa_clipped * 86400
        nasa_daily.attrs["units"] = "kg/m^2"

        # Convert time to datetime64 within the function itself
        if isinstance(nasa_daily["time"].values[0], cftime.datetime):
            logging.debug(f"Converting time to datetime64 for {nasa_variable_name} in year {year}")
            nasa_daily["time"] = pd.to_datetime([t.strftime("%Y-%m-%d") for t in nasa_daily["time"].values])

        nasa_data_list.append(nasa_daily.chunk({'time':1}))

        logging.info(f"Processed NASA historical data for year {year} (DOYS: {sorted(doys)})")
    except Exception as e:
        logging.error(f"Error processing NASA data for year {year}: {e}")

    #concatenate all AORC and NASA after processing all years
    nasa_combined = xr.concat(nasa_data_list, dim="time")
    aorc_combined = xr.concat(aorc_data_list, dim="time")

    #regrid AORC data after concat
    try: 
        regridded_aorc_data = []

        # Define Pyresample source and target geometries
        source_lons = aorc_combined["lon"].values
        source_lats = aorc_combined["lat"].values
        target_lons = nasa_combined["lon"].values
        target_lats = nasa_combined["lat"].values

        if source_lons.ndim == 1 and source_lats.ndim == 1:
            source_lons, source_lats = np.meshgrid(source_lons, source_lats)
        if target_lons.ndim == 1 and target_lats.ndim == 1:
            target_lons, target_lats = np.meshgrid(target_lons, target_lats)

        source_grid = geometry.SwathDefinition(lons=source_lons, lats=source_lats)
        target_grid = geometry.SwathDefinition(lons=target_lons, lats=target_lats)

        # Resample AORC data to match NASA grid using Pyresample
        for time_step in aorc_combined["time"]:
            time_slice = aorc_combined.sel(time=time_step).values
            regridded_slice = kd_tree.resample_nearest(
                source_grid,
                time_slice,
                target_grid,
                radius_of_influence=25000,  # Adjust based on your spatial resolution needs
                fill_value=np.nan,
            )
            regridded_aorc_data.append(regridded_slice)

        # Stack regridded slices into a DataArray
        aorc_combined = xr.DataArray(
            np.stack(regridded_aorc_data, axis=0),
            dims=("time", "lat", "lon"),
            coords={"time": aorc_combined["time"], "lat": nasa_combined["lat"], "lon": nasa_combined["lon"]},
            attrs=aorc_combined.attrs,
        )
        
    except Exception as e:
        logging.error(f"Error regridding AORC data: {e}")

    # remove feb 29 from aorc and shift DOYS
    aorc_combined = aorc_combined.sel(time=~((aorc_combined["time"].dt.month == 2) & (aorc_combined["time"].dt.day == 29)))

    aorc_combined = aorc_combined.reindex(time=nasa_combined.time, method="nearest")

    logging.info(f"Completed processing historical data for model: {model}, year {year}")
    return nasa_combined, aorc_combined, original_aorc_grid



@delayed
def process_future_data(
    nasa_future_path_template,
    buffered_bounds,
    aoi_gdf,
    nasa_variable_name,
    quantile_mappers,
    aorc_combined,
    original_aorc_grid,
    start_year,
    end_year,
    output_dir,
    s3_private,
    s3_public,
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
        aoi_gdf (GeoDataFrame): aoi_gdf of the region of interest.
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
    initialize_logger()
    logging.info(f"Processing future data for model: {model}, SSP: {ssp}, years {start_year}-{end_year}")

    nasa_future_list = []

    for year in range(start_year, end_year + 1):
        for doy in doys:
            date = datetime(year, 1, 1) + pd.Timedelta(days=doy - 1)

        # Load NASA future data
        kerchunk_json_path = nasa_future_path_template.format(model=model, ssp=ssp, year=year)
        try:
            with s3_private.open(kerchunk_json_path, mode="r") as f:
                nasa_data_refs = json.load(f)
        except Exception as e:
            logging.error(f"Error loading NASA future data for year {year}: {e}")
            continue

        try:
            fs = fsspec.filesystem(
                "reference", 
                fo=nasa_data_refs, 
                target_options={"anon": True}, 
                remote_protocol="s3", 
                remote_options={"anon": True}
                )
            fs_mapper = fs.get_mapper()
            nasa_future = xr.open_zarr(fs_mapper, consolidated=False)[nasa_variable_name]
        except Exception as e:
            logging.error(f"Failed to load Kerchunk JSON for {year}: {e}")
            nasa_future = None

        nasa_future = nasa_future.sel(time=nasa_future.time.dt.dayofyear.isin(doys))
        nasa_future = nasa_future.assign_coords(lon=(((nasa_future.lon + 180) % 360) - 180)).sortby("lon")
        nasa_future = nasa_future.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        nasa_future = nasa_future.rio.write_crs("EPSG:4326", inplace=True)
        nasa_future_clipped = nasa_future.rio.clip(buffered_bounds.geometry, aoi_gdf.crs)
        nasa_future_daily = nasa_future_clipped * 86400
        nasa_future_daily.attrs["units"] = "kg/m^2"

        # Convert time to datetime64 directly in the function
        if isinstance(nasa_future_daily["time"].values[0], cftime.datetime):
            logging.debug(f"Converting time to datetime64 for {nasa_variable_name} in year {year}")
            nasa_future_daily["time"] = pd.to_datetime(
                [t.strftime("%Y-%m-%d") for t in nasa_future_daily["time"].values]
            )

        # Sort and filter future data by DOY
        nasa_future_daily = nasa_future_daily.sortby("time")
        nasa_future_daily = nasa_future_daily.sel(time=nasa_future_daily.time.dt.dayofyear.isin(doys))

        nasa_future_list.append(nasa_future_daily)
        nasa_future = xr.concat(nasa_future_list, dim="time")

        # You can now process future data here, like fitting quantile mapping, etc
        logging.info(f"Processed NASA future data for year {year} (DOYS: {sorted(doys)})")
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
    initialize_logger()
    logging.info(f"Fitting and applying quantile mapping for {len(doys)} DOYs")

    quantile_mappers = {}

    nasa_doys = set(nasa_data.time.dt.dayofyear.values)
    aorc_doys = set(aorc_data.time.dt.dayofyear.values)
    if not nasa_doys:
        logging.error("NASA data has no valid DOYS")
    if not aorc_doys:
        logging.error("AORC data has no valid DOYS")
    logging.info(f"NASA DOYs available: {sorted(nasa_doys)}")
    logging.info(f"AORC DOYs available: {sorted(aorc_doys)}")

    if nasa_doys != aorc_doys:
        logging.error(f"DOYs in NASA and AORC data do not match: NASA {sorted(nasa_doys)} vs AORC {sorted(aorc_doys)} ")

    # Fit quantile mappers by DOY for historical data
    for doy, nasa_group in nasa_data.groupby("time.dayofyear"):
        if doy not in aorc_data.groupby("time.dayofyear").groups:
            logging.warning(f"Skipping DOY {doy} as it is missing in AORC data.")
            # Log available DOYs in historical and future data
            logging.info(f"Available DOYs in NASA historical: {nasa_data.time.dt.dayofyear.values}")
            logging.info(f"Available DOYs in AORC historical: {aorc_data.time.dt.dayofyear.values}")
            logging.info(f"Available DOYs in NASA future: {nasa_future_data.time.dt.dayofyear.values}")
            continue
        
        matching_aorc = aorc_data.sel(time=nasa_group.time)
        if matching_aorc.time.size != nasa_group.time.size:
            logging.warning(f"Mismatch in time steps for DOY {doy}. skipping")
            continue 

        # Flatten and mask out NaN values
        nasa_flat = nasa_group.values.flatten().reshape(-1, 1)
        aorc_flat = matching_aorc.values.flatten().reshape(-1, 1)
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
    initialize_logger()
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
def regrid_and_save(compiled_year_data, original_aorc_grid, output_dir, year, model, ssp, buffered_bounds, s3_private,
                    overwrite=True): #set overwrite to true or false depending on if you want to overwrite the data here
    initialize_logger()
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

    #extract existing time information
    existing_years = compiled_year_data["time"].dt.year.values.astype(str)
    existing_doys = compiled_year_data["time"].dt.dayofyear.values.astype(str)

    # loop through time steps and regrid each slice
    for time_step in compiled_year_data["time"]:
        #extract doy for current time step
        doy = time_step.dt.dayofyear.values.item()
        # select data for current time step
        time_slice = compiled_year_data.sel(time=time_step).values
        source_lons = compiled_year_data["lon"].values
        source_lats = compiled_year_data["lat"].values
        if source_lons.ndim == 1 and source_lats.ndim == 1:
            source_lons, source_lats = np.meshgrid(source_lons, source_lats)
        source_grid = geometry.SwathDefinition(lons=source_lons, lats=source_lats)

        # print time only
        time_str = pd.to_datetime(str(time_step.values)).strftime("%Y-%m-%d")
        logging.info(f"Regridding data for {year}, model {model}, SSP {ssp}, time {time_str}")
        logging.debug(f"Targeting source: {source_grid.lons.shape}, {source_grid.lats.shape}")
        if min(source_grid.lons.shape) < 8 or min(source_grid.lats.shape) < 8:
            raise ValueError("Source grid is too small for regridding. Select a larger region or increase buffer size")

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

    s3_store = s3fs.S3Map(root=output_file, s3=s3_private, check=False)
    zarr_store = zarr.storage.KVStore(s3_store)

    try:
        with xr.open_zarr(s3_store) as ds:
            existing_years = ds["time"].dt.year.values.astype(str)
            existing_doys = ds["time"].dt.dayofyear.values.astype(str)

            #check if both year and DOY exist in dataset
            if str(year) in existing_years and str(doy) in existing_doys:
                exists = True
            else:
                exists = False
    except (FileNotFoundError, zarr.errors.GroupNotFoundError, KeyError):
        exists = False

    if exists and overwrite:
        # Handle all cases
        logging.info(f"Overwriting existing data for {year}, DOYs {doy}, model {model}, SSP {ssp} in {output_file}")
        regridded_data_da.to_zarr(store=zarr_store, mode="w", consolidated=True)
    elif exists and not overwrite:
        logging.info(f"Skipping existing data for {year}, model {model}, SSP {ssp} to existing {output_file}, overwrite=False")
    else:
        regridded_data_da.to_zarr(store=zarr_store, mode="w", consolidated=True)
        logging.info(f"Saved regridded data for {year}, model {model}, SSP {ssp} to new file {output_file}")

    return regridded_data_da




def run_downscaling_workflow(
    aoi_gdf,
    aorc_path_template,
    nasa_historical_path_template,
    future_path_template,
    historical_years,
    future_years,
    models,
    ssps,
    buffered_bounds,
    doys,
    output_dir,
    s3_private,
    s3_public,
):
    from dask import delayed
    import logging
    import xarray as xr

    initialize_logger()
    logging.info("Phase 1: Processing historical data")

    all_tasks = []

    for model in models:
        for ssp in ssps:
            logging.info(f"Preparing tasks for model {model}, SSP {ssp}")

            # Step 1: Process historical data in parallel by year
            historical_outputs = []
            for year in historical_years:
                hist = process_historical_data(
                    year=year,
                    aorc_path_template=aorc_path_template,
                    nasa_historical_path_template=nasa_historical_path_template,
                    aoi_gdf=aoi_gdf,
                    buffered_bounds=buffered_bounds,
                    aorc_variable_name="APCP_surface",
                    nasa_variable_name="pr",
                    s3_private=s3_private,
                    s3_public=s3_public,
                    model=model,
                    doys=doys,
                )
                historical_outputs.append(hist)

            # Step 2: Combine historical outputs
            @delayed
            def combine_histories(hist_outputs):
                nasa_all = xr.concat([h[0] for h in hist_outputs], dim="time")
                aorc_all = xr.concat([h[1] for h in hist_outputs], dim="time")
                aorc_grid = hist_outputs[0][2]  # use grid from any one year
                return nasa_all, aorc_all, aorc_grid

            combined = combine_histories(historical_outputs)

            # Step 3: Loop over future years
            for year in future_years:
                logging.info(f"Queueing tasks for future year: {year}")

                # Step 3.1: Process future data for the year
                nasa_future = process_future_data(
                    nasa_future_path_template=future_path_template,
                    buffered_bounds=buffered_bounds,
                    aoi_gdf=aoi_gdf,
                    nasa_variable_name="pr",
                    quantile_mappers=None,
                    aorc_combined=combined[1],
                    original_aorc_grid=combined[2],
                    start_year=year,
                    end_year=year,
                    output_dir=output_dir,
                    s3_private=s3_private,
                    s3_public=s3_public,
                    model=model,
                    ssp=ssp,
                    doys=doys,
                )

                # Step 3.2: Apply quantile mapping to the future data
                mapped = fit_and_apply_quantile_map(
                    aorc_data=combined[1],
                    nasa_data=combined[0],
                    nasa_future_data=nasa_future,
                    doys=doys,
                )

                # Step 3.3: Compile and regrid the mapped data
                compiled = compile_year_data([mapped])
                regridded = regrid_and_save(
                    compiled_year_data=compiled,
                    original_aorc_grid=combined[2],
                    output_dir=output_dir,
                    year=year,
                    model=model,
                    ssp=ssp,
                    buffered_bounds=buffered_bounds,
                    s3_private=s3_private,
                    overwrite=True,
                )

                all_tasks.append(regridded)

    return all_tasks















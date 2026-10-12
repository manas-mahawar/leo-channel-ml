import xarray as xr

FILE = "data/raw/athens_era5_test.nc"

print("Opening ERA5 file...")

ds = xr.open_dataset(FILE, engine="netcdf4")

print("\n========== DATASET ==========")
print(ds)

print("\n========== VARIABLES ==========")

for variable in ds.data_vars:
    data = ds[variable]

    print(f"\nVariable: {variable}")
    print(f"Dimensions: {data.dims}")
    print(f"Shape: {data.shape}")
    print(f"Units: {data.attrs.get('units', 'N/A')}")
    print(f"Long name: {data.attrs.get('long_name', 'N/A')}")

print("\n========== FIRST VALUES ==========")

for variable in ds.data_vars:
    print(f"\n{variable}:")
    print(ds[variable].values.flatten()[:10])

ds.close()
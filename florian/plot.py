from scipy.io import netcdf
#import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import geopandas


f1 = netcdf.netcdf_file("tas_Amon_IPSL-CM5A-MR_rcp45_r1i1p1_200601-210012.nc")

lon = np.array(f1.variables['lon'][:])
lat = np.array(f1.variables['lat'][:])
tas = np.array(f1.variables['tas'][:])
time = np.array(f1.variables['time'][:])

print(time.shape,tas.shape,lon.shape,lat.shape)
#


weights = np.cos(np.deg2rad(lat))
weights = weights/weights.sum()
weights = weights[np.newaxis, :, np.newaxis] #* np.ones_like(ano)
plt.figure()

fig = plt.figure(figsize=(8,5))
#ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
ax = plt.subplot(1,1,1)




#assert 0

tas0 = tas[0]
print(tas0)

mask = lon >= 180
print(mask)
lon[mask] -= 360

arr1inds = lon.argsort()

lon = lon[arr1inds]

#ax.set_global()
tas0 = tas0[:,arr1inds]
lon = np.concatenate((lon,np.array([180.])))
tas0 = np.concatenate((tas0,tas0[:,0,None]),axis=-1)
print(lon)
ax.set_aspect('equal')
#ax.set_title(f"{title}")
world.plot(ax=ax, color='lightblue', edgecolor='black')


lo, la = np.meshgrid(lon, lat)
print(lo, la, tas[0])
print(lo.shape, la.shape, tas0.shape)

#layer = ax.pcolormesh(lo, la, tas[0], transform=ccrs.PlateCarree(), cmap="plasma")
layer = ax.pcolormesh(lo, la, tas0, cmap="plasma",alpha = 0.5)
cbar = fig.colorbar(layer, orientation="horizontal") 
cbar.set_label("Temperature (K)")
ax.set_title(time[0])
# ax.gridlines()
#ax.coastlines()
plt.show()

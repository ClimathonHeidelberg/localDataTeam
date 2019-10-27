from scipy.io import netcdf
#import cartopy.crs as ccrs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import geopandas
import pandas as pd



f1 = netcdf.netcdf_file("tas_Amon_IPSL-CM5A-MR_rcp45_r1i1p1_200601-210012.nc")

lon = np.array(f1.variables['lon'][:])
lat = np.array(f1.variables['lat'][:])
tas = np.array(f1.variables['tas'][:])-273
time = np.array(f1.variables['time'][:])

#df = pd.DataFrame({'time': time, 'tas':tas})
#print(df)
#tempf = pd.DataFrame(tas)
#df = df.append(tempf)
#print(1)
#assert 0
print(time.shape,tas.shape,lon.shape,lat.shape)
#

print(lon[4],lat[110])
print(time)

import datetime
def serial_date_to_date(srl_no):
    """https://stackoverflow.com/questions/39986041/converting-days-since-epoch-to-date/39988256
    
    Arguments:
        srl_no {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    new_date = np.datetime64('2006-01-01') + np.timedelta64(srl_no,'D')
    #year = new_date/np.timedelta64(1,'Y')
    return new_date#, year#.strftime("%Y-%m-%d")

print(serial_date_to_date(1))
print(np.array(time,dtype=int))
time = np.array(time,dtype=np.int32)
time2 = time.copy()
time2 = np.array(time,dtype=datetime.date)
year = np.array(time,dtype=datetime.date)

#year = time2/np.timedelta64(1,'Y')
print(time2)
for i in range(len(time)):
    time2[i] = serial_date_to_date(time[i])
    year[i] = int(str(time2[i])[:4])

year_numbers = np.unique(year)
print(year)
print(year_numbers)
tas_list = []
for year_numb in year_numbers:
    mask = (year == year_numb)
    print("number of same years",np.sum(mask))
    year_tas = np.average(tas[mask,:,:],axis=0)
    tas_list.append(year_tas)
    #print(year_numb,year_tas)
tases = np.array(tas_list)
print(tases.shape,year_numbers.shape)
print(time2[0])
print(int(str(time2[0])[:4]))
#assert 0
#print(np.timedelta64(np.int32([2]), 'D'))
print(type(time),type(time[0]))
print(time2)
print(year)
#print(serial_date_to_date(time))
#print(np.array(time.timedelta64()))
#print(serial_date_to_date(time.tolist()))
#assert 0
plt.figure()

plt.plot(year_numbers,tases[:,110,4])
#plt.show(block=True)

weights = np.cos(np.deg2rad(lat))
weights = weights/weights.sum()
weights = weights[np.newaxis, :, np.newaxis] #* np.ones_like(ano)


tas0 = tas[0]
print(tas0)

mask = lon >= 180
print(mask)
lon[mask] -= 360

arr1inds = lon.argsort()

lon = lon[arr1inds]

#ax.set_global()
tas = tas[:,:,arr1inds]
lon = np.concatenate((lon,np.array([180.])))
tas = np.concatenate((tas,tas[:,:,0,None]),axis=-1)
print(lon)
lo, la = np.meshgrid(lon, lat)
print(lo, la, tas[0])
print(lo.shape, la.shape, tas.shape)








fig = plt.figure(figsize=(10,6))
#ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
ax = plt.subplot(1,1,1)




#assert 0
#levels = np.percentile(value[mask], np.linspace(0,100,101))
#norm = matplotlib.colors.BoundaryNorm(levels,256)
norm = matplotlib.colors.Normalize(tases.min(),tases.max())

ax.set_aspect('equal')
#ax.set_title(f"{title}")
world.plot(ax=ax, color='lightblue', edgecolor='black')

#layer = ax.pcolormesh(lo, la, tas[0], transform=ccrs.PlateCarree(), cmap="plasma")
layer = ax.pcolormesh(lo, la, tas[0]*np.nan, cmap="plasma",norm=norm,alpha = 0.5)
cbar = fig.colorbar(layer,norm=norm, orientation="horizontal")
cbar.set_label("Temperature (K)")
ax.set_title(time2[0])
# ax.gridlines()
#ax.coastlines()
#plt.show()

def animate(i):
    #plt.figure()
    #ax = plt.subplot(1,1,1)
    #layer = ax.pcolormesh(lo, la, tas[i], cmap="plasma",alpha = 0.5)
    ax.clear()
    world.plot(ax=ax, color='lightblue', edgecolor='black')

    plt.pcolormesh(lo, la, tases[i], cmap="plasma",alpha = 0.5)
    
    #norm = matplotlib.colors.Normalize(vmin=cm.min,vmax=cm.max)
    #plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax)
    #cbar = fig.colorbar(layer, orientation="horizontal")
    ax.set_title(year_numbers[i])
    #print(tases.shape,year_numbers.shape)

    print(tas[i])
    #layer.set_array(tas[i].ravel())
    return layer,
#ax.pcolormesh(lo, la, tas[50], cmap="plasma",alpha = 0.5)
#layer.set_array(tas[20])
import matplotlib.animation as animation
anim = animation.FuncAnimation(fig, animate,
                               frames=200, interval=20, blit=True)

plt.show()
#anim.save('sine_wave.gif', writer='ffmpeg')
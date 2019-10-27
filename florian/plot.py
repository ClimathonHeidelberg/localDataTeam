from scipy.io import netcdf
#import cartopy.crs as ccrs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import geopandas
import pandas as pd
import datetime
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy.polynomial.polynomial as poly


filenames = ["tas_Amon_IPSL-CM5A-MR_rcp26_r1i1p1_200601-210012.nc","tas_Amon_IPSL-CM5A-MR_rcp45_r1i1p1_200601-210012.nc","d401c16d-f083-415d-8f53-d6a017cc9cd2-tas_Amon_IPSL-CM5A-MR_rcp60_r1i1p1_200601-210012.nc","tas_Amon_IPSL-CM5A-MR_rcp85_r1i1p1_200601-210012.nc"]
names = ["rcp26","rcp45","rcp60","rcp85"]

def extract(filename=filenames[1]):
    f1 = netcdf.netcdf_file(filename)

    lon = np.array(f1.variables['lon'][:])
    lat = np.array(f1.variables['lat'][:])
    tas = np.array(f1.variables['tas'][:])-273
    time = np.array(f1.variables['time'][:])
    return lon,lat,tas,time



#df = pd.DataFrame({'time': time, 'tas':tas})
#print(df)
#tempf = pd.DataFrame(tas)
#df = df.append(tempf)
#print(1)
#assert 0
#print(time.shape,tas.shape,lon.shape,lat.shape)
#

#print(lon[4],lat[110])

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



def extract_years(time,tas):
    time = np.array(time,dtype=np.int32)
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
    return year, year_numbers, tases, time2

#print(tases.shape,year_numbers.shape)
#print(time2[0])
#print(int(str(time2[0])[:4]))
#assert 0
#print(np.timedelta64(np.int32([2]), 'D'))
#print(type(time),type(time[0]))
#print(time2)
#print(year)
#print(serial_date_to_date(time))
#print(np.array(time.timedelta64()))
#print(serial_date_to_date(time.tolist()))
#assert 0
_,_,tas_s,time_s,year_s, year_numbers_s, tases_s, time2_s = [],[],[],[],[],[],[],[]
for i in range(len(names)):
    lon,lat,tas,time = extract(filename=filenames[i])
    year, year_numbers, tases, time2 = extract_years(time,tas)
    tas_s.append(tas)
    time_s.append(time)
    year_s.append(year)
    year_numbers_s.append(year_numbers)
    tases_s.append(tases)
    time2_s.append(time2)

weights = np.cos(np.deg2rad(lat))
weights = weights/weights.sum()
weights = weights[np.newaxis, :, None] * np.ones_like(tases)
fig_trend = plt.figure(figsize=(10,5))
ax1 = plt.subplot(1,1,1)

def show_trends(i):
    #plt.figure(names[i])

    #plt.show(block=True)

    ax1.clear()
    ax1.set_title(names[i])

    ax1.plot(year_numbers_s[i],tases_s[i][:,110,4],label = "Heidelberg")

    ax1.plot(year_numbers_s[i],np.average(tases_s[i][:,:,:],weights=weights,axis=(1,2)),label="Global")
    x = np.array(year_numbers_s[i], dtype='float')
    y = tases_s[i][:,110,4]
    A = np.vstack([x, np.ones(len(x))]).T
    print(x.shape,y.shape,A.shape)
    print(y)
    print(A)
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]    #hd_fit=poly.polyfit(year_numbers,tases[:,110,4],1)
    #global_fit=poly.polyfit(year_numbers, np.average(tases[:,:,:],weights=weights,axis=(1,2)),1)
    #hd_fit = poly.polyval(year_numbers, hd_fit)
    #global_fit = poly.polyval(year_numbers, global_fit)
    ax1.plot(x, m*x + c, 'r', label='Durchschnittlicher Verlauf')
    #plt.plot(year_numbers,hd_fit,label = "Heidelberg")
    #plt.plot(year_numbers,global_fit,label = "Global")
    fig_trend.legend()
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Average Temperature in °C")
    


#for i in range(len(names)):
#    lon,lat,tas,time = extract(filename=filenames[i])
#    year, year_numbers, tases, time2 = extract_years(time,tas)
#    show_trends(year_numbers,tases,lat,name=names[i])
#def show_interactive_trends():
    
show_trends(0)

color_radios_ax = fig_trend.add_axes([0.9, 0.5, 0.2, 0.15])
color_radios_trend = RadioButtons(color_radios_ax, ("rcp26","rcp45","rcp60","rcp85"), active=0)
def color_radios_on_clicked(label):
    i = names.index(label)
    show_trends(i)
color_radios_trend.on_clicked(color_radios_on_clicked)
#plt.show()
#plt.show()

#show_interactive_trends()
#plt.show()
#tases = tases - tases[None,0]
mask = lon >= 180
#print(mask)
lon[mask] -= 360

arr1inds = lon.argsort()

lon = lon[arr1inds]

#ax.set_global()
lon = np.concatenate((lon,np.array([180.])))
for j in range(len(tas_s)):
    tas_s[i] = tas_s[i][:,:,arr1inds]
    tas_s[i] = np.concatenate((tas_s[i],tas_s[i][:,:,0,None]),axis=-1)
#print(lon)
lo, la = np.meshgrid(lon, lat)
#print(lo, la, tas[0])
#print(lo.shape, la.shape, tas.shape)








fig = plt.figure(figsize=(10,6))
#ax = fig.add_subplot(1,1,1, projection=ccrs.Robinson())
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
ax = plt.subplot(1,1,1)




#assert 0
#levels = np.percentile(value[mask], np.linspace(0,100,101))
#norm = matplotlib.colors.BoundaryNorm(levels,256)
#max_tas = max(abs(tases.min()),abs(tases.max()))
#norm = matplotlib.colors.Normalize(-max_tas,max_tas)
norm = matplotlib.colors.Normalize((np.array(tases_s)-tases_s[0]).min(),(np.array(tases_s)-tases_s[0]).max())

#print((np.array(tases_s)-tases_s[0]).min(),(np.array(tases_s)-tases_s[0]).max())
#assert 0
ax.set_aspect('equal')
#ax.set_title(f"{title}")
world.plot(ax=ax, color='lightblue', edgecolor='black')
freq_slider_ax = fig.add_axes([0.85, 0.3, 0.03, 0.65])#, axisbg=axis_color)
freq_slider = Slider(freq_slider_ax, 'Jahr', 0, 95.0, valinit=0,orientation="vertical")

color_radios_ax = fig.add_axes([0.9, 0.5, 0.2, 0.15])
color_radios = RadioButtons(color_radios_ax, ("rcp26","rcp45","rcp60","rcp85"), active=0)

#layer = ax.pcolormesh(lo, la, tas[0], transform=ccrs.PlateCarree(), cmap="plasma")
layer = ax.pcolormesh(lo, la, tas[0]*np.nan, cmap="plasma",norm=norm,alpha = 0.5)
cbar = fig.colorbar(layer,norm=norm, ax=ax, orientation="horizontal")
cbar.set_label("Änderung in Temperatur in °C seit 2006")
ax.set_title(time2[0])
# ax.gridlines()
#ax.coastlines()
#plt.show()
selected_year = 0
selected_model = 0
def animate(i):
    global selected_year
    selected_year = i
    #plt.figure()
    #ax = plt.subplot(1,1,1)
    #layer = ax.pcolormesh(lo, la, tas[i], cmap="plasma",alpha = 0.5)
    ax.clear()
    world.plot(ax=ax, color='lightblue', edgecolor='black')

    ax.pcolormesh(lo, la, tases_s[selected_model][i]-tases_s[selected_model][0], norm=norm,cmap="plasma",alpha = 0.5)
    print(selected_model,i)
    #norm = matplotlib.colors.Normalize(vmin=cm.min,vmax=cm.max)
    #plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax)
    #cbar = fig.colorbar(layer, orientation="horizontal")
    ave_temp = np.average(tases_s[selected_model][i],weights=weights[i])

    ax.set_title(f"{names[selected_model]} Jahr {year_numbers[i]}, Globale Temp: {ave_temp:.2f} °C und in Heidelberg {tases_s[selected_model][i,110,4]:.2f} °C")
    #print(tases.shape,year_numbers.shape)

    print(tas[i])
    #layer.set_array(tas[i].ravel())
    return layer,
#ax.pcolormesh(lo, la, tas[50], cmap="plasma",alpha = 0.5)
#layer.set_array(tas[20])
#import matplotlib.animation as animation
#anim = animation.FuncAnimation(fig, animate,
#                               frames=200, interval=20, blit=True)



# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    print(val)
    val = int(val)
    #line.set_ydata(signal(amp_slider.val, freq_slider.val))
    #fig.canvas.draw_idle()
    animate(val)
freq_slider.on_changed(sliders_on_changed)

def change_map_year(i):
    global selected_model
    selected_model = i
    animate(selected_year)
    #norm = matplotlib.colors.Normalize(vmin=cm.min,vmax=cm.max)
    #plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax)
    #cbar = fig.colorbar(layer, orientation="horizontal")


def color_radios_on_clicked_map(label):
    i = names.index(label)
    change_map_year(i)

color_radios.on_clicked(color_radios_on_clicked_map)
plt.show()

plt.show()
#anim.save('sine_wave.gif', writer='ffmpeg')

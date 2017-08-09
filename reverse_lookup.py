#------------------------------------------------------------------------#
#              Author: Miles Winter                                      #
#              Date: 06-29-2017                                          #
#              Desc: Use GeoPy to find country from lat,lon              #
#              Note: need GeoPy installed: pip install geopy             #
#------------------------------------------------------------------------#

from geopy.geocoders import Nominatim
import numpy as np

#import data file
event_data = np.genfromtxt('deco.csv',delimiter=',',dtype='S')
geolocator = Nominatim()

#select the lat and lon 
latitude = event_data[1:,12]
longitude = event_data[1:,13]
del event_data


#create arrays to hold unique countries and coordinates
country_list = np.array([],dtype='S')
coord_list = np.array([],dtype='S')

#Add new coordinates and countries to the array 
for i in range(len(latitude)):
    coordinate = "%s,%s"%(latitude[i],longitude[i])
    if coordinate not in coord_list:
        coord_list = np.append(coord_list,coordinate)
        print 'finding country for the coordinates ', coordinate
        location = geolocator.reverse(coordinate)
        country = location.address.split(',')[-1]
        if country not in country_list:
            country_list = np.append(country_list,country)

print np.unique(country_list)


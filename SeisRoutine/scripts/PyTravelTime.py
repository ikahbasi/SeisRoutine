#!/usr/bin/env python3
"""
Created on Sat Jan  7 13:32:08 2023
           Shanbeh  1401-10-18 

@author: Iman Kahbasi
"""
from obspy.geodetics.base import gps2dist_azimuth
from obspy import read_events
import matplotlib.pyplot as plt
import pandas as pd

def TravelTimePlot(catalog, df_station_coords):
    station_coords.set_index("name", inplace = True)
    phase_hint_lst = [{pick.phase_hint for pick in ev.picks} for ev in catalog]
    phase_hint_lst = list(set().union(*phase_hint_lst))
    traveltime = {}
    for phase_hint in phase_hint_lst:
        traveltime[phase_hint] = []
    for ev in catalog:
        otime = ev.preferred_origin().time
        evlat  = ev.preferred_origin().latitude
        evlon = ev.preferred_origin().longitude
        picks = ev.picks
        for pick in picks:
            ptime = pick.time
            phase_hint = pick.phase_hint
            station_name = pick.waveform_id.station_code
            stlat = station_coords.loc[station_name].latitude
            stlon = station_coords.loc[station_name].longitude
            dist, az1, az2 = gps2dist_azimuth(lat1=stlat, lon1=stlon,
                                              lat2=evlat, lon2=evlon,
                                              a=6378137.0,
                                              f=0.0033528106647474805)
            traveltime[phase_hint].append((dist/1000, ptime-otime))


    plt.figure(figsize=(12, 10))
    for phase_hint in phase_hint_lst:
        print(phase_hint)
        dist, time = zip(*traveltime[phase_hint])
        plt.plot(dist, time, '.', label=phase_hint)
    plt.legend()
    plt.xlabel('Distance [km]')
    plt.ylabel('Time [s]')
    plt.savefig('traveltime.png')


if __name__=="__main__":
    catalog_path = input('\nEnter the catalog file:\n')
    station_coords_path = input('\nEnter the station file as csv(name,ltitude,longitude):\n')
    catalog = read_events(catalog_path)
    station_coords = pd.read_csv(station_coords_path)
    TravelTimePlot(catalog=catalog, df_station_coords=station_coords)
    
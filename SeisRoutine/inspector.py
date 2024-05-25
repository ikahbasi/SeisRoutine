import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from obspy.imaging.cm import pqlx


def select_pick_of_arrival(arrival, picks):
    '''
    Docstring
    '''
    find_pick = False
    for pick in picks:
        if pick.resource_id == arrival.pick_id:
            find_pick = True
            break
    if not find_pick:
        pick = False
    return pick


def make_autopct(values):
    '''
    Docstring
    '''
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}% ({v:d})'.format(p=pct, v=val)
    return my_autopct


class catalog:
    '''
    Docstring
    '''
    def __init__(self, cat):
        self.df_phases = None
        self.catalog = cat
        self.__make_df()

    def __make_df(self):
        ######################### Events #########################
        lst = []
        for ev in self.catalog:
            origin = ev.preferred_origin()
            magnitude = ev.preferred_magnitude()
            if magnitude is None:
                magnitude = None
            else:
                magnitude = magnitude.mag
            d = {'otime': origin.time,
                    'latitude': origin.latitude,
                    'longitude': origin.longitude,
                    'magnitude': magnitude}
            lst.append(d)
        self.df_events = pd.DataFrame(lst)
        ######################### Phases #########################
        lst = []
        for ev in self.catalog:
            origin = ev.preferred_origin()
            magnitude = ev.preferred_magnitude()
            if magnitude is None:
                magnitude = None
            else:
                magnitude = magnitude.mag
            for arrival in origin.arrivals:
                pick = select_pick_of_arrival(arrival, ev.picks)
                d = {'otime': origin.time, 'magnitude': magnitude}
                d.update(pick.items())
                for key, val in arrival.items():
                    d[key] = val
                lst.append(d)
        self.df_phases = pd.DataFrame(lst)
        self.df_phases['station'] = self.df_phases.apply(
            lambda row: row.waveform_id.station_code, axis=1)
        self.df_phases['network'] = self.df_phases.apply(
            lambda row: row.waveform_id.network_code, axis=1)
        self.df_phases['channel'] = self.df_phases.apply(
            lambda row: row.waveform_id.channel_code, axis=1)
        self.df_phases['traveltime'] = self.df_phases.apply(
            lambda row: row.time-row.otime, axis=1)
        # Convert distance from degree to kilometre.
        self.df_phases['distance'] = self.df_phases.apply(
            lambda row: row.distance * 111, axis=1)
        

    def plot_hist_of_numeric(self, **kwargs):
        self.df_phases.hist(**kwargs)
        plt.tight_layout()

    def plot_pie_of_none_numeric(self, **kwargs):
        lst = ['network', 'channel']
        for key in lst:
            df = self.df_phases[key]
            counts = df.value_counts()
            # print(counts)
            counts.plot(kind='pie',
                        autopct=make_autopct(counts.values),
                        title=key,
                        **kwargs)
            plt.tight_layout()
            plt.show()

    def plot_bar_of_none_numeric(self, **kwargs):
        lst = ['network', 'channel']
        for key in lst:
            df = self.df_phases[key]
            counts = df.value_counts()
            # print(counts)
            counts.plot(kind='bar', title=key, **kwargs)
            plt.tight_layout()
            plt.show()

    def plot_traveltime(self):
        _, ax = plt.subplots()
        sns.scatterplot(self.df_phases,
                        x='distance', y='traveltime', s=10, hue='phase')
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Travel Time [sec]')

    def plot_bar_phasetype(self):
        counts = self.df_phases['phase'].value_counts()
        ax = counts.plot(kind='bar', edgecolor='k')
        ax.set_xlabel('Phase Type')
        ax.set_ylabel('Abundance [count]')

    def plot_residual_vs_distance(self):
        _, ax = plt.subplots()
        sns.scatterplot(self.df_phases,
                        x='distance', y='time_residual',
                        alpha=0.4, s=20, color='black',
                        ax=ax)
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Abundance [count]')

    def plot_hist_SminusP(self, bins=30, figsize=(7, 4)):
        # Selecting P- and S-type phases
        msk_p = self.df_phases['phase'].str.upper().str.startswith('P')
        msk_s = self.df_phases['phase'].str.upper().str.startswith('S')
        #
        keys = ['otime', 'resource_id', 'network', 'station', 'phase', 'time']
        df_p = self.df_phases[msk_p][keys]
        df_s = self.df_phases[msk_s][keys]
        # Changing type of the origin time
        df_p['otime'] = df_p['otime'].apply(lambda x: x.timestamp)
        df_s['otime'] = df_s['otime'].apply(lambda x: x.timestamp)
        # merge
        df_merge = df_p.merge(df_s, how='inner',
                              on=['station', 'network', 'otime'])
        sp_interval = df_merge['time_y'] - df_merge['time_x']
        print(f'Number of calculated s-p: {sp_interval.size}')
        print(f'Number of all phases: {self.df_phases.shape[0]}')
        print(f'Number of P-type phases: {df_p.shape[0]}')
        print(f'Number of S-type phases: {df_s.shape[0]}')
        ax = sp_interval.hist(edgecolor='k', bins=bins, figsize=figsize)
        ax.set_xlabel('S Minus P Time [sec]')
        ax.set_ylabel('Abundance [count]')

    def plot_phase_mag_dist(self, lst_stations=None):
        df = self.df_phases[['magnitude', 'distance', 'station']]
        if lst_stations:
            lst_stations = '|'.join(lst_stations)
            df = df[df['station'].str.contains(lst_stations, case=False)]
        msk = df.isna().sum(axis=1)
        msk = msk == 0
        mag = df['magnitude'][msk]
        dist = df['distance'][msk]
        #
        fig, ax0 = plt.subplots()
        hight, xedges, yedges = np.histogram2d(dist, mag,
                                               bins=(40, 20), density=False)
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        z = hight.T
        # z[z==0] = None
        im = ax0.pcolormesh(xcenters, ycenters, z,
                            cmap=pqlx, shading='auto', norm='log')
        fig.colorbar(im, ax=ax0)
        # plt.xlim(right=1600)
        plt.xlabel('Distance [km]')
        plt.ylabel('Magnitude')
        ax0.set_title('Phases distribution\n'
                      'according to magnitude and distance')

    def plot_station_participation(self, station_coords,
                                   map_focus='total', map_margin=0.05):
        phase_counts = self.df_phases['station'].value_counts()
        df = pd.merge(phase_counts, station_coords,
                      how='inner', on=['station'])
        _, ax = plt.subplots()
        ax.set_aspect('equal')
        sns.scatterplot(self.df_events,
                        x='longitude', y='latitude',
                        alpha=0.2, s=20, color='black', ax=ax)
        df.plot.scatter(x='longitude', y='latitude',
                        c='count', colormap='varidis',
                        edgecolors='r', linewidth=0.5,
                        marker='v', s=50, ax=ax)
        station_coords[['longitude', 'latitude', 'station']].apply(
            lambda x: ax.text(*x), axis=1)
        if map_focus == 'stations':
            lons = [station_coords['longitude'].min() - map_margin,
                    station_coords['longitude'].max() + map_margin]
            lats = [station_coords['latitude'].min() - map_margin,
                    station_coords['latitude'].max() + map_margin]
            plt.xlim(lons)
            plt.ylim(lats)

    def __str__(self):
        row, col = self.df_phases.shape
        txt = f'Number of Phases: {row}'
        txt += f'\nNumber of Attributes: {col}'
        return txt

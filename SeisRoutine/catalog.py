import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from obspy.imaging.cm import pqlx
import SeisRoutine.plot as srp
import SeisRoutine.core as src
import scipy as sp

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


class inspector:
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

    def plot_traveltime(self, ax=None, **kwargs):
        '''
        docstring ???
        '''
        if ax is None:
            fig, ax = plt.subplots()
        #
        sns.scatterplot(
            self.df_phases, x='distance', y='traveltime',
            s=10, hue='phase', ax=ax
        )
        srp._finalise_ax(
            ax,
            xlabel='Distance [km]', ylabel='Travel Time [sec]',
            **kwargs
        )
        srp._finalise_figure(ax.figure, **kwargs)

    def plot_bar_phasetype(self):
        counts = self.df_phases['phase'].value_counts()
        ax = counts.plot(kind='bar', edgecolor='k')
        ax.set_xlabel('Phase Type')
        ax.set_ylabel('Abundance [count]')

    def plot_residual_vs_distance(self, kind='density',
                                  ystep=0.5, xstep=5,
                                  histlog=True,
                                  **kwargs):
        '''
        kind: scatter or density
        '''
        distance = self.df_phases['distance'].to_numpy()
        residual = self.df_phases['time_residual'].to_numpy()
        phase = self.df_phases['phase'].to_numpy()
        srp.density_hist(
            x=distance, y=residual,
            xstep=xstep, ystep=ystep,
            kind=kind, histlog=histlog,
            xlabel='Distance [km]', ylabel='Residual Time [s]',
            hue=phase,
            **kwargs
        )

    def plot_hist_SminusP(self, bins=30, ax=None,
                          **kwargs):
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
        ax = sp_interval.hist(edgecolor='k', bins=bins, ax=ax)
        srp._finalise_ax(
            ax,
            xlabel='S Minus P Time [sec]', ylabel='Abundance [count]', **kwargs
        )
        srp._finalise_figure(ax.figure, **kwargs)


    def plot_phase_mag_dist(
            self, dist_step=10, mag_step=0.5,
            lst_stations=None, ax=None, **kwargs):
        '''
        docstring ???
        '''
        if ax is None:
            fig, ax = plt.subplots()
        #
        df = self.df_phases[['magnitude', 'distance', 'station']]
        if lst_stations:
            lst_stations = '|'.join(lst_stations)
            df = df[df['station'].str.contains(lst_stations, case=False)]
        #
        msk = df.isna().sum(axis=1)
        msk = (msk==0)
        mag = df['magnitude'][msk]
        dist = df['distance'][msk]
        #
        xcenters, ycenters, z = src.density_meshgrid(
            x=dist, y=mag,
            xstep=dist_step, ystep=mag_step,
            zreplace=0.9
        )
        im = ax.pcolormesh(
            xcenters, ycenters, z,
            cmap=pqlx, shading='auto', norm='log'
        )
        fig.colorbar(im, ax=ax)
        #
        title = ('Phases distribution\n'
                 'according to magnitude and distance')
        srp._finalise_ax(
            ax,
            xlabel='Distance [km]', ylabel='Magnitude', **kwargs
        )
        srp._finalise_figure(ax.figure, title=title, **kwargs)


    def plot_station_participation(
            self, station_coords, ax=None,
            map_focus='total', map_margin=0.05, **kwargs):
        '''
        docstring ???
        '''
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect('equal')
        #
        phase_counts = self.df_phases['station'].value_counts()
        df = pd.merge(phase_counts, station_coords,
                      how='inner', on=['station'])
        sns.scatterplot(
            self.df_events,
            x='longitude', y='latitude', size='magnitude',
            alpha=0.2, s=20, color='black', ax=ax
        )
        # sns.scatterplot(
        #     df,
        #     x='longitude', y='latitude',
        #     hue='count', colormap='viridis',
        #     edgecolors='r', linewidth=0.5,
        #     marker='v', s=50, ax=ax)
        # df.plot.scatter(x='longitude', y='latitude',
        #                 c='count', colormap='viridis',
        #                 edgecolors='r', linewidth=0.5,
        #                 marker='v', s=50, ax=ax)
        #
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        x = df['longitude'].to_numpy()
        y = df['latitude'].to_numpy()
        c = df['count'].to_numpy()
        points = ax.scatter(x, y, c=c, marker='v', s=50, cmap="plasma")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(points, cax=cax)
        #
        station_coords[['longitude', 'latitude', 'station']].apply(
            lambda x: ax.text(*x), axis=1)
        if map_focus == 'stations':
            lons = [station_coords['longitude'].min() - map_margin,
                    station_coords['longitude'].max() + map_margin]
            lats = [station_coords['latitude'].min() - map_margin,
                    station_coords['latitude'].max() + map_margin]
            plt.xlim(lons)
            plt.ylim(lats)
        srp._finalise_ax(
            ax,
            xlabel='Longitude', ylabel='Latitude', **kwargs
        )
        srp._finalise_figure(ax.figure, **kwargs)

    def plot_statistical_station_participation_and_time_residuals(self, **kwargs):
        df_selection = self.df_phases.sort_values(by=['station'])
        #
        x = df_selection['station'].values
        y = df_selection['time_residual']
        #
        phase = df_selection['phase_hint'].apply(lambda x: x[0]).values
        xp = x[phase=='P']
        yp = y[phase=='P']
        xs = x[phase=='S']
        ys = y[phase=='S']
        #
        dfg = df_selection.groupby(['station'])['time_residual']
        mode = dfg.apply(lambda x: sp.stats.mode(x)[0])
        counts = dfg.count()
        mean = dfg.mean()
        std = dfg.std()
        #
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 6), sharex=True, height_ratios=[1.5, 4])
        #
        bars = ax0.bar(counts.index, counts.values, align='center', log=False)
        ax0.set_ylim(top=counts.values.max()*1.15)
        ax0.bar_label(bars, size=8)
        m = dfg.count().values.max()
        m = round(m, -len(str(m))+1)
        ax0.set_yticks([m, m//2])
        ax0.set_ylabel('N-Phases')
        #
        kw = {'facecolors': 'none', 'marker': '.', 'alpha': 0.7, 's': 150}
        ax1.scatter(xp, yp, edgecolors='r', label='P-phase', **kw)
        ax1.scatter(xs, ys, edgecolors='b', label='S-phase', **kw)
        ax1.plot(mode.index, mode.values, 'k-.', label='Mode')
        ax1.plot(mean.index, mean.values, 'g', label='Mean')
        ax1.fill_between(
            mean.index,
            mean.values - std.values,
            mean.values + std.values,
            alpha=0.2, label='STD'
            )
        ax1.set_ylabel('Time Residuals [s]')
        ax1.legend(loc=3, ncol=5)
        ax1.set_ylim([-1, 1])
        ax1.grid()
        #
        plt.subplots_adjust(bottom=0.15, hspace=0)
        plt.xlabel('Station Name')
        plt.xticks(rotation=60)
        #
        srp._finalise_figure(fig, **kwargs)

    def plot_seismicity_phases(self):
        '''
        It must perform on each type of P and S seperatly ???
        '''
        gb = self.df_phases.sort_values(
            by=['time']).groupby(by=['network', 'station'])
        gb = gb['time'].apply(lambda x: x.diff())
        gb.plot.hist(alpha=1, edgecolor='k', facecolor='g')

    def plot_seismicity_events(self):
        dt = self.df_events['otime'].apply(lambda x: x.datetime)
        dt.hist(alpha=1, edgecolor='k', facecolor='g', grid=False)
        plt.xticks(rotation=90)

    def __str__(self):
        row, col = self.df_phases.shape
        txt = f'Number of Phases: {row}'
        txt += f'\nNumber of Attributes: {col}'
        return txt

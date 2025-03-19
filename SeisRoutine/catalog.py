import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from obspy.imaging.cm import pqlx
import SeisRoutine.plot as srp
import SeisRoutine.core as src
import scipy as sp
from obspy.core.event import Catalog


def select_pick_of_arrival(arrival, picks):
    """
    Find and return the corresponding pick for a given arrival.

    Parameters:
        arrival (obspy.core.event.origin.Arrival): 
            The arrival object containing a reference to a pick.
        picks (list of obspy.core.event.origin.Pick): 
            A list of pick objects to search within.

    Returns:
        obspy.core.event.origin.Pick or bool: 
            The corresponding pick object if found, otherwise `False`.

    Notes:
        - The function iterates through the list of picks and matches them 
          with the `pick_id` of the given `arrival`.
        - If no matching pick is found, it returns `False`.
    """
    find_pick = False
    for pick in picks:
        if pick.resource_id == arrival.pick_id:
            find_pick = True
            break
    if not find_pick:
        pick = False
    return pick


def select_arrival_related_to_the_pick(pick, arrivals):
    '''
    This function selects the arrival related to the given pick from a list of arrivals.

    Parameters:
    pick (Pick): The obspy pick object to find the related arrival for.
    arrivals (list): A list of obspy arrival objects to search through.

    Returns:
    Arrival: The arrival object related to the given pick, or False if no related arrival is found.
    '''
    find_arrival = False
    for arrival in arrivals:
        if arrival.pick_id == pick.resource_id:
            find_arrival = True
            break
    if not find_arrival:
        arrival = False
    return arrival


def make_autopct(values):
    """
    Generate a function for displaying percentage and absolute value in a pie chart.

    This function creates a custom autopct function for use in `matplotlib.pyplot.pie`,
    showing both the percentage and the absolute count of each segment.

    Parameters:
        values (list or array-like): The dataset values used to calculate absolute counts.

    Returns:
        function: A formatting function that takes a percentage (`pct`) and returns a string
                  formatted as '{percentage}% (absolute count)'.

    Example:
        import matplotlib.pyplot as plt

        sizes = [30, 20, 50]
        plt.pie(sizes, autopct=make_autopct(sizes))
        plt.show()
    """
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}% ({v:d})'.format(p=pct, v=val)
    return my_autopct


def select_picks(picks, station_name, without_amplitued=True):
    """
    Filters seismic picks for a specific station.

    Parameters:
        picks (list): A list of seismic picks.
        station_name (str): The name of the station to filter picks for.
        without_amplitued (bool, optional): 
            - If True (default), removes picks where the phase hint starts with 'AM'.
            - If False, keeps only picks where the phase hint starts with 'AM'.

    Returns:
        list: A list of picks that are filtered by station and sorted by time.
    """
    cond = lambda p: (p.waveform_id.station_code==station_name) and (p.phase_hint.startswith('AM')!=without_amplitued)
    picks = list(filter(cond, picks))
    picks = sorted(picks, key=lambda p: p.time)
    return picks


def print_phase_frequency(catalog, case_sensitivity=True):
    """
    Prints the frequency of different seismic phase hints in a catalog 
    and counts the number of P-S phase pairs.

    Parameters:
        catalog (list): A list of seismic events, where each event contains picks.
        case_sensitivity (bool, optional): 
            - If True (default), phase hints are treated as case-sensitive.
            - If False, phase hints are converted to lowercase before counting.

    Returns:
        None: The function prints the frequency of phase hints and 
              the total number of P-S phase pairs.
    """
    lst = []
    for ev in catalog:
        for pick in ev.picks:
            lst.append(pick.phase_hint)
    if not case_sensitivity:
        lst = [el.lower() for el in lst]
    ###
    number_of_ps_pairs = 0
    for event in catalog:
        stations_in_event = {pick.waveform_id.station_code for pick in event.picks}
        for station_name in stations_in_event:
            picks = select_picks(picks=event.picks,
                                 station_name=station_name)
            phase_hints = [pick.phase_hint[0].upper() for pick in picks if pick.phase_hint]
            if ('P' in phase_hints) and ('S' in phase_hints) and (len(phase_hints) >= 2):
                number_of_ps_pairs += 1
    ###
    for el in set(lst):
        print(el, lst.count(el))
    print('The Number of P-S Pairs in the Catalog is:', number_of_ps_pairs)


class inspector:
    """
    A class to inspect and analyze seismic event catalogs.
    
    Attributes:
        catalog (obspy.core.event.Catalog):
            Input seismic event catalog.
        df_phases (pandas.DataFrame):
            Dataframe containing phase information.
        df_events (pandas.DataFrame):
            Dataframe containing event information.
        classified_catalog (dict):
            Dictionary of classified events.
        quality_citeria_list (dict):
            Classification criteria for event quality.
    """
    def __init__(self, cat):
        self.df_phases = None
        self.catalog = cat
        self.__make_df_events()
        self.__make_df_phases()
        self.classified_catalog = None
        self.quality_citeria_list = {
            'Hypoellipse & NLLOC': {
                1: {"stations": [3, 999], "gap": [0, 270], "rms": [0, 0.5], "errorH": [0, 3.0], "errorZ": [0, 3.0], 'evaluation': 'manual', 'label': 'A'},
                2: {"stations": [3, 999], "gap": [0, 270], "rms": [0, 0.5], "errorH": [0, 5.0], "errorZ": [0, 5.0], 'evaluation': 'manual', 'label': 'B'},
                3: {"stations": [3, 999], "gap": [0, 360], "rms": [0, 0.5], "errorH": [0, 5.0], "errorZ": [0, 5.0], 'evaluation': 'manual', 'label': 'C'},
                4: {"stations": [3, 999], "gap": [0, 360], "rms": [0, 9.9], "errorH": [0, 999], "errorZ": [0, 999], 'evaluation': 'manual', 'label': 'D'},
                }
            }

    def __make_df_events(self):
        """
        Creates a DataFrame containing event information from the seismic catalog.

        This method iterates over each event in the catalog, extracts relevant information
        about the event's origin and magnitude, as well as quality metrics. The information
        is stored in a list of dictionaries and then converted into a pandas DataFrame.

        The resulting DataFrame (`self.df_events`) contains the following columns:
            - 'otime': Origin time of the seismic event.
            - 'latitude': Latitude of the seismic event origin.
            - 'longitude': Longitude of the seismic event origin.
            - 'depth': Depth of the seismic event origin (in kilometers).
            - 'magnitude': Magnitude of the seismic event (if available).
            - 'num_stations': Number of stations used for the event location.
            - 'azimutal_gap': Azimuthal gap in degrees (range from 0 to 360).
            - 'rms': Root mean square of the residuals for the event location.
            - 'errorH': Horizontal uncertainty in kilometers (calculated from latitude and longitude errors).
            - 'errorZ': Vertical uncertainty in kilometers (calculated from depth errors).
            - 'evaluation': Evaluation mode of the event (e.g., automatic or manual).

        The method updates the `self.df_events` attribute with the newly created DataFrame.
        """
        ######################### Events #########################
        lst = []
        for ev in self.catalog:
            origin = ev.preferred_origin()
            magnitude = ev.preferred_magnitude()
            if magnitude is None:
                magnitude = None
            else:
                magnitude = magnitude.mag
            ###
            d = {'otime': origin.time,
                 'latitude': origin.latitude,
                 'longitude': origin.longitude,
                 'depth': origin.depth,
                 'magnitude': magnitude,
            }
            if origin.quality is not None:
                 d.update({
                    'num_stations': origin.quality.used_station_count,
                    #  'stations': len({pick.waveform_id.station_code for pick in ev.picks}),
                    'azimutal_gap': origin.quality.azimuthal_gap,
                    'rms': origin.quality.standard_error,
                    #  'rms': np.sqrt(
                    #             np.mean(
                    #                 np.array(
                    #                     [arrival.time_residual for arrival in origin.arrivals
                    #                      if arrival.time_residual is not None]) ** 2
                    #             )),
                    'errorH': np.sqrt(
                                    (origin.latitude_errors['uncertainty']*111)**2 +
                                    (origin.longitude_errors['uncertainty']*111)**2),
                    #  'errorH': origin.origin_uncertainty.horizontal_uncertainty / 1000,
                    'errorZ': origin.depth_errors['uncertainty'] / 1000,
                    'evaluation': origin.evaluation_mode,
                })
            lst.append(d)
        self.df_events = pd.DataFrame(lst)

    def __make_df_phases(self):
        """
        Creates a DataFrame containing seismic phase information from the seismic catalog.

        This method iterates over each event in the catalog, extracts relevant information 
        about the event's origin, magnitude, and seismic phase arrivals. For each arrival, 
        it identifies the associated pick, and stores both phase and arrival information 
        into a list of dictionaries. Afterward, it converts the list into a pandas DataFrame 
        and computes additional columns for the station, network, channel, travel time, and 
        distance.

        The resulting DataFrame (`self.df_phases`) contains the following columns:
            - 'otime': Origin time of the seismic event.
            - 'magnitude': Magnitude of the seismic event (if available).
            - 'waveform_id': Information about the waveform associated with the arrival.
            - 'station': Station code where the phase was detected.
            - 'network': Network code associated with the station.
            - 'channel': Channel code for the seismic measurement.
            - 'traveltime': Time difference between the arrival time and the origin time (in seconds).
            - 'distance': Distance from the seismic event's origin to the station (in kilometers).
            - Plus any additional keys from the arrival items (e.g., phase, arrival time, etc.).

        The method updates the `self.df_phases` attribute with the newly created DataFrame.
        """
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
    
    def quality_classification(self, citeria='Hypoellipse & NLLOC'):
        if isinstance(citeria, str):
            citeria = self.quality_citeria_list[citeria]
        #
        classified_catalog = {val['label']: Catalog() for key, val in citeria.items()}
        classified_catalog['Not Classified'] = Catalog()
        #
        for ev in self.catalog:
            origin = ev.preferred_origin()
            try:
                stations = origin.quality.used_station_count
                if stations == 0:
                    continue
            except Exception as error:
                print('Skipping form an event. Be careful!', error, origin,
                      sep='\n')
                continue
            # stations = len({pick.waveform_id.station_code for pick in ev.picks})
            gap = origin.quality.azimuthal_gap
            rms = origin.quality.standard_error
            # rms = np.array([arrival.time_residual for arrival in origin.arrivals if arrival.time_residual is not None])
            # rms = np.sqrt(np.mean(rms ** 2))
            errorH = np.sqrt((origin.latitude_errors['uncertainty']*111)**2 +
                             (origin.longitude_errors['uncertainty']*111)**2)
            # errorH = origin.origin_uncertainty.horizontal_uncertainty / 1000
            errorZ = origin.depth_errors['uncertainty'] / 1000
            evaluation = origin.evaluation_mode
            #
            classified = False
            for key, params in citeria.items():
                if (params['stations'][0] < stations < params['stations'][1]) and \
                    (params['gap'][0] < gap < params['gap'][1]) and \
                    (params['rms'][0] < rms < params['rms'][1]) and \
                    (params['errorH'][0] < errorH < params['errorH'][1]) and \
                    (params['errorZ'][0] < errorZ < params['errorZ'][1]) and \
                    (params['evaluation'] ==  evaluation):
                    classified_catalog[params['label']] += ev
                    classified = True
            if not classified:
                classified_catalog['Not Classified'] += ev
        self.classified_catalog = classified_catalog

    def plot_quality_classification(self):
        n = self.catalog.count()
        precent = {}
        for key, val in self.classified_catalog.items():
            precent[key] = round(val.count() / n * 100, 2)
        
        fig, ax = plt.subplots()
        x = list(precent.keys())
        y = list(precent.values())
        ax.bar(x, y,
               align='center', edgecolor='k')
        for p in ax.patches:
            txt = str(p.get_height())
            xy = (p.get_x() + 0.5,
                  p.get_height()  + min(y) * 0.1)
            ax.annotate(text=txt, xy=xy, ha='center')
        plt.show()

    def plot_station_participation_per_event(self):
        '''
        Histogram plot.
        '''
        ax = self.df_events['num_stations'].fillna(-1).astype(int)\
            .value_counts().sort_index().plot(kind='bar', rot=0)
        _ = ax.bar_label(ax.containers[0])
        ax.set_title('How many stations are used for location')
        ax.set_xlabel('Number of Stations')
        ax.set_ylabel('Number of Events')

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

    def plot_seismicity_phases(self, target_phase='P', bins=10):
        '''
        It must perform on each type of P and S seperatly ???
        '''
        msk = self.df_phases['phase_hint'].apply(lambda x: x.upper()[0] == target_phase)
        df_phases = self.df_phases[msk]
        gb = df_phases.sort_values(
            by=['time']).groupby(by=['network', 'station'])
        gb = gb['time'].apply(lambda x: x.diff())
        ax = gb.plot.hist(alpha=1, edgecolor='k', facecolor='g', bins=bins)
        ax.set_xlabel("Time Interval [s]")

    def plot_seismicity_events(self):
        dt = self.df_events['otime'].apply(lambda x: x.datetime)
        dt = dt.groupby(dt.dt.date).count()
        dt.plot(kind='bar', edgecolor='k', facecolor='g', grid=False)
        # dt.hist(alpha=1, edgecolor='k', facecolor='g', grid=False)
        plt.xticks(rotation=90)

    def __str__(self):
        row, col = self.df_phases.shape
        txt = f'Number of Phases: {row}'
        txt += f'\nNumber of Attributes: {col}'
        return txt

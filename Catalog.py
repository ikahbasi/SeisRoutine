import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def select_PickOfArrival(arrival, picks):
    find_pick = False
    for pick in picks:
        if pick.resource_id == arrival.pick_id:
            find_pick = True
            break
    if not find_pick:
        pick = False
    return pick


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}% ({v:d})'.format(p=pct,v=val)
    return my_autopct


class inspector:
    def __init__(self, catalog):
        self.df_phases = None
        self.catalog = catalog
        self.__make_df()
    def __make_df(self):
        lst = []
        for ev in self.catalog:
            origin = ev.preferred_origin()
            for arrival in origin.arrivals:
                pick = select_PickOfArrival(arrival, ev.picks)
                d = {'otime': origin.time}
                d.update(pick.items())
                for key, val in arrival.items():
                    d[key] = val
                lst.append(d)
        self.df_phases = pd.DataFrame(lst)
        self.df_phases['station'] = self.df_phases.apply(lambda row: row.waveform_id.station_code, axis=1)
        self.df_phases['network'] = self.df_phases.apply(lambda row: row.waveform_id.network_code, axis=1)
        self.df_phases['channel'] = self.df_phases.apply(lambda row: row.waveform_id.channel_code, axis=1)
        self.df_phases['traveltime'] = self.df_phases.apply(lambda row: row.time-row.otime, axis=1)
    def plot_hist_of_numeric(self, **kwargs):
        self.df_phases.hist(**kwargs)
        plt.tight_layout()
    def plot_pie_of_none_numeric(self, **kwargs):
        lst = ['network', 'channel']
        for key in lst:
            df = self.df_phases[key]
            counts = df.value_counts()
            # print(counts)
            counts.plot(kind='pie', autopct=make_autopct(counts.values), title=key, **kwargs)
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
        sns.scatterplot(self.df_phases, x='distance', y='traveltime', s=10, hue='phase')

    def plot_bar_phasetype(self):
        counts = self.df_phases['phase'].value_counts()
        _ = counts.plot(kind='bar', edgecolor='k')

    def plot_residual_vs_distance(self):
        sns.scatterplot(self.df_phases, x='distance', y='time_residual', alpha=0.4, s=20, color='black')
    
    def plot_hist_SminusP(self, bins=30, figsize=(7, 4)):
        ### Selecting P- and S-type phases
        msk_p = self.df_phases['phase'].str.upper().str.startswith('P')
        msk_s = self.df_phases['phase'].str.upper().str.startswith('S')
        ###
        keys = ['otime', 'resource_id', 'network', 'station', 'phase', 'time']
        df_p = self.df_phases[msk_p][keys]
        df_s = self.df_phases[msk_s][keys]
        ### Changing type of the origin time
        df_p['otime'] = df_p['otime'].apply(lambda x: x.timestamp)
        df_s['otime'] = df_s['otime'].apply(lambda x: x.timestamp)
        ### merge
        df_merge = df_p.merge(df_s, how='inner', on=['station', 'network', 'otime'])
        sp_interval = df_merge['time_y'] - df_merge['time_x']
        print(f'Number of calculated s-p: {sp_interval.size}')
        print(f'Number of all phases: {self.df_phases.shape[0]}')
        print(f'Number of P-type phases: {df_p.shape[0]}')
        print(f'Number of S-type phases: {df_s.shape[0]}')
        sp_interval.hist(edgecolor='k', bins=bins, figsize=figsize)



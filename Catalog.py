import pandas as pd

def select_PickOfArrival(arrival, picks):
    origin = ev.preferred_origin()
    find_pick = False
    for pick in picks:
        if pick.resource_id == arrival.pick_id:
            find_pick = True
            break
    if not find_pick:
        pick = False
    return pick



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


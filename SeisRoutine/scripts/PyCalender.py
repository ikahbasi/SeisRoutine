#!/usr/bin/env python3
#
# version1:  1401/05/04
# writen by: Iman Kahbasi
#
from datetime import datetime as dt
from khayyam import JalaliDate
import sys
import pandas as pd


def get_switchs(inp):
    args = {}
    for el in inp['options'].split():
        key, val = el.split('=')
        args[key] = val
    return args

def get_detail_time(time):
    mon_name = time.strftime("%B")
    day_name = time.strftime("%A")
    year, week, weekday = time.isocalendar()
    tt = time.timetuple()
    detail = [time, mon_name, day_name, week, tt.tm_yday]
    '''
    detail = {'Date': time,
              'Month': mon_name,
              'Day': day_name,
              'Week': week,
              'Julday': tt.tm_yday}
    '''
    return detail


if __name__ == "__main__":
    arg_names = ['command', 'options']
    args = dict(zip(arg_names, sys.argv))
    ###
    if args.get('options', False):
        options = get_switchs(args)
        if utc:=options.get('utc', False):
            time_miladi = dt.strptime(utc, '%Y-%m-%d')
            time_miladi = time_miladi.date()
            time_shamsi = JalaliDate(time_miladi)
        elif jal:=options.get('jal', False):
            time_shamsi = JalaliDate.strptime(jal, '%Y-%m-%d')
            time_miladi = time_shamsi.todate()
    else:
        time_miladi = dt.now()
        time_miladi = time_miladi.date()
        time_shamsi = JalaliDate(time_miladi)
    ###
    d = {'Miladi': get_detail_time(time_miladi),
         'Shamsi': get_detail_time(time_shamsi)}
    df = pd.DataFrame(data=d, index=['Date', 'Month', 'Day', 'Week', 'Julday'])
    ###
    print(df)

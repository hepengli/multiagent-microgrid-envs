"""
Oct 16, 2020
Created by Hepeng Li

Read uncertainty data
"""
import os, re
import numpy as np
import pandas as pd
from scipy.io import loadmat

def read_data(train=True):
    price_path = '/home/lihepeng/Documents/Github/tmp/MG/data/price'
    load_path = '/home/lihepeng/Documents/Github/tmp/MG/data/load'
    renewable_path = '/home/lihepeng/Documents/Github/tmp/MG/data/renewable'
    tdays = 21
    if train:
        price_files = [os.path.join(price_path, f) for f in os.listdir(price_path) if re.match(r'^2016\d+.mat$', f)]
        price_data = [loadmat(f)['price'].transpose()[:tdays,:].ravel() for f in price_files]
        price_data = np.maximum(np.hstack(price_data).ravel() * 0.2, 1)
        price_data = np.minimum(price_data, 18.0)
        price_data = np.round(price_data, 2)

        load_files = [os.path.join(load_path, f) for f in os.listdir(load_path) if re.match(r'^2016\d+.mat$', f)]
        load_data = [loadmat(f)['demand'].transpose()[:tdays,:].ravel() for f in load_files]
        load_data = np.hstack(load_data).ravel() * 3.0

        renew_files = [os.path.join(renewable_path, f) for f in os.listdir(renewable_path) if re.match(r'^2016\d+.mat$', f)]
        solar_data = [loadmat(f)['solar_power'].transpose()[:tdays,:].ravel() for f in renew_files]
        wind_data = [loadmat(f)['wind_power'].transpose()[:tdays,:].ravel() for f in renew_files]
        solar_data = np.hstack(solar_data).ravel() * 6 / 1000
        wind_data = np.hstack(wind_data).ravel() * 6 / 1000
    else:
        price_files = [os.path.join(price_path, f) for f in os.listdir(price_path) if re.match(r'^2016\d+.mat$', f)]
        price_data = [loadmat(f)['price'].transpose()[tdays:,:].ravel() for f in price_files]
        price_data = np.maximum(np.hstack(price_data).ravel() * 0.2, 1)
        price_data = np.minimum(price_data, 18.0)
        price_data = np.round(price_data, 3)

        load_files = [os.path.join(load_path, f) for f in os.listdir(load_path) if re.match(r'^2016\d+.mat$', f)]
        load_data = [loadmat(f)['demand'].transpose()[tdays:,:].ravel() for f in load_files]
        load_data = np.hstack(load_data).ravel() * 3.0

        renew_files = [os.path.join(renewable_path, f) for f in os.listdir(renewable_path) if re.match(r'^2016\d+.mat$', f)]
        solar_data = [loadmat(f)['solar_power'].transpose()[tdays:,:].ravel() for f in renew_files]
        wind_data = [loadmat(f)['wind_power'].transpose()[tdays:,:].ravel() for f in renew_files]
        solar_data = np.hstack(solar_data).ravel() * 6 / 1000
        wind_data = np.hstack(wind_data).ravel() * 6 / 1000

    size = price_data.size
    days = price_data.size // 24

    return {'load': load_data, 'solar': solar_data, 'wind': wind_data, 'price':price_data, 'days':days, 'size':size}


def read_pickle_data():
    import pickle, os
    home_path = '/home/lihepeng/Documents/Github/'
    f = open(os.path.join(home_path,'multiagent-microgrid-envs','data','data2018-2020.pkl'), 'rb')
    data = pickle.load(f)
    f.close()

    return data
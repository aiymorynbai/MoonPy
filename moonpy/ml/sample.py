import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
import numpy as np

def df_train_test_split(df_ref: pd.DataFrame, test_size: float, random_seed: int, stratification_clm: str):
    '''
    Performs split of polygons to train and test. Input parameters:
    
    - df_ref: reference GeoDataFrame with all polygons;
    - test_size: test size in floats, exp: 0.5 for 50%;
    - random_seed: random seed =);
    - stratification_clm: column name used for stratification
    
    '''
    stratif_vals = df_ref[stratification_clm].values
    assert df_ref.index.is_unique, 'the index of input GeoDataFrame is not unique'
    index_vals = df_ref.index.values
    
    idx_train, idx_test = train_test_split(index_vals, test_size=test_size,
                                           random_state=random_seed, stratify=stratif_vals)
    print('Lenth of training polygons: ', len(idx_train))
    print('Lenth of validation polygons: ', len(idx_test))
    
    assert (len(idx_train) + len(idx_test)) == df_ref.shape[0], 'Lenths of train + test is different to reference. Please check your data or entries'
    
    df_train = df_ref.loc[np.sort(idx_train)]
    df_test = df_ref.loc[np.sort(idx_test)]
    assert df_test.index.isin(df_train.index).sum() == 0, 'One or more polygons accured in train and test'
    
    return df_train, df_test


def sample_pixs_equally_stratified(df: pd.DataFrame, col_cropid: str, col_parcelid: str, sample_size: int, random_state: int):
    '''
    sample equally distributed number of pixels
    
    parameters: 
    
    df: pandas DataFrame where rows represent single pixel;
    col_cropid: column of the input df indicating crop ids or names;
    col_parcelid: column of the input df indicating parcel ids;
    sample size: number of pixels to samples from each crop type;
    random_state: random state.
    
    '''
    
    lst_trainset = []
    for cropid, _df in df.groupby([col_cropid], group_keys=False):
        ps_polids = pd.Series(_df[col_parcelid].unique())
    
        # if number of parcel more or equal to the sample size sample one pixel per parcel
        if ps_polids.shape[0] >= sample_size:
            sampled_pols = ps_polids.sample(n=sample_size, random_state=random_state)
            lst_train_pixs = []
            for pol in list(sampled_pols):
                lst_train_pixs.append(_df.loc[_df[col_parcelid] == pol].sample(n=1, random_state=random_state))
            lst_trainset.append(pd.concat(lst_train_pixs))
        # when number of polygons are less than sample size, equally distribute pixels over all parcels
        else: 
            _df_stat = pd.DataFrame(_df[col_parcelid].value_counts()).reset_index().rename(columns={'index':'pol_id', col_parcelid:'numb_pixs'})
            _df_stat['available'] = _df_stat['numb_pixs']
            _df_stat['to_sample'] = 0
            #assert _df_stat.numb_pixs.sum() > sample_size, 'Number of pixels are less than sample size'
            if _df_stat.numb_pixs.sum() > sample_size:
                while _df_stat.to_sample.sum() < sample_size:
                    _df_stat.loc[_df_stat.available >= 1, 'to_sample'] = _df_stat.loc[_df_stat.available >= 1, 'to_sample'] + 1
                    _df_stat.loc[_df_stat.available >= 1, 'available'] = _df_stat.loc[_df_stat.available >= 1, 'available'] - 1

                _diff = _df_stat.to_sample.sum() - sample_size
                if _diff != 0:
                    _ = _df_stat[_df_stat.to_sample > 1].sample(n=_diff, random_state=random_state)
                    _df_stat.loc[_.index.values, 'to_sample'] = _df_stat.loc[_.index.values, 'to_sample'] - 1
                    assert _df_stat.to_sample.sum() == sample_size, 'Number of pixels selected to sample is not equal to sample size'
                else: 
                    print('Difference = 0')
            else:
                _df_stat['to_sample'] = _df_stat['numb_pixs']

            lst_train_pixs = []
            for pol in ps_polids:
                numb_pixs_to_sample = int(_df_stat[_df_stat.pol_id == pol].to_sample.values)
                lst_train_pixs.append(_df.loc[_df[col_parcelid] == pol].sample(n=numb_pixs_to_sample, random_state=random_state))
            lst_trainset.append(pd.concat(lst_train_pixs))
        
    df_out = pd.concat(lst_trainset)
    return df_out


def sample_pixs_proportionally_stratified(df, max_samples_number, random_state, col_cropid, col_parcelid):
    list_dfs = []
    for crop, group in df.groupby(col_cropid, group_keys=False):
        group_percentage = ((group.shape[0]*100)/df.shape[0])
        sample_numbs_from_percentage = round((max_samples_number * group_percentage) / 100)
        list_dfs.append(sample_pixs_equally_stratified(df=group, col_parcelid=col_parcelid, col_cropid=col_cropid, sample_size=sample_numbs_from_percentage, random_seed=random_state))
    df_output = pd.concat(list_dfs)
    return df_output


def sample_disproportionately_stratified(df, max_samples_number, perc_increase, random_state, col_cropid, col_parcelid):
    ps_pixperc = ((df[col_cropid].value_counts() / df.shape[0]) * 100).round(1)
    df_perc = pd.DataFrame(ps_pixperc).rename(columns={col_cropid:'perc'})
    perc_mean = ps_pixperc.mean().round(1)
    df_perc['perc_recalc'] = df_perc['perc']
    # for classes with less than 10% increase the share by 1 percent
    df_perc.loc[df_perc.perc <= perc_mean, 'perc_recalc'] = df_perc.loc[df_perc.perc <= perc_mean, 'perc'] + perc_increase
    increased_diff = df_perc[df_perc.perc <= perc_mean].perc_recalc.sum().round(1) - df_perc[df_perc.perc <= perc_mean].perc.sum().round(1)
    perc_substract = ((df_perc[df_perc.perc > perc_mean] / df_perc[df_perc.perc > perc_mean].perc.sum()) * 100).round(0)

    for idx, perc_substr in perc_substract.iterrows():
        value_substr = ((increased_diff * perc_substr.values[0]) / 100).round(0)
        df_perc.loc[df_perc.index == idx, 'perc_recalc'] = df_perc.loc[df_perc.index == idx, 'perc'].values[0] - value_substr
    #assert int(df_perc.perc_recalc.sum().round(0)) == 100, 'Sum of percentages is not 100'
    df_perc['pix_numb'] = 0

    for idx, row in df_perc.iterrows():
        df_perc.loc[df_perc.index == idx, 'pix_numb'] = (max_samples_number * row.perc_recalc) / 100
    
    list_dfs = []
    for crop, group in df.groupby(col_cropid, group_keys=False):
        pix_numb = int(df_perc.loc[df_perc.index == crop, 'pix_numb'].values[0])
        list_dfs.append(sample_pixs_equally_stratified(df=group, col_parcelid =col_parcelid, col_cropid = col_cropid, 
                                                           sample_size = pix_numb, random_state = random_state))
    df_output = pd.concat(list_dfs)
    return df_output
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q3 reference reconstruction + matched-sample ENSO lag robustness.

Requirements: numpy pandas scipy, alongside
  enso_formal_interaction_models.py
  enso_hydroclimate_panel_models.py

For every GUB and calendar month, reference area Q3 is the mean vegetation area
in 2019-2025 among Q2-qualified records only (at least --min-hist Q2 years).
Regressions use only Q2-qualified GUB-year-month observations for BOTH outcomes:
(1) reference mean over all seven years (old outcome)
(2) new Q3 quality-qualified historical reference.
Common exact rows across lags 0-3; 5-degree block x year-month clusters.

Outputs include reference audit, sample retention, all H0/H1/H2 regressions for
all lags, matched original vs Q3 estimates, and optional Q3 M0/M1 for lag 1.
Results are associative, not ENSO causal identification.
"""
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd

import enso_formal_interaction_models as base
import enso_hydroclimate_panel_models as hmod

KEY=['ORIG_FID','year','month']

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',required=True)
    p.add_argument('--quality',required=True,help='SCL_quality_all72.csv with clear2 coverage')
    p.add_argument('--exposure',required=True,help='enso_exposure_long_all16.csv')
    p.add_argument('--geo',required=True)
    p.add_argument('--roni',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--lags',nargs='+',type=int,choices=range(4),default=[0,1,2,3])
    p.add_argument('--min-clear',type=float,default=80)
    p.add_argument('--min-clear2',type=float,default=50)
    p.add_argument('--max-snow',type=float,default=5)
    p.add_argument('--min-hist',type=int,default=5)
    p.add_argument('--min-reference-area',type=float,default=10000)
    p.add_argument('--twoway',action='store_true',default=True,help='Retained for CLI compatibility; default is two-way')
    p.add_argument('--also-m',action='store_true',help='Also estimate M0/M1 at lag 1, Q3 and old outcome')
    return p.parse_args()

def load_quality(path, a):
    q=pd.read_csv(path,dtype={'ORIG_FID':'string'},low_memory=False)
    q.columns=q.columns.str.strip()
    req=KEY+['clear_coverage_pct','clear2_coverage_pct','snow_any_pct']
    missing=set(req)-set(q)
    if missing:raise ValueError(f'Quality missing {missing}. Use SCL_quality_all72.csv, not Q2_for_H_model.csv')
    q=q[req].copy()
    q.ORIG_FID=q.ORIG_FID.str.strip()
    for k in req[1:]:q[k]=pd.to_numeric(q[k],errors='coerce')
    if q.duplicated(KEY).any():raise ValueError('Duplicate GUB-year-month quality rows')
    for k in ['clear_coverage_pct','clear2_coverage_pct','snow_any_pct']:
        v=q[k].dropna()
        if v.lt(0).any() or v.gt(100.001).any():raise ValueError(f'Quality field {k} outside 0-100')
    q['q2_pass']=(q.clear_coverage_pct.ge(a.min_clear)&
                  q.clear2_coverage_pct.ge(a.min_clear2)&
                  q.snow_any_pct.le(a.max_snow))
    return q[KEY+['q2_pass']]

def common_beta_keys(path,lags):
    b=pd.read_csv(path,dtype={'ORIG_FID':'string'},low_memory=False)
    b.columns=b.columns.str.strip()
    required={'ORIG_FID','season','enso_lag','ENSO_VPD_beta','ENSO_SM_beta'}
    if not required.issubset(b):
        if 'lag' in b.columns and 'enso_lag' not in b.columns:
            b=b.rename(columns={'lag':'enso_lag'})
        else:raise ValueError(f'Exposure needs all-lag table, missing {sorted(required-set(b))}')
    b.ORIG_FID=b.ORIG_FID.str.strip()
    b['season']=b.season.astype('string').str.upper().str.strip()
    b['enso_lag']=pd.to_numeric(b.enso_lag,errors='coerce')
    for c in ['ENSO_VPD_beta','ENSO_SM_beta']:
        b[c]=pd.to_numeric(b[c],errors='coerce')
    if b.duplicated(['ORIG_FID','season','enso_lag']).any():raise ValueError('Duplicate exposure keys')
    b=b.loc[b.enso_lag.isin(lags)&b.season.isin(['DJF','MAM','JJA','SON']) &
            np.isfinite(b.ENSO_VPD_beta)&np.isfinite(b.ENSO_SM_beta)].copy()
    n_expected=4*len(lags)
    counts=b.groupby('ORIG_FID',sort=False).size()
    eligible=set(counts.index[counts.eq(n_expected)])
    if not eligible:raise ValueError('No GUB with all required seasonal × lag beta estimates')
    print(f'Complete exposure IDs across {len(lags)} lags: {len(eligible):,}',flush=True)
    return b, eligible

def prep_common(a,q,allowed,beta_one,geo):
    panel=base.read_panel(a.panel)
    panel=panel.loc[panel.ORIG_FID.isin(allowed)].copy()
    # Quality baseline computed from original observed areas BEFORE any FE estimation.
    # Crucially, unqualified historical records are NOT allowed into new denominator.
    pq=panel[KEY+['greenspace area']].merge(q,on=KEY,how='left',validate='one_to_one',indicator=True)
    if (~pq['_merge'].eq('both')).any():
        raise ValueError(f'Quality missing on {int((~pq._merge.eq("both")).sum()):,} panel records')
    pq['q2_pass']=pq['q2_pass'].fillna(False)
    historical=pq.loc[pq.year.between(2019,2025)&pq.q2_pass].copy()
    refs=historical.groupby(['ORIG_FID','month'],sort=False)['greenspace area'].agg(['mean','count']).reset_index()
    refs=refs.rename(columns={'mean':'ref_Q3','count':'n_ref_Q3'})
    panel_rows=pq[['ORIG_FID','year','month','greenspace area','q2_pass']]
    del historical,pq
    gc.collect()
    # Reuse historical climate standardization and original vegetation reference
    # from the original published model. keep SAME base eligible GUBs for A and B.
    original, _=base.prepare(panel,beta_one,geo,require_balanced=True,min_reference_area=a.min_reference_area)
    # Merging original observed area back is necessary to calculate the Q3 outcome.
    prepared=original.merge(panel_rows,on=KEY,how='left',validate='one_to_one')
    prepared=prepared.merge(refs,on=['ORIG_FID','month'],how='left',validate='many_to_one')
    del original,panel_rows,panel
    gc.collect()
    before=len(prepared)
    regional_before=prepared.groupby('subregion',dropna=False).agg(n_obs_before=('ORIG_FID','size'),n_gub_before=('ORIG_FID','nunique')).reset_index()
    prepared=prepared.loc[prepared.q2_pass & prepared.n_ref_Q3.ge(a.min_hist) &
                          prepared.ref_Q3.ge(a.min_reference_area) &
                          np.isfinite(prepared.ref_Q3)].copy()
    if prepared.empty:raise ValueError('No eligible Q3 data; inspect quality thresholds')
    prepared['veg_y_original']=prepared.veg_y
    prepared['veg_y_Q3']=100*(prepared['greenspace area']-prepared.ref_Q3)/prepared.ref_Q3
    if not np.isfinite(prepared.veg_y_Q3).all():raise ValueError('Nonfinite Q3 anomalies')
    # Every historical reference is a mean of qualified years, so historical
    # within-group Q3 vegetation anomaly averages zero on qualified observations.
    audit=prepared.groupby(['year','month']).agg(n_obs=('ORIG_FID','size'),n_gub=('ORIG_FID','nunique'),
                   n_spatial_blocks=('block5','nunique'),median_original=('veg_y_original','median'),
                   median_Q3=('veg_y_Q3','median')).reset_index()
    regional_after=prepared.groupby('subregion',dropna=False).agg(n_obs_after=('ORIG_FID','size'),n_gub_after=('ORIG_FID','nunique')).reset_index()
    region_audit=regional_before.merge(regional_after,on='subregion',how='left')
    region_audit[['n_obs_after','n_gub_after']]=region_audit[['n_obs_after','n_gub_after']].fillna(0).astype(int)
    region_audit['obs_retained_share']=region_audit.n_obs_after/region_audit.n_obs_before
    ref_stats=pd.DataFrame([{
        'n_original_sample_obs':before,'n_q3_obs':len(prepared),
        'n_q3_gub':prepared.ORIG_FID.nunique(),
        'n_q3_blocks':prepared.block5.nunique(),
        'n_gub_calendar_month_reference':prepared[['ORIG_FID','month']].drop_duplicates().shape[0],
        'min_historical_qualified_years':a.min_hist,
        'mean_historical_qualified_years':float(prepared[['ORIG_FID','month','n_ref_Q3']].drop_duplicates().n_ref_Q3.mean()),
        'median_2026_original_anomaly':float(prepared.loc[prepared.year.eq(2026),'veg_y_original'].median()),
        'median_2026_Q3_anomaly':float(prepared.loc[prepared.year.eq(2026),'veg_y_Q3'].median()),
        'same_observation_rows_across_lags':True
    }])
    return prepared,audit,ref_stats,region_audit


def plot_main_lags(subset,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    model_order=['H0_global_time_FE','H1_global_time_FE','H0_region_time_FE']
    fig,axes=plt.subplots(2,3,figsize=(12.2,6.5),sharex=True,layout='constrained')
    for j,term in enumerate(['HV','HS']):
        for k,model in enumerate(model_order):
            ax=axes[j,k]
            for label,color,shift in [('original_reference','#7C8794',-.075),('Q3_reference','#2D7B7C',.075)]:
                z=subset.loc[(subset.term==term)&(subset.model==model)&
                             (subset.outcome_reference==label)].sort_values('lag')
                if z.empty:continue
                x=z.lag.to_numpy(dtype=float)+shift
                y=z.estimate.to_numpy(dtype=float)
                lo=y-z.ci_low.to_numpy(dtype=float)
                hi=z.ci_high.to_numpy(dtype=float)-y
                ax.errorbar(x,y,yerr=[lo,hi],marker='o',capsize=3,color=color,
                            linewidth=1.3,markersize=4,label='Original reference' if label.startswith('original') else 'Q3 reference')
            ax.axhline(0,color='gray',linestyle='--',linewidth=.8)
            ax.set_xticks([0,1,2,3]);ax.set_xlabel('RONI lag (months)')
            if k==0:ax.set_ylabel(f'{term} coefficient (pp / SD)')
            if j==0:ax.set_title(['H0: ENSO exposure','H1: + actual climate','H0: subregion × month FE'][k],fontsize=10)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,fontsize=9)
    fig.savefig(out/'Figure_Q3_H_lag_comparison.png',dpi=220)
    fig.savefig(out/'Figure_Q3_H_lag_comparison.pdf')
    plt.close(fig)

def run_models(df,a,all_beta,roni,out):
    all_results=[];scale_rows=[]
    constant_columns=['ORIG_FID','year','month','veg_y_original','veg_y_Q3','vpd_z','soil_moisture_z','temp_z','precip_z','D2026','ym','block5','subregion']
    common=df[constant_columns].copy()
    unique_rows=len(common)
    for lag in a.lags:
        print(f'Lag {lag}: constructing original/Q3 paired models',flush=True)
        current=all_beta.loc[all_beta.enso_lag.eq(lag),['ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta']]
        d=common.copy()
        d['season']=d.month.map(base.SEASONS)
        d=d.merge(current,on=['ORIG_FID','season'],how='left',validate='many_to_one')
        if not (np.isfinite(d.ENSO_VPD_beta).all() and np.isfinite(d.ENSO_SM_beta).all()):
            raise ValueError(f'Lag {lag} exposure missing after matched-sample check')
        d,scales,_=hmod.attach_enso(d,roni,lag)
        scales['lag']=lag;scale_rows.append(scales)
        for refname,refcol in [('original_reference','veg_y_original'),('Q3_reference','veg_y_Q3')]:
            d['veg_y']=d[refcol]
            for kind,region_time in [('H0',False),('H1',False),('H0',True)]:
                result=hmod.fit_custom(d,kind,region_time=region_time,twoway=True)
                result['lag']=lag
                result['outcome_reference']=refname
                all_results.append(result)
                # Save a tidy model table for easier reporting
                modelname=result.model.iloc[0]
                result.to_csv(out/f'{refname}_lag{lag}_{modelname}.csv',index=False)
        del d
        gc.collect()
    tab=pd.concat(all_results,ignore_index=True)
    tab.to_csv(out/'H_all_lags_Q3_vs_original_same_sample.csv',index=False)
    pd.concat(scale_rows,ignore_index=True).to_csv(out/'H_exposure_scales_by_lag.csv',index=False)
    assert tab.n_obs.eq(unique_rows).all(), 'Lag sample changed unexpectedly'
    # paired comparison calculated from same GUB-month rows, not original full vs Q3
    subset=tab[tab.term.isin(['HV','HS'])].copy()
    pivot=subset.pivot_table(index=['lag','model','term'],columns='outcome_reference',values=['estimate','ci_low','ci_high','p'],aggfunc='first')
    pivot.columns=[f'{k}_{v}' for k,v in pivot.columns]
    pivot=pivot.reset_index().sort_values(['model','term','lag'])
    pivot.to_csv(out/'H_main_comparison_for_SI.csv',index=False)
    plot_main_lags(subset,out)
    return tab

def run_m_models(df,a,all_beta,out):
    if 1 not in a.lags:
        print('Skipping M0/M1: lag1 not included in --lags',flush=True);return
    print('M0/M1 original vs Q3 same-sample check at lag1',flush=True)
    beta=all_beta.loc[all_beta.enso_lag.eq(1),['ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta']]
    d=df.copy()
    # base.prepare previously set lag-specific EV/ES from lag1 exposure in prep_common.
    # Existing climate variables and FE definitions are unchanged by Q3.
    results=[]
    for label,col in [('original_reference','veg_y_original'),('Q3_reference','veg_y_Q3')]:
        d['veg_y']=d[col]
        for kind in ['baseline','exposure']:
            model=base.fit(d,kind,twoway=True)
            tab=model['tidy'].copy()
            tab['model']='M0' if kind=='baseline' else 'M1'
            tab['outcome_reference']=label
            tab['lag']=1
            tab['n_obs']=model['n'];tab['n_gub']=model['units'];tab['n_block5']=model['clusters']
            results.append(tab)
            tab.to_csv(out/f'{label}_{tab.model.iloc[0]}_twoway_coefficients.csv',index=False)
    pd.concat(results,ignore_index=True).to_csv(out/'M0_M1_Q3_vs_original_same_sample.csv',index=False)

def main():
    a=args()
    if len(set(a.lags))!=len(a.lags):raise ValueError('Duplicate lags')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    print('Reading monthly quality',flush=True)
    q=load_quality(a.quality,a)
    print('Reading historical ENSO exposure',flush=True)
    beta,ids=common_beta_keys(a.exposure,a.lags)
    geo=base.load_geo(a.geo)
    beta1=beta.loc[beta.enso_lag.eq(1),['ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta']]
    if beta1.empty:
        # Arbitrary comparison lag used only to determine common original sample
        first=a.lags[0]
        beta1=beta.loc[beta.enso_lag.eq(first),['ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta']]
        if a.also_m:raise ValueError('--also-m requires lag1 included')
    d,month_audit,stat,region_audit=prep_common(a,q,ids,beta1,geo)
    month_audit.to_csv(out/'Q3_retention_by_year_month.csv',index=False)
    stat.to_csv(out/'Q3_reference_sample_audit.csv',index=False)
    region_audit.to_csv(out/'Q3_retention_by_subregion.csv',index=False)
    print('Q3 samples:',len(d),'rows;',d.ORIG_FID.nunique(),'GUB;',d.block5.nunique(),'spatial clusters',flush=True)
    roni=hmod.roni_series(a.roni)
    run_models(d,a,beta,roni,out)
    if a.also_m:run_m_models(d,a,beta,out)
    print('DONE. Key tables:',out/'H_main_comparison_for_SI.csv',out/'Q3_reference_sample_audit.csv',flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global urban vegetation and ENSO-related hydroclimatic exposure, 2019-2026.

Model H0: vegetation anomaly ~ historical ENSO-predicted VPD & SM anomalies
Model H1: H0 + actual VPD, SM, temperature, precipitation anomalies
Model H2: H0 + GUB FE + (UN subregion x complete year-month) FE [strict check]
Model H3: H1 + predicted exposure x 2026 indicator [exploratory]
Optional: individual subregion fits of H0/H1 with GUB and complete year-month FE.

Historical predictions: beta(GUB, season, lag) * [RONI(year,month-lag) -
mean RONI(2019-2025,same month-lag)]. A within-unit same-calendar-month
reference is used for vegetation and observed climate variables.

Coefficients are associations, not ENSO attribution or causal mediation.

Requires sibling 'enso_formal_interaction_models.py' in the same folder.
Install: pip install numpy pandas scipy matplotlib
"""
import argparse
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import t as tdist

import enso_formal_interaction_models as base


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--panel', required=True, help='Original 2019-2026 GUB panel CSV')
    ap.add_argument('--exposure', required=True, help='Historical ENSO beta CSV (lag 1, or all-lag file)')
    ap.add_argument('--geo', required=True, help='GUB_Country_Region_Lookup.csv')
    ap.add_argument('--roni', required=True, help='Monthly RONI CSV: year/month/roni or YR/MTH/ANOM')
    ap.add_argument('--out',required=True)
    ap.add_argument('--lag',type=int,choices=[0,1,2,3],default=1)
    ap.add_argument('--min-reference-area',type=float,default=10000)
    ap.add_argument('--regional-models',action='store_true')
    ap.add_argument('--min-region-blocks',type=int,default=30)
    ap.add_argument('--twoway',action='store_true',help='5-degree spatial block x year-month clustering')
    ap.add_argument('--allow-unbalanced',action='store_true',help='Allow missing months (not default)')
    ap.add_argument('--quality',default=None,help='Optional GUB x year x month SCL quality CSV')
    ap.add_argument('--min-clear',type=float,default=70.,help='Minimum clear_coverage_pct on 0-100 scale')
    ap.add_argument('--max-snow',type=float,default=10.,help='Maximum snow_any_pct on 0-100 scale')
    return ap.parse_args()


def roni_series(path):
    df=pd.read_csv(path)
    lc={str(c).lower().strip():c for c in df.columns}
    def get(choices):
        for x in choices:
            if x in lc:return lc[x]
        raise ValueError('Missing RONI field; need year/month/roni or YR/MTH/ANOM. Columns: '+str(list(df.columns)))
    yc,mc,rc=get(['year','yr']),get(['month','mth']),get(['roni','anom','value'])
    df=df[[yc,mc,rc]].rename(columns={yc:'year',mc:'month',rc:'roni'})
    for c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna()
    df[['year','month']]=df[['year','month']].astype('int32')
    if df.duplicated(['year','month']).any():raise ValueError('RONI duplicate year-month; use exactly one RONI per month.')
    ps=pd.PeriodIndex(pd.to_datetime({'year':df['year'], 'month':df['month'], 'day':1}),freq='M')
    return pd.Series(df.roni.to_numpy('float64'),index=ps).sort_index()


def attach_enso(df, roni, lag):
    # 2019-2026 each month Jan-Sep: these are year-month fixed-effect common values.
    keys=df[['year','month']].drop_duplicates().sort_values(['year','month']).copy()
    periods=pd.PeriodIndex(pd.to_datetime({'year':keys['year'], 'month':keys['month'], 'day':1}),freq='M')-lag
    keys['roni_lagged']=roni.reindex(periods).to_numpy()
    if keys.roni_lagged.isna().any():
        raise ValueError('RONI monthly series does not cover all year-months after lag; missing: '+
                         str(keys.loc[keys.roni_lagged.isna(),['year','month']].values.tolist()))
    baseline=keys[keys.year.between(2019,2025)].groupby('month')['roni_lagged'].agg(['mean','count'])
    if not baseline['count'].eq(7).all():raise ValueError('Need all 7 reference years of RONI for each month')
    keys['roni_ref']=keys.month.map(baseline['mean'])
    keys['roni_delta']=keys.roni_lagged-keys.roni_ref
    df=df.merge(keys,on=['year','month'],how='left',validate='many_to_one')
    df['H_V_raw']=df.ENSO_VPD_beta.to_numpy('float64')*df.roni_delta.to_numpy('float64')
    df['H_S_raw']=df.ENSO_SM_beta.to_numpy('float64')*df.roni_delta.to_numpy('float64')
    scales=[]
    # historical SD of expected anomalies, NOT mean beta and NOT the climate anomaly SD.
    for raw,short in [('H_V_raw','HV'),('H_S_raw','HS')]:
        sd=float(df.loc[df.year.between(2019,2025),raw].std(ddof=1))
        if not np.isfinite(sd) or sd<1e-10:raise ValueError(f'{raw} has zero historical variation: verify beta/RONI')
        df[short]=df[raw]/sd  # zero refers to reference RONI; intercepts handled by FEs
        scales.append({'predictor':raw,'historic_sd':sd,'units':'kPa for VPD; m3/m3 for SM','lag':lag})
    return df, pd.DataFrame(scales),keys


def make_x(df, kind):
    x={'HV':df.HV.to_numpy('float64'),'HS':df.HS.to_numpy('float64')}
    if kind in ['H1','H3']:
        for n,col in [('V','vpd_z'),('S','soil_moisture_z'),('T','temp_z'),('P','precip_z')]:
            x[n]=df[col].to_numpy('float64')
    if kind=='H3':
        d=df.D2026.to_numpy('float64')
        x['HV_D']=x['HV']*d
        x['HS_D']=x['HS']*d
        for n in ['V','S','T','P']:x[n+'_D']=x[n]*d
    return x


def fit_custom(df,kind,region_time=False,twoway=False):
    x=make_x(df,kind)
    names=list(x)
    X=np.column_stack([x[k] for k in names])
    y=df.veg_y.to_numpy('float64')
    groups=pd.factorize(df.ORIG_FID,sort=False)[0]
    if region_time:
        # Account for shocks common to all GUBs in same UN subregion and month.
        time=pd.factorize(df.subregion.astype('str')+'|'+df.ym.astype('str'),sort=False)[0]
    else:
        time=pd.factorize(df.ym,sort=False)[0]
    yy,xx,it=base.demean_two_way(y,X,groups,time)
    gram=xx.T@xx
    eig=np.linalg.eigvalsh(gram)
    if not np.isfinite(eig).all() or eig.min()<=eig.max()*1e-10:
        raise ValueError(f'{kind} nearly singular. Smallest eigenvalue ratio={eig.min()/eig.max():.3g}')
    b=np.linalg.solve(gram,xx.T@yy)
    res=yy-xx@b
    inv=np.linalg.inv(gram)
    meat,nblock=base.cluster_meat(xx,res,df.block5)
    n=len(df);k=len(b)
    fac=(nblock/(nblock-1))*(n-1)/(n-k)
    var=inv@(meat*fac)@inv
    dfree=nblock-1
    if twoway:
        mt,nt=base.cluster_meat(xx,res,df.ym)
        m_bt,ntb=base.cluster_meat(xx,res,df.block5.astype('str')+'|'+df.ym.astype('str'))
        c_t=(nt/(nt-1))*(n-1)/(n-k)
        c_bt=(ntb/(ntb-1))*(n-1)/(n-k)
        var=inv@(meat*fac+mt*c_t-m_bt*c_bt)@inv
        dfree=min(nblock,nt)-1
    diag=np.diag(var)
    if np.any(diag<0):
        warnings.warn('Negative diagonal variance in two-way covariance; returning NaN SE on affected terms')
    se=np.sqrt(np.maximum(diag,0))
    p=2*tdist.sf(np.abs(b/se),dfree)
    critical=tdist.ppf(.975,dfree)
    result=pd.DataFrame({'term':names,'estimate':b,'se':se,'p':p,
        'ci_low':b-critical*se,'ci_high':b+critical*se})
    result['n_obs']=n
    result['n_gub']=df.ORIG_FID.nunique()
    result['n_block5']=nblock
    result['n_time_groups']=np.unique(time).size
    result['within_r2']=1-(res@res)/(yy@yy)
    result['model']=kind+('_region_time_FE' if region_time else '_global_time_FE')
    result['cluster']='block5_x_yearmonth' if twoway else 'block5'
    result['absorption_iterations']=it
    return result


def bh_fdr(df):
    out=df.copy()
    out['q_bh']=np.nan
    good=out.p.notna()
    if not good.any():return out
    ix=out.index[good]
    p=out.loc[ix,'p'].to_numpy()
    order=np.argsort(p); vals=p[order];m=len(vals)
    adj=np.minimum.accumulate((vals*m/np.arange(1,m+1))[::-1])[::-1]
    mapped=np.empty(m); mapped[order]=np.minimum(adj,1)
    out.loc[ix,'q_bh']=mapped
    return out


def main():
    a=parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    print('[1/5] Reading panel, geographic groups, historical beta and RONI',flush=True)
    panel=base.read_panel(a.panel)
    exposure=base.load_exposure(a.exposure,a.lag)
    geo=base.load_geo(a.geo)
    roni=roni_series(a.roni)
    print('[2/5] Common analytical sample and 2019-2025 references',flush=True)
    df,exposc=base.prepare(panel,exposure,geo,
        require_balanced=not a.allow_unbalanced,min_reference_area=a.min_reference_area)
    df,hs,monthly=attach_enso(df,roni,a.lag)
    if a.quality:
        print('  Applying observation-quality filter after historical reference construction',flush=True)
        q=pd.read_csv(a.quality,dtype={'ORIG_FID':'string'})
        q.columns=q.columns.str.strip()
        required=['ORIG_FID','year','month','clear_coverage_pct','snow_any_pct']
        if any(c not in q for c in required):
            raise ValueError('Quality CSV needs: '+', '.join(required))
        q['ORIG_FID']=q.ORIG_FID.str.strip()
        for c in required[1:]:q[c]=pd.to_numeric(q[c],errors='coerce')
        if q.duplicated(['ORIG_FID','year','month']).any():
            raise ValueError('Quality file contains duplicate GUB-year-month keys')
        df=df.merge(q[required],on=['ORIG_FID','year','month'],how='left',validate='many_to_one')
        df=df.loc[df.clear_coverage_pct.ge(a.min_clear)&df.snow_any_pct.le(a.max_snow)].copy()
        if df.empty:raise ValueError('No observations after quality screening')
    hs.to_csv(out/'00_hydroclimate_scales.csv',index=False)
    monthly.to_csv(out/'00_roni_lagged_reference.csv',index=False)
    audit=pd.DataFrame([dict(lag=a.lag,n_obs=len(df),n_gub=df.ORIG_FID.nunique(),
       n_regions=df.subregion.nunique(),n_spatial_blocks=df.block5.nunique(),
       n_yearmonth=df.ym.nunique(),min_reference_area=a.min_reference_area,
       balanced=not a.allow_unbalanced,quality_filter_applied=bool(a.quality),
       min_clear=a.min_clear if a.quality else np.nan,
       max_snow=a.max_snow if a.quality else np.nan)])
    audit.to_csv(out/'00_sample_audit.csv',index=False)
    print('[3/5] H0: ENSO-associated predicted hydroclimate; H1: observed climate adjustment',flush=True)
    models=[]
    for name,regiontime in [('H0',False),('H1',False),('H0',True),('H3',False)]:
        print('   fitting',name,'region_time',regiontime,flush=True)
        d=fit_custom(df,name,region_time=regiontime,twoway=a.twoway)
        d.to_csv(out/f'01_{d.model.iloc[0]}.csv',index=False)
        models.append(d)
    print('[4/5] Region summaries and optional regional models',flush=True)
    summary=df.groupby('subregion').agg(n_obs=('ORIG_FID','size'),
        n_gub=('ORIG_FID','nunique'),n_blocks=('block5','nunique')).reset_index()
    summary.to_csv(out/'02_region_sample_audit.csv',index=False)
    if a.regional_models:
        subres=[]
        for rec in summary.itertuples(index=False):
            if rec.n_blocks<a.min_region_blocks:continue
            sub=df.loc[df.subregion.eq(rec.subregion)]
            for name in ['H0','H1']:
                print('    region',rec.subregion,name,flush=True)
                try:
                    fitted=fit_custom(sub,name,twoway=a.twoway)
                    fitted['subregion']=rec.subregion
                    subres.append(fitted)
                except (ValueError,np.linalg.LinAlgError) as ex:
                    print('    skipped',rec.subregion,name,':',ex,flush=True)
        if subres:
            r=bh_fdr(pd.concat(subres,ignore_index=True))
            r.to_csv(out/'03_subregion_hydroclimate_models.csv',index=False)
    combined=pd.concat(models,ignore_index=True)
    combined.to_csv(out/'01_all_global_hydroclimate_models.csv',index=False)
    print('[5/5] Finished. Compare H0/H1/H0_region_time_FE, and check H3 as exploratory.',flush=True)
    print(audit.to_string(index=False))
    print(combined.loc[combined.term.isin(['HV','HS','HV_D','HS_D']),
        ['model','term','estimate','ci_low','ci_high','p']].to_string(index=False))

if __name__=='__main__':main()

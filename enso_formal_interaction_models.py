#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal GUB 2019-2026 ENSO-exposure x 2026 x climate interaction models.

Inputs:
  --panel: ORIG_FID, year, month, greenspace area, temp, precip, soil_moisture, vpd
  --exposure: ORIG_FID, season, ENSO_VPD_beta, ENSO_SM_beta (lag1 file or long all16)
  --geo: ORIG_FID, longitude, latitude, subregion [and optionally match_status]

Fits GUB and year-month fixed effects using alternating within transforms (no FE dummies).
Inference: 5-degree spatial-block CR1 as primary; optional 5deg x year-month
multiway clustering sensitivity. Coefficients are statistical associations, NOT
identified causal ENSO effects. Exposures use historical 1981-2018 estimates;
all 2019-2026 months are used for the slope-change model, not just 2026.

Install: pip install numpy pandas scipy matplotlib
"""
import argparse
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEASONS = {1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON'}
CLIMATE = ['temp','precip','soil_moisture','vpd']
# Original supplementary-data pooled historical anomaly standard deviations:
SDS = {'temp':1.21176,'precip':48.8825,'soil_moisture':0.0347669,'vpd':0.182707}


def read_panel(path):
    want = set(['ORIG_FID','year','month','greenspace area'] + CLIMATE)
    df = pd.read_csv(path, dtype={'ORIG_FID':'string'},
                     usecols=lambda x: x.strip() in want, low_memory=False)
    df.columns = df.columns.str.strip()
    missing = want-set(df.columns)
    if missing: raise ValueError(f'Panel missing columns: {sorted(missing)}')
    df['ORIG_FID'] = df.ORIG_FID.str.strip()
    for c in ['year','month','greenspace area']+CLIMATE:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.loc[df.year.between(2019,2026)&df.month.between(1,9)].copy()
    if df.ORIG_FID.isna().any() or df[['year','month']].isna().any().any():
        raise ValueError('Missing panel ID/year/month')
    df[['year','month']]=df[['year','month']].astype('int16')
    if df.duplicated(['ORIG_FID','year','month']).any():
        raise ValueError('Duplicate panel ORIG_FID/year/month combinations: resolve in source')
    return df


def load_exposure(path, lag):
    d=pd.read_csv(path,dtype={'ORIG_FID':'string'},low_memory=False)
    d.columns=d.columns.str.strip()
    req={'ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta'}
    if not req.issubset(d): raise ValueError(f'Exposure missing fields: {sorted(req-set(d))}')
    d.ORIG_FID=d.ORIG_FID.str.strip()
    d.season=d.season.astype('string').str.upper().str.strip()
    if 'lag' in d:
        d=d[pd.to_numeric(d.lag,errors='coerce').eq(lag)].copy()
    elif 'enso_lag' in d:
        d=d[pd.to_numeric(d.enso_lag,errors='coerce').eq(lag)].copy()
    elif lag != 1:
        raise ValueError('Exposure file has no lag label; only lag1 file is supported without lag column')
    for c in ['ENSO_VPD_beta','ENSO_SM_beta']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
        d[c]=d[c].replace([np.inf,-np.inf],np.nan)
    if d.duplicated(['ORIG_FID','season']).any():
        raise ValueError('Duplicate ORIG_FID-season in historical exposure file')
    return d[['ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta']]


def load_geo(path):
    d=pd.read_csv(path,dtype={'ORIG_FID':'string'},low_memory=False)
    d.columns=d.columns.str.strip()
    req={'ORIG_FID','longitude','latitude','subregion'}
    if not req.issubset(d): raise ValueError(f'Geography missing: {sorted(req-set(d))}')
    d.ORIG_FID=d.ORIG_FID.str.strip()
    if d.ORIG_FID.duplicated().any(): raise ValueError('Duplicate ORIG_FID in geo lookup')
    for c in ['longitude','latitude']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['subregion']=d['subregion'].astype('string').str.strip()
    keep=['ORIG_FID','longitude','latitude','subregion']
    if 'match_status' in d: keep+=['match_status']
    return d[keep]


def prepare(panel, expos, geo, require_balanced=True, min_reference_area=10000):
    # Strong consistency: historical baseline reference must have all seven years.
    g=panel.groupby(['ORIG_FID','month'],sort=False,observed=True)
    refs=panel[panel.year.between(2019,2025)].groupby(['ORIG_FID','month'],sort=False,observed=True)
    stats=refs[['greenspace area']+CLIMATE].agg(['mean','count'])
    stats.columns=[f'{v}_ref_{s}' for v,s in stats.columns]
    df=panel.merge(stats.reset_index(),on=['ORIG_FID','month'],how='left',validate='many_to_one')
    for v in CLIMATE:
        if (SDS[v] <=0): raise ValueError(f'Invalid SD for {v}')
        df[v+'_z']=(df[v]-df[v+'_ref_mean'])/SDS[v]
    vegref=df['greenspace area_ref_mean']
    df['veg_y']=100*(df['greenspace area']-vegref)/vegref
    ok=vegref.ge(min_reference_area) & df['greenspace area_ref_count'].eq(7)
    for v in CLIMATE: ok &= df[v+'_ref_count'].eq(7)
    for v in CLIMATE+['greenspace area']:
        ok &= np.isfinite(df[v])
    for v in CLIMATE:
        ok &= np.isfinite(df[v+'_z'])
    ok &= np.isfinite(df['veg_y'])
    df=df.loc[ok, ['ORIG_FID','year','month','veg_y']+[v+'_z' for v in CLIMATE]].copy()
    if require_balanced:
        valid=df.groupby('ORIG_FID',sort=False).size()
        ids=valid.index[valid.eq(72)]
        df=df.loc[df.ORIG_FID.isin(ids)].copy()
    df['season']=df.month.map(SEASONS)
    df=df.merge(expos,on=['ORIG_FID','season'],how='left',validate='many_to_one')
    df=df.merge(geo,on='ORIG_FID',how='left',validate='many_to_one')
    if 'match_status' in df:
        good=df.match_status.eq('matched').fillna(False)
    else:
        good=pd.Series(True,index=df.index)
    ok=(good & df.longitude.between(-180,180)&df.latitude.between(-90,90)
        & df.subregion.notna() & ~df.subregion.isin(['not found','UNMATCHED','Unknown',''])
        & np.isfinite(df.ENSO_VPD_beta) & np.isfinite(df.ENSO_SM_beta))
    df=df.loc[ok].copy()
    if require_balanced:
        valid=df.groupby('ORIG_FID',sort=False).size()
        ids=valid.index[valid.eq(72)]
        df=df.loc[df.ORIG_FID.isin(ids)].copy()
    df['D2026']=(df.year==2026).astype('float64')
    df['ym']=(df.year.astype('int32')*100+df.month.astype('int32')).astype('string')
    x=np.floor((df.longitude.to_numpy()+180)/5).astype('int16')
    y=np.floor((df.latitude.to_numpy()+90)/5).astype('int16')
    df['block5']=(x.astype('int32')*100+y.astype('int32')).astype('int32')
    if df.empty: raise ValueError('All rows excluded. Check reference areas, exposure joins and geographic match_status')
    # Standardize exposures across distinct GUB-season values, not row-weighted by season length.
    unique=df[['ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta']].drop_duplicates(['ORIG_FID','season'])
    scales={}
    for name,source in [('EV','ENSO_VPD_beta'),('ES','ENSO_SM_beta')]:
        mean=float(unique[source].mean()); sd=float(unique[source].std(ddof=1))
        if sd<1e-12: raise ValueError(f'No variation in historical exposure {source}')
        df[name]=(df[source]-mean)/sd
        scales[name]={'raw_field':source,'mean':mean,'sd':sd}
    # Export scalable diagnostic groups, avoid massive panel export by default.
    df=df.reset_index(drop=True)
    return df, scales


def columns(df, kind):
    """Construct explicit interaction columns so coefficient names are stable."""
    x={}
    for v,raw in [('V','vpd_z'),('S','soil_moisture_z'),('T','temp_z'),('P','precip_z')]:
        x[v]=df[raw].to_numpy(dtype='float64')
    D=df.D2026.to_numpy(dtype='float64')
    for v in 'VSTP': x[v+'_D']=x[v]*D
    if kind=='exposure':
        EV=df.EV.to_numpy(dtype='float64')
        ES=df.ES.to_numpy(dtype='float64')
        x['EV']=EV; x['ES']=ES
        x['V_EV']=x['V']*EV; x['S_ES']=x['S']*ES
        x['D_EV']=D*EV; x['D_ES']=D*ES
        x['V_D_EV']=x['V']*D*EV; x['S_D_ES']=x['S']*D*ES
    return x


def demean_two_way(y, X, gids, tids, tol=1e-9, max_iter=150):
    """Iterative absorption of two sets of FE. No dense dummy matrices."""
    z=np.column_stack((y,X)).astype('float64',copy=True)
    for it in range(max_iter):
        old=z.copy()
        for group in [gids,tids]:
            n=int(group.max())+1
            counts=np.bincount(group,minlength=n)
            for j in range(z.shape[1]):
                means=np.bincount(group,weights=z[:,j],minlength=n)/counts
                z[:,j]-=means[group]
        change=np.max(np.abs(z-old))
        if change<tol:
            return z[:,0], z[:,1:], it+1
    warnings.warn('FE absorption did not reach tolerance; inspect residualization')
    return z[:,0],z[:,1:],max_iter


def cluster_meat(X, resid, groups):
    g,_=pd.factorize(groups,sort=False)
    k=X.shape[1]; ng=int(g.max())+1
    score=np.column_stack([np.bincount(g,weights=X[:,j]*resid,minlength=ng) for j in range(k)])
    return score.T@score,ng


def fit(df, kind, cluster='block5', twoway=False):
    x=columns(df,kind)
    names=list(x)
    X=np.column_stack([x[k] for k in names])
    y=df.veg_y.to_numpy(dtype='float64')
    ids=pd.factorize(df.ORIG_FID,sort=False)[0]
    ts=pd.factorize(df.ym,sort=False)[0]
    yw,Xw,niters=demean_two_way(y,X,ids,ts)
    # Check collinearity: use SVD on KxK gram for moderate K (~16), fail instead of silently dropping terms.
    gram=Xw.T@Xw
    if not np.isfinite(gram).all(): raise ValueError('Nonfinite model crossproducts')
    eig=np.linalg.eigvalsh(gram)
    if eig[0]<=1e-10*eig[-1]:
        raise ValueError(f'Singular/near-singular {kind} model: min/max eig={eig[0]/eig[-1]:.3g}')
    b=np.linalg.solve(gram,Xw.T@yw)
    resid=yw-Xw@b
    inv=np.linalg.inv(gram)
    m,g=cluster_meat(Xw,resid,df[cluster])
    n=len(df); k=len(b); df1=n-k
    if g<15: warnings.warn(f'Only {g} spatial clusters. Inference fragile.')
    correction=(g/(g-1))*((n-1)/df1) if g>1 and df1>0 else np.nan
    vc=inv@(m*correction)@inv
    dfree=g-1
    if twoway:
        mt,gt=cluster_meat(Xw,resid,df.ym)
        joint=df[cluster].astype(str)+'|'+df.ym.astype(str)
        mj,gj=cluster_meat(Xw,resid,joint)
        vc=inv@(m*correction+mt*(gt/(gt-1))*((n-1)/df1)-mj*(gj/(gj-1))*((n-1)/df1))@inv
        dfree=min(g,gt)-1
    if np.diag(vc).min()<0:
        warnings.warn('Two-way covariance has negative diagonal entries; do not use this inference')
        raise ValueError('Invalid covariance diagonal')
    se=np.sqrt(np.diag(vc))
    t=b/se
    p=2*student_t.sf(abs(t),df=dfree)
    ci=student_t.ppf(0.975,dfree)
    tidy=pd.DataFrame({'term':names,'estimate':b,'se':se,'p':p,'ci_low':b-ci*se,'ci_high':b+ci*se})
    return {'names':names,'coef':b,'cov':vc,'tidy':tidy,'n':n,'units':df.ORIG_FID.nunique(),
            'clusters':g,'times':df.ym.nunique(),'df':dfree,'within_r2':1-(resid@resid)/(yw@yw),
            'iterations':niters}


def contrast(model, terms, label, exposure_value=np.nan, variable='V'):
    vec=np.zeros(len(model['names']))
    for term,coef in terms.items():
        if term not in model['names']: raise ValueError(f'Missing term {term}')
        vec[model['names'].index(term)]=coef
    est=float(vec@model['coef']); se=float(np.sqrt(max(0,vec@model['cov']@vec)))
    crit=student_t.ppf(.975,model['df'])
    return dict(label=label,variable=variable,exposure_z=exposure_value,estimate=est,
                se=se,ci_low=est-crit*se,ci_high=est+crit*se,
                p=2*student_t.sf(abs(est/se),df=model['df']) if se>0 else np.nan)


def slopes(model,kind):
    rows=[]
    for var,exp in [('V','EV'),('S','ES')]:
        if kind=='baseline':
            for nm,terms in [('historical',{var:1}),('2026',{var:1,var+'_D':1}),
                             ('change',{var+'_D':1})]:
                rows.append(contrast(model,terms,nm,variable=var))
        else:
            for e in [-1,0,1]:
                for nm,terms in [
                    ('historical',{var:1,var+'_'+exp:e}),
                    ('2026',{var:1,var+'_'+exp:e,var+'_D':1,var+'_D_'+exp:e}),
                    ('change',{var+'_D':1,var+'_D_'+exp:e})]:
                    rows.append(contrast(model,terms,nm,exposure_value=e,variable=var))
    return pd.DataFrame(rows)


def run_one(df,kind,out,name,twoway=False):
    print(f'Fitting {name}: n={len(df):,}, GUBs={df.ORIG_FID.nunique():,}, 5deg blocks={df.block5.nunique():,}',flush=True)
    model=fit(df,kind,twoway=twoway)
    label=f'{name}_{"twoway" if twoway else "block5"}'
    model['tidy'].assign(model=label,n=model['n'],n_clusters=model['clusters']).to_csv(
        out/f'{label}_coefficients.csv',index=False)
    sl=slopes(model,kind)
    sl.assign(model=label).to_csv(out/f'{label}_conditional_slopes.csv',index=False)
    return model,sl


def chart(sl,out):
    ss=sl[(sl.label=='change')&(sl.exposure_z.notna())].copy()
    if ss.empty:return
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for a,(v,title) in zip(ax,[('V','VPD'),('S','Soil moisture')]):
        z=ss[ss.variable==v].sort_values('exposure_z')
        a.errorbar(z.exposure_z,z.estimate,yerr=[z.estimate-z.ci_low,z.ci_high-z.estimate],
                   fmt='o-',capsize=4)
        a.axhline(0,color='grey',linestyle='--',linewidth=1)
        a.set(title=f'{title}: 2026 minus historical slope',xlabel='Historical ENSO exposure (SD)',
              ylabel='Slope difference (vegetation pp / climate SD)')
        a.set_xticks([-1,0,1])
    fig.savefig(out/'figure_conditional_slope_changes.png',dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--panel',required=True)
    p.add_argument('--exposure',required=True)
    p.add_argument('--geo',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--lag',type=int,default=1)
    p.add_argument('--min-reference-area',type=float,default=10000)
    p.add_argument('--allow-unbalanced',action='store_true')
    p.add_argument('--twoway',action='store_true',help='Additional 5deg x year-month clustered inference')
    p.add_argument('--regions',default='',help='Comma-separated preselected UN subregions, or ALL')
    p.add_argument('--min-region-blocks',type=int,default=30)
    p.add_argument('--winsor',action='store_true',help='Sensitivity: clip vegetation outcome at global 1st/99th percentiles')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    panel=read_panel(a.panel)
    expos=load_exposure(a.exposure,a.lag)
    geo=load_geo(a.geo)
    d,scales=prepare(panel,expos,geo,not a.allow_unbalanced,a.min_reference_area)
    audit=pd.DataFrame([{'panel_rows':len(panel),'panel_gubs':panel.ORIG_FID.nunique(),
        'model_rows':len(d),'model_gubs':d.ORIG_FID.nunique(),'months':d.ym.nunique(),
        'spatial_blocks5':d.block5.nunique(),'lag':a.lag,'balanced':not a.allow_unbalanced,
        'min_reference_area_m2':a.min_reference_area,'outcome':'percent change vs 2019-2025 same month'}])
    audit.to_csv(out/'00_sample_audit.csv',index=False)
    pd.DataFrame(scales).T.to_csv(out/'00_exposure_scales.csv')
    print(audit.to_string(index=False),flush=True)
    d['subregion']=d.subregion.astype('string').fillna('UNMATCHED')
    # Make the common global sample explicit. Both models use identical complete cases.
    if a.winsor:
        lo,hi=np.quantile(d.veg_y,[.01,.99]);d['veg_y']=d.veg_y.clip(lo,hi)
        print('Winsorized outcome to:',lo,hi,flush=True)
    m0,s0=run_one(d,'baseline',out,'M0_global')
    m1,s1=run_one(d,'exposure',out,'M1_global')
    chart(s1,out)
    if a.twoway:
        run_one(d,'baseline',out,'M0_global',twoway=True)
        run_one(d,'exposure',out,'M1_global',twoway=True)
    regions=[]
    if a.regions=='ALL': regions=sorted(d.subregion.dropna().unique().tolist())
    elif a.regions: regions=[v.strip() for v in a.regions.split(',') if v.strip()]
    records=[]; regional_slopes=[]
    for idx,r in enumerate(regions):
        sub=d[d.subregion.eq(r)].copy()
        if sub.block5.nunique()<a.min_region_blocks or sub.ORIG_FID.nunique()<100 or sub.ym.nunique()<72:
            records.append({'subregion':r,'status':'skipped','gubs':sub.ORIG_FID.nunique(),
                            'blocks5':sub.block5.nunique()});continue
        try:
            mm=fit(sub,'exposure')
            ss=slopes(mm,'exposure').assign(subregion=r)
            regional_slopes.append(ss)
            records.append({'subregion':r,'status':'estimated','gubs':sub.ORIG_FID.nunique(),
                            'blocks5':sub.block5.nunique(),'within_r2':mm['within_r2']})
            tmp=mm['tidy'].assign(subregion=r,gubs=mm['units'],blocks5=mm['clusters'])
            tmp.to_csv(out/f'region_{idx:02d}_coefficients.csv',index=False)
        except (ValueError,np.linalg.LinAlgError) as e:
            records.append({'subregion':r,'status':'failed','reason':str(e)})
    if regions:
        pd.DataFrame(records).to_csv(out/'03_region_audit.csv',index=False)
        if regional_slopes:
            pd.concat(regional_slopes,ignore_index=True).to_csv(out/'03_regional_conditional_slopes.csv',index=False)
    print('Saved outputs to',out,flush=True)
    print('Warning: observational climate associations; triple interactions do not identify ENSO causation.')
    print('Check 2026 vs historical slopes at E=-1,0,+1, and robustness to lag/quality/space/time clustering.')

if __name__=='__main__': main()

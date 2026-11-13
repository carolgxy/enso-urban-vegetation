#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce publication Figure 1, Figure 2, and focused SI figures on the
EXACT Q3 Sentinel-2 quality sample used in the Q3 common-sample regressions.

Required siblings (included in the package):
  enso_formal_interaction_models.py
  enso_hydroclimate_panel_models.py
  scl_Q3_common_sample_lag_robustness.py

Core outputs: Fig1_Q3.png/pdf, Fig2_Q3.png/pdf, FigS1_quality.png/pdf,
FigS2_sample.png/pdf, FigS3_reference_lags.png/pdf,
FigS4_clustering.png/pdf (if optional input), derived CSVs, figure_audit.csv.

Cartography: Geo lookup contains centroids only. Maps are CENTROID maps and
cannot recreate original GUB polygons without polygon geometries. Optional
--boundaries-shp adds a visual coastline/country-outline background.

All data-derived Figure 1 and Figure 2 region and exposure panels use Q3 filtered sample.
Old region results are intentionally NOT reused. Region models are fitted on
Q3 with lag 3 and dual clusters when --fit-regions is set.
"""
import argparse
from pathlib import Path
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

import enso_formal_interaction_models as base
import enso_hydroclimate_panel_models as hmod
import scl_Q3_common_sample_lag_robustness as qmod

TEAL='#217e7a'; BROWN='#a5643d'; BLUE='#4378a6'; GRAY='#000000'; PURPLE='#7c65a0'; INK='#000000'
COLORS={'H0_global_time_FE':BROWN,'H1_global_time_FE':TEAL,'H0_region_time_FE':BLUE}
LABELS={'H0_global_time_FE':'H0 · GUB + year–month FE',
        'H1_global_time_FE':'H1 · + observed climate',
        'H0_region_time_FE':'H0 · subregion × year–month FE'}
MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--panel',required=True,help='Original full 2019–2026 GUB panel')
    p.add_argument('--quality',required=True,help='SCL_quality_all72.csv, not already filtered')
    p.add_argument('--exposure',required=True,help='enso_exposure_long_all16.csv')
    p.add_argument('--geo',required=True,help='GUB_Country_Region_Lookup.csv')
    p.add_argument('--roni',required=True,help='RONI.csv')
    p.add_argument('--q3-summary',required=True,help='Q3 output H_main_comparison_for_SI.csv')
    p.add_argument('--q3-audit',default=None,help='Q3_reference_sample_audit.csv; validates n_obs and n_GUB')
    p.add_argument('--quality-monthly',default=None,help='01_monthly_quality_summary.csv (full GUB universe)')
    p.add_argument('--cluster-comparison',default=None,help='single_vs_twoway_comparison.csv')
    p.add_argument('--q3-m',default=None,help='optional Q3 output M0_M1_Q3_vs_original_same_sample.csv')
    p.add_argument('--out',required=True)
    p.add_argument('--fit-regions',action='store_true',help='Reestimate Q3 lag3 H0/H1 per UN subregion, two-way clustered')
    p.add_argument('--min-region-blocks',type=int,default=30)
    p.add_argument('--min-hist',type=int,default=5)
    p.add_argument('--min-clear',type=float,default=80)
    p.add_argument('--min-clear2',type=float,default=50)
    p.add_argument('--max-snow',type=float,default=5)
    p.add_argument('--min-reference-area',type=float,default=10000)
    p.add_argument('--boundaries-shp',default=None,help='Optional world outline vector shapefile or GeoJSON; geopandas required')
    p.add_argument('--map-point-size',type=float,default=1.2)
    p.add_argument('--region-results',default=None,help='Existing same-Q3 lag3 region CSV; avoids rerunning regional models. Requires --fit-regions to generate if absent.')
    return p.parse_args()


def style():
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':11,
      'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
      'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
      'axes.edgecolor':'#cbd2d7','axes.labelcolor':INK,'text.color':INK,
      'grid.color':'#edf0f1','savefig.facecolor':'white','pdf.fonttype':42})


def save(fig,out,name):
    for suffix in ['png','pdf']:
        fig.savefig(out/f'{name}.{suffix}',dpi=300 if suffix=='png' else None,
                    bbox_inches='tight',pad_inches=.12)
    plt.close(fig)


def tag(ax, letter):
    ax.text(-.065,1.04,letter,transform=ax.transAxes,fontsize=14,fontweight='bold',va='bottom')


def outlines(ax,shp):
    if shp is None: return
    try:
        import geopandas as gpd
    except ImportError as e:
        raise RuntimeError('To use --boundaries-shp, install geopandas') from e
    world=gpd.read_file(shp)
    if world.crs is not None: world=world.to_crs('EPSG:4326')
    world.boundary.plot(ax=ax,color='#c5ced0',linewidth=.35,zorder=0)


def q3_sample(a):
    # Reuses EXACT reference and sample-construction logic of the existing Q3 code.
    class Opt:pass
    x=Opt();x.min_clear=a.min_clear;x.min_clear2=a.min_clear2;x.max_snow=a.max_snow
    x.min_hist=a.min_hist;x.min_reference_area=a.min_reference_area;x.panel=a.panel
    print('Reading the quality flags and common lag0–3 exposure support...',flush=True)
    quality=qmod.load_quality(a.quality,x)
    beta,eligible=qmod.common_beta_keys(a.exposure,[0,1,2,3])
    geo=base.load_geo(a.geo)
    b1=beta.loc[beta.enso_lag.eq(1),['ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta']]
    print('Reconstructing Q3 outcome and sample (same code as regressions)...',flush=True)
    d,month_audit,summary,region_audit=qmod.prep_common(x,quality,eligible,b1,geo)
    return d,beta,month_audit,summary,region_audit


def validate_audit(summary,filename):
    if not filename:return
    ref=pd.read_csv(filename).iloc[0]
    for c in ['n_q3_obs','n_q3_gub','n_q3_blocks']:
        actual=int(summary.iloc[0][c]);expected=int(ref[c])
        if actual!=expected:raise ValueError(f'Q3 SAMPLE MISMATCH {c}: plotting={actual}, model={expected}. '
                                              'Check all input files, thresholds and ID matching.')
    print('AUDIT PASSED: Q3 n_obs/n_gub/n_blocks exactly match regression.',flush=True)


def figure1(d,out,a):
    now=d.loc[d.year.eq(2026),['ORIG_FID','month','latitude','longitude','subregion',
                                'veg_y_Q3','greenspace area','ref_Q3']].copy()
    if now.empty:raise ValueError('No 2026 data in Q3 sample.')
    now['near_zero']=now.veg_y_Q3.between(-5,5,inclusive='both')
    now['negative']=now.veg_y_Q3.lt(-5)
    now['positive']=now.veg_y_Q3.gt(5)
    # Include per-month varying N (Q2 quality masks produce an UNBALANCED panel).
    months=now.groupby('month').agg(n_obs=('ORIG_FID','size'),n_gub=('ORIG_FID','nunique'),
        median_anomaly=('veg_y_Q3','median'),q25=('veg_y_Q3',lambda z:z.quantile(.25)),
        q75=('veg_y_Q3',lambda z:z.quantile(.75)),negative_share=('negative','mean'),
        near_zero_share=('near_zero','mean'),positive_share=('positive','mean')).reset_index()
    months.to_csv(out/'Fig1_2026_monthly_Q3.csv',index=False)
    region=now.groupby(['subregion','month'],dropna=False).agg(
        n_gub=('ORIG_FID','nunique'),median_anomaly=('veg_y_Q3','median'),
        negative_share=('negative','mean')).reset_index()
    region.to_csv(out/'Fig1_region_monthly_Q3.csv',index=False)
    # Unit average 2026 anomaly; NOT pooled unit-month nor a sum of detected areas.
    unit=now.groupby('ORIG_FID',sort=False).agg(mean_anomaly=('veg_y_Q3','mean'),
                    months_valid=('month','nunique'),subregion=('subregion','first')).reset_index()
    unit.to_csv(out/'Fig1_unit_2026_mean_Q3.csv',index=False)

    fig=plt.figure(figsize=(15.2,10.1),layout='constrained')
    gs=fig.add_gridspec(3,4,height_ratios=[1,1,1.02],hspace=.12,wspace=.12)
    norm=TwoSlopeNorm(vmin=-40,vcenter=0,vmax=40)
    cm=plt.get_cmap('BrBG')
    # BrBG maps negative anomalies to brown and positive anomalies to teal.
    cm=plt.get_cmap('BrBG')
    last_sc=None
    for i,(m,letter) in enumerate(zip([2,4,6,8],'abcd')):
        ax=fig.add_subplot(gs[i//2, (i%2)*2:(i%2+1)*2])
        subset=now.loc[now.month.eq(m)].copy()
        if subset.empty:continue
        outlines(ax,a.boundaries_shp)
        subset=subset.sample(frac=1,random_state=200+m) # deterministic draw order
        last_sc=ax.scatter(subset.longitude,subset.latitude,c=subset.veg_y_Q3,
             cmap=cm,norm=norm,s=a.map_point_size,linewidths=0,alpha=.86,rasterized=True,zorder=2)
        ax.set_xlim(-180,180);ax.set_ylim(-62,85)
        ax.set_xticks([-120,-60,0,60,120]);ax.set_yticks([-30,0,30,60]);ax.grid(alpha=.65,zorder=-1)
        ax.set_title(f'{MONTHS[m-1]} 2026  ·  n={len(subset):,}',pad=7)
        tag(ax,letter)
    # show consistent colorbar for the four maps
    if last_sc is not None:
        cb=fig.colorbar(last_sc,ax=fig.axes[:4],shrink=.65,location='right',pad=.015,
                        extend='both',fraction=.015)
        cb.set_label('Vegetation area anomaly (%)')

    ax=fig.add_subplot(gs[2,0]);tag(ax,'e')
    ax.plot(months.month,months.median_anomaly,color=TEAL,marker='o',lw=2,ms=4)
    ax.fill_between(months.month,months.q25,months.q75,color=TEAL,alpha=.15,label='25th–75th percentile')
    ax.axhline(0,c='#9aa9b0',lw=.9,ls='--')
    ax.set_xticks(range(1,10),MONTHS,rotation=35);ax.set_title('Monthly GUB anomalies');ax.set_ylabel('Anomaly (%)')
    ax.grid(axis='y');ax.legend(frameon=False,fontsize=7,loc='best')

    ax=fig.add_subplot(gs[2,1]);tag(ax,'f')
    m=months.month.to_numpy(); bottom=np.zeros(len(months))
    for label,field,color in [('Negative < −5%','negative_share',BROWN),('Within ±5%','near_zero_share','#d8d9d5'),('Positive > +5%','positive_share',TEAL)]:
        v=months[field].to_numpy()*100
        ax.bar(m,v,bottom=bottom,width=.83,color=color,label=label)
        bottom+=v
    ax.set_xticks(range(1,10),MONTHS,rotation=35);ax.set_ylim(0,100)
    ax.set_ylabel('Share of valid GUB–months (%)');ax.set_title('Anomaly direction')
    ax.legend(fontsize=6.9,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=1)

    ax=fig.add_subplot(gs[2,2:4]);tag(ax,'g')
    # No cherry picking by anomaly; region order by sample coverage.
    regtot=now.groupby('subregion').ORIG_FID.nunique().sort_values(ascending=False)
    order=regtot.index.tolist()
    mat=region.pivot(index='subregion',columns='month',values='median_anomaly').reindex(index=order,columns=range(1,10))
    # Heatmap includes all eligible subregions, with missing months shown grey.
    vals=np.ma.masked_invalid(mat.to_numpy('float64'))
    image=ax.imshow(vals,cmap=cm,norm=norm,aspect='auto',interpolation='nearest')
    ax.set_yticks(np.arange(len(mat)),[str(v) for v in mat.index],fontsize=7)
    ax.set_xticks(range(9),MONTHS,fontsize=8)
    ax.set_title('UN subregion median anomalies')
    ax.tick_params(length=0)
    ax.set_facecolor('#f1f2f2')
    fig.colorbar(image,ax=ax,shrink=.8,pad=.01,label='Anomaly (%)')
    #fig.suptitle('Urban vegetation area anomalies in 2026 · Q3 quality-qualified sample',fontsize=14,fontweight='bold')
    save(fig,out,'Figure1_Q3')
    print('Figure1 prepared; valid 2026 GUB-month sample varies by month.',flush=True)
    return now,months,region


def read_q3_coeff(path):
    t=pd.read_csv(path)
    needed={'lag','model','term','estimate_Q3_reference','ci_low_Q3_reference','ci_high_Q3_reference'}
    if not needed.issubset(t):raise ValueError(f'Q3 coefficient file missing {needed-set(t)}')
    return t


def fit_regions(d,beta,roni,a,out):
    print('Reestimating Q3 lag3 regional H0/H1 (same Q3 rows; 5° × year–month clustering)...',flush=True)
    exposed=beta.loc[beta.enso_lag.eq(3),['ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta']]
    cols=['ORIG_FID','year','month','veg_y_Q3','vpd_z','soil_moisture_z','temp_z','precip_z',
          'D2026','ym','block5','subregion']
    z=d[cols].copy().rename(columns={'veg_y_Q3':'veg_y'})
    z['season']=z.month.map(base.SEASONS)
    z=z.merge(exposed,on=['ORIG_FID','season'],how='left',validate='many_to_one')
    z,_,_=hmod.attach_enso(z,roni,3)
    results=[];audit=[]
    for region,sub in z.groupby('subregion',sort=True):
        blocks=int(sub.block5.nunique());obs=len(sub);units=sub.ORIG_FID.nunique()
        audit.append({'subregion':region,'n_obs':obs,'n_gub':units,'n_blocks5':blocks,
                      'included':blocks>=a.min_region_blocks})
        if blocks<a.min_region_blocks:continue
        for kind in ['H0','H1']:
            try:
                result=hmod.fit_custom(sub,kind,twoway=True)
                result['subregion']=region
                result['lag']=3
                result['reference']='Q3'
                results.append(result)
            except (ValueError,np.linalg.LinAlgError) as err:
                print(f'WARNING: skipped region {region} {kind}: {err}',flush=True)
    pd.DataFrame(audit).to_csv(out/'Fig2_region_sample_audit.csv',index=False)
    regional=pd.concat(results,ignore_index=True) if results else pd.DataFrame()
    if not regional.empty:
        regional=hmod.bh_fdr(regional)
        regional.to_csv(out/'Fig2_region_Q3_lag3_H0_H1_twoway.csv',index=False)
    return regional


def draw_forest(ax, regional):
    """Show all eligible lag-3 Q3 H0 subregions, ordered by point estimate."""
    selection=regional.loc[(regional.model=='H0_global_time_FE')&(regional.term=='HV')].copy()
    required={'subregion','estimate','ci_low','ci_high'}
    if not required.issubset(selection):
        raise ValueError('Regional result missing '+str(required-set(selection)))
    selection=selection.dropna(subset=['estimate','ci_low','ci_high']).sort_values('estimate')
    if selection.empty:
        raise ValueError('No eligible Q3 lag-3 regional HV/H0 rows; use --fit-regions or --region-results')
    y=np.arange(len(selection))
    x=selection.estimate.to_numpy(float)
    errs=np.vstack([x-selection.ci_low.to_numpy(float), selection.ci_high.to_numpy(float)-x])
    if (errs<0).any():raise ValueError('Invalid regional confidence intervals')
    ax.errorbar(x,y,xerr=errs,fmt='o',color=TEAL,ecolor=TEAL,capsize=2.5,
                markersize=4.5,elinewidth=1.25,zorder=3)
    ax.axvline(0,color='#9aa7b0',ls='--',lw=.95)
    ax.set_yticks(y,selection.subregion,fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('HV coefficient (pp / historical SD)')
    ax.set_title('Regional association · lag 3',pad=10)
    ax.grid(axis='x',alpha=.7)
    return selection


def lag_axis(ax,coeff,term='HV',models=('H0_global_time_FE','H0_region_time_FE')):
    # One core lag plot. H1 is NOT suppressed: companion inset and SI show its full CIs.
    for model in models:
        z=coeff.loc[(coeff.model==model)&(coeff.term==term)].sort_values('lag')
        if len(z)!=4:raise ValueError('Missing lag 0-3 coefficient rows for '+model+'/'+term)
        x=z.lag.to_numpy(float)
        y=z.estimate_Q3_reference.to_numpy(float)
        low=z.ci_low_Q3_reference.to_numpy(float)
        high=z.ci_high_Q3_reference.to_numpy(float)
        ax.plot(x,y,color=COLORS[model],lw=2.15,marker='o',markersize=5.4,
                label=LABELS[model],zorder=3)
        ax.errorbar(x,y,yerr=[y-low,high-y],color=COLORS[model],fmt='none',
                    ecolor=COLORS[model],alpha=.55,lw=1.0,capsize=2,zorder=2)
    ax.axhline(0,color='#97a4a9',ls='--',lw=.9,zorder=1)
    ax.set_xticks([0,1,2,3]);ax.set_xlim(-.28,3.28)
    ax.set_xlabel('RONI lag (months)')
    ax.set_ylabel('HV coefficient (pp / historical SD)')
    ax.grid(axis='y',alpha=.8)
    ax.set_title('Global ENSO–dryness associations',pad=10)


def add_h1_annotation(ax,coeff):
    """Show H1 explicitly as a concise readout; full H1 95% CIs in SI Figure S3."""
    vals=coeff.loc[(coeff.model=='H1_global_time_FE') & (coeff.term=='HV')].sort_values('lag')
    if len(vals)!=4:raise ValueError('Missing H1 lag coefficients')
    best=vals.loc[vals.lag.eq(3)].iloc[0]
    #ax.text(.98,.97, f'H1 (+ observed climate), lag 3: {best.estimate_Q3_reference:+.2f} '
    #        f'[{best.ci_low_Q3_reference:+.2f}, {best.ci_high_Q3_reference:+.2f}]',
    #        transform=ax.transAxes,fontsize=7.8,color=TEAL,ha='right',
    #        bbox=dict(boxstyle='round,pad=.36',facecolor='white',edgecolor='#d8e4e2',alpha=.95),
    #        va='top')


def exposure_for_2026(d,beta,roni):
    """Actual 2026 lag-3 predicted VPD exposure: beta(lag3,season) * contemporaneous lagged RONI delta,
    standardized by 2019-2025 historical SD, EXACT definition used in H models.
    """
    cols=['ORIG_FID','year','month','subregion']
    z=d[cols].copy()
    z['season']=z.month.map(base.SEASONS)
    b=beta.loc[beta.enso_lag.eq(3),['ORIG_FID','season','ENSO_VPD_beta','ENSO_SM_beta']]
    z=z.merge(b,on=['ORIG_FID','season'],how='left',validate='many_to_one')
    if z[['ENSO_VPD_beta','ENSO_SM_beta']].isna().any().any():
        raise ValueError('Some Q3 rows lack lag3 historical beta')
    z,sd,keys=hmod.attach_enso(z,roni,3)
    z=z.loc[z.year.eq(2026)].copy()
    return z,sd


def exposure_axis(ax,exposed,regional,out):
    # Order uses observed GUB counts (rather than association size or significance).
    included=set(regional.loc[(regional.model=='H0_global_time_FE') & (regional.term=='HV'),'subregion'])
    z=exposed.loc[exposed.subregion.isin(included)].copy()
    if z.empty:raise ValueError('No 2026 predicted exposure for plotted Q3 regions')
    tbl=z.groupby(['subregion','month'],dropna=False).agg(
        mean_HV=('HV','mean'),median_HV=('HV','median'),
        mean_HV_raw=('H_V_raw','mean'),n_obs=('ORIG_FID','size'),
        n_gub=('ORIG_FID','nunique')).reset_index()
    tbl.to_csv(out/'Fig2_2026_subregion_expected_HV_lag3.csv',index=False)
    order=z.groupby('subregion').ORIG_FID.nunique().sort_values(ascending=False).index.tolist()
    mat=tbl.pivot(index='subregion',columns='month',values='mean_HV').reindex(index=order,columns=range(1,10))
    x=mat.to_numpy('float64')
    bound=max(.1,float(np.nanmax(np.abs(x))))
    norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
    im=ax.imshow(np.ma.masked_invalid(x),aspect='auto',cmap='BrBG',norm=norm,
                 interpolation='nearest')
    ax.set_yticks(range(len(order)),order,fontsize=8)
    ax.set_xticks(range(9),MONTHS,fontsize=8)
    ax.tick_params(length=0)
    ax.set_title('ENSO-related VPD exposure · lag 3',pad=10)
    ax.set_facecolor('#f1f2f2')
    cbar=ax.figure.colorbar(im,ax=ax,shrink=.8,pad=.018,fraction=.04)
    cbar.set_label('Predicted VPD exposure (historical SD)',fontsize=8)


def observed_correlations(d,out):
    """Purely descriptive monthwise 2026 spatial Spearman correlations, no fake inferential intervals."""
    z=d.loc[d.year.eq(2026),['month','ORIG_FID','veg_y_Q3','vpd_z','soil_moisture_z']].copy()
    data=[]
    for m,part in z.groupby('month',sort=True):
        for var,lab in [('vpd_z','VPD'),('soil_moisture_z','Soil moisture')]:
            x=part[[var,'veg_y_Q3']].replace([np.inf,-np.inf],np.nan).dropna()
            if len(x)<3:raise ValueError(f'Insufficient 2026 GUBs for {lab} month {m}')
            stat=spearmanr(x[var].to_numpy(),x.veg_y_Q3.to_numpy()).statistic
            data.append({'month':m,'variable':lab,'spearman_rho':stat,'n_gub':len(x),
                         'sample':'Q3 2026; single month cross-GUB spatial'})
    t=pd.DataFrame(data)
    t.to_csv(out/'Fig2_2026_observed_climate_vegetation_spatial_correlations.csv',index=False)
    return t


def observed_axis(ax,t):
    for label,color in [('VPD',BROWN),('Soil moisture',TEAL)]:
        z=t.loc[t.variable.eq(label)].sort_values('month')
        ax.plot(z.month,z.spearman_rho,marker='o',markersize=4.5,color=color,
                lw=1.9,label=label)
    ax.axhline(0,ls='--',c='#9da6aa',lw=.9)
    ax.set_xticks(range(1,10),MONTHS,rotation=0)
    ax.set_xlabel('2026 month')
    ax.set_ylabel('Spatial Spearman correlation (ρ)')
    ax.set_ylim(-1,1)
    ax.set_title('Observed climate–vegetation correspondence',pad=10)
    ax.grid(axis='y')
    ax.legend(frameon=False,loc='best',fontsize=8,ncol=2)
    ax.text(.015,.035,'Within-month cross-GUB correlations; descriptive',
            transform=ax.transAxes,fontsize=8,color=GRAY,va='bottom')


def figure2(coeff,regional,d,beta,roni,out):
    if regional.empty:
        raise ValueError('Figure 2 needs Q3 lag3 regional results. Provide --region-results or --fit-regions')
    exposed,sd=exposure_for_2026(d,beta,roni)
    corrs=observed_correlations(d,out)
    fig,axes=plt.subplots(2,2,figsize=(14.5,9.4),layout='constrained',
                          gridspec_kw={'width_ratios':[1.07,1.08],'height_ratios':[1,1.03]})
    tag(axes[0,0],'a');lag_axis(axes[0,0],coeff);add_h1_annotation(axes[0,0],coeff)
    tag(axes[0,1],'b');draw_forest(axes[0,1],regional)
    tag(axes[1,0],'c');exposure_axis(axes[1,0],exposed,regional,out)
    tag(axes[1,1],'d');observed_axis(axes[1,1],corrs)
    axes[0,0].legend(frameon=False,loc='lower left',fontsize=8)
    #fig.suptitle('ENSO-related hydroclimate and urban vegetation',fontsize=14,fontweight='bold',y=1.025)
    save(fig,out,'Figure2_Q3')
    print('Fig.2: HV global lag & 95% CI; regional H0 & 95% CI; predicted exposure; '
          'descriptive observed correlations.',flush=True)


def supplemental_figs(d,months,maudit,coeff,a,out):
    # S1: quality trends from full observation universe, labelled separately.
    if a.quality_monthly:
        q=pd.read_csv(a.quality_monthly)
        req={'year','month','frac_Q1','frac_Q2','clear_mean','snow_mean'}
        if not req.issubset(q):raise ValueError(f'Quality monthly missing {req-set(q)}')
        fig,axes=plt.subplots(2,2,figsize=(12,6.5),layout='constrained')
        for yr,g in q.groupby('year'):
            color=TEAL if yr==2026 else GRAY
            alpha=1 if yr==2026 else .25
            axes[0,0].plot(g.month,g.frac_Q1*100,color=color,alpha=alpha,lw=2 if yr==2026 else 1)
            axes[0,1].plot(g.month,g.frac_Q2*100,color=color,alpha=alpha,lw=2 if yr==2026 else 1)
            axes[1,0].plot(g.month,g.clear_mean,color=color,alpha=alpha,lw=2 if yr==2026 else 1)
            axes[1,1].plot(g.month,g.snow_mean,color=color,alpha=alpha,lw=2 if yr==2026 else 1)
        for i,(ax,title) in enumerate(zip(axes.ravel(),['Q1 pass rate','Q2 pass rate','Mean clear coverage','Mean snow coverage'])):
            tag(ax,'abcd'[i]);ax.set_title(title);ax.set_xticks(range(1,10),MONTHS,rotation=35)
            ax.set_ylabel('%');ax.grid(axis='y')
        fig.legend(handles=[Line2D([0],[0],color=GRAY,lw=2,label='2019–2025'),
            Line2D([0],[0],color=TEAL,lw=2,label='2026')],loc='outside lower center',ncol=2,frameon=False)
        fig.suptitle('SCL observation-quality diagnostics · full GUB universe',fontsize=13)
        save(fig,out,'Supplementary_FigS1_SCL_quality')
    # S2: quality sample audit; source 4.27m and Q3 observations.
    fig,ax=plt.subplots(figsize=(8.3,4.5),layout='constrained');tag(ax,'a')
    counts=maudit.groupby('year').n_obs.sum().reindex(range(2019,2027)).fillna(0)
    bars=ax.bar(counts.index.astype(str),counts.to_numpy()/1e6,color=[TEAL if y==2026 else '#82919b' for y in counts.index])
    ax.set_ylabel('Qualified observations (millions)');ax.set_title('Q3 retained monthly records by year')
    ax.grid(axis='y');save(fig,out,'Supplementary_FigS2_sample_retention')
    # S3: paired reference for all lags and HV/HS under all model types.
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.0),layout='constrained')
    for row,term in enumerate(['HV','HS']):
        for col,model in enumerate(LABELS):
            ax=axes[row,col];tag(ax,chr(ord('a')+row*3+col))
            ss=coeff.loc[(coeff.term==term)&(coeff.model==model)].sort_values('lag')
            for offset,ref,color in [(-.075,'original_reference',GRAY),(.075,'Q3_reference',TEAL)]:
                if ss.empty:continue
                x=ss.lag.to_numpy()+offset;value=ss[f'estimate_{ref}'].to_numpy()
                low=ss[f'ci_low_{ref}'].to_numpy();high=ss[f'ci_high_{ref}'].to_numpy()
                ax.errorbar(x,value,yerr=[value-low,high-value],fmt='o-',lw=1.3,
                            color=color,capsize=3,label='Original' if offset<0 else 'Q3')
            ax.axhline(0,color='#a8b1b5',ls='--',lw=.8);ax.set_xticks(range(4));ax.grid(axis='y')
            if row==0:ax.set_title(LABELS[model],fontsize=9)
            if col==0:ax.set_ylabel(f'{term} coefficient (pp / historical SD)')
            if row==1:ax.set_xlabel('RONI lag (months)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',frameon=False,ncol=2)
    fig.suptitle('Historical reference reconstruction · paired-sample sensitivity',fontsize=13)
    save(fig,out,'Supplementary_FigS3_reference_sensitivity')
    # S4: different inference, SAME coefficients; show interval widening.
    if a.cluster_comparison:
        t=pd.read_csv(a.cluster_comparison)
        keep=t.loc[((t.term=='HV')&t.model.str.contains('H0|H1'))|
                   ((t.term.isin(['V_D','S_D']))&(t.model=='M0'))].copy()
        keep['model_term']=keep.model+' / '+keep.term
        keys=keep.model_term.unique().tolist()
        fig,ax=plt.subplots(figsize=(10,max(3.5,len(keys)*.22+1.6)),layout='constrained')
        for idx,key in enumerate(keys):
            z=keep.loc[keep.model_term.eq(key)]
            for tagname,offset,color in [('spatial',-.12,GRAY),('two_way',.12,TEAL)]:
                one=z.loc[z.inference.eq(tagname)]
                if one.empty:continue
                row=one.iloc[0];v=row.estimate
                ax.errorbar(v,idx+offset,xerr=[[v-row.ci_low],[row.ci_high-v]],
                            fmt='o',color=color,capsize=3,markersize=4)
        ax.set_yticks(np.arange(len(keys)),keys,fontsize=8);ax.invert_yaxis()
        ax.axvline(0,c='#9ba7ad',ls='--',lw=.8);ax.grid(axis='x')
        ax.set_xlabel('Estimated coefficient and 95% CI (full, unfiltered sample)')
        ax.set_title('One-way versus two-way clustered inference')
        ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=GRAY,label='5° block'),
              Line2D([0],[0],marker='o',ls='',color=TEAL,label='5° block + year–month')],frameon=False)
        save(fig,out,'Supplementary_FigS4_clustering')


def main():
    a=parse_args();style()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    coef=read_q3_coeff(a.q3_summary)
    d,beta,maudit,summary,raudit=q3_sample(a)
    validate_audit(summary,a.q3_audit)
    maudit.to_csv(out/'FigS2_Q3_retention_by_year_month.csv',index=False)
    summary.to_csv(out/'Figure_data_sample_audit.csv',index=False)
    raudit.to_csv(out/'FigS2_Q3_retention_by_subregion.csv',index=False)
    now,monthly,region=figure1(d,out,a)
    # Region model fitted on same Q3 sample; old lag1/unfiltered regional results disallowed.
    roni=hmod.roni_series(a.roni)
    if a.region_results:
        regional=pd.read_csv(a.region_results)
        if not {'lag','reference','model','term','subregion','estimate','ci_low','ci_high'}.issubset(regional):
            raise ValueError('Region CSV needs lag/reference/model/term/subregion/estimate/ci_low/ci_high')
        if not regional.lag.eq(3).all() or not regional.reference.eq('Q3').all():
            raise ValueError('Region CSV must be Q3 lag3, not old unfiltered or other lag')
        regional.to_csv(out/'Fig2_region_Q3_lag3_H0_H1_twoway.csv',index=False)
    elif a.fit_regions:
        regional=fit_regions(d,beta,roni,a,out)
    else:
        raise ValueError('Provide --region-results with Q3 lag3 region CSV, or --fit-regions')
    figure2(coef,regional,d,beta,roni,out)
    supplemental_figs(d,monthly,maudit,coef,a,out)
    print('\nFINISHED. Main figures: Figure1_Q3.png/pdf and Figure2_Q3.png/pdf',flush=True)
    print('Fig.2: panel c is actual Q3 2026 lag3 exposure; panel d is descriptive spatial correlations.',flush=True)
    print('Summary Q3 sample:',summary.to_dict(orient='records')[0],flush=True)

if __name__=='__main__':main()

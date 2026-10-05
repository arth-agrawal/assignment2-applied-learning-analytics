"""Reproduce the OULAD results. Usage: python analysis_oulad.py --data PATH --out PATH.
Requires pandas, numpy, scipy, scikit-learn and matplotlib.
The original archive is https://ndownloader.figshare.com/files/8606371.
No Moodle results are generated without the course dataset.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42))

def metrics(y,p):
    pred=p>=.5
    return dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()),accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),confusion=confusion_matrix(y,pred).tolist())

def auc_boot(y,p,q=None):
    rng=np.random.default_rng(42); vals=[]
    y=np.asarray(y)
    for _ in range(1000):
        ix=rng.integers(len(y),size=len(y))
        if len(np.unique(y[ix]))<2: continue
        a=roc_auc_score(y[ix],p[ix])
        vals.append(a if q is None else a-roc_auc_score(y[ix],q[ix]))
    return np.quantile(vals,[.025,.975]).tolist()

def coef_boot(X,y,col,increment):
    rng=np.random.default_rng(43); vals=[]
    for _ in range(300):
        ix=rng.integers(len(y),size=len(y)); m=model().fit(X.iloc[ix],y.iloc[ix])
        coef=m[-1].coef_[0][X.columns.get_loc(col)]/m[0].scale_[X.columns.get_loc(col)]
        vals.append(np.exp(coef*increment))
    m=model().fit(X,y)
    j=X.columns.get_loc(col); value=np.exp(m[-1].coef_[0][j]/m[0].scale_[j]*increment)
    return dict(odds_ratio=float(value),ci=np.quantile(vals,[.025,.975]).tolist(),increment=increment)

def savefig(out,name):
    plt.tight_layout(); plt.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight'); plt.close()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',required=True); parser.add_argument('--out',default='results'); args=parser.parse_args()
    data=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    info=pd.read_csv(data/'studentInfo.csv'); reg=pd.read_csv(data/'studentRegistration.csv'); ass=pd.read_csv(data/'assessments.csv'); sub=pd.read_csv(data/'studentAssessment.csv')
    keys=['code_module','code_presentation','id_student']
    assert not info.duplicated(keys).any() and not reg.duplicated(keys).any()
    assert not sub.duplicated(['id_assessment','id_student']).any()
    cohort=info[(info.code_module=='BBB') & info.code_presentation.isin(['2013J','2014J'])].merge(reg,on=keys,validate='one_to_one')
    # Preserve daily resource counts, then sum across resources. Days before start are not predictors.
    parts=[]; raw_n=0; selected_n=0; invalid_n=0
    for chunk in pd.read_csv(data/'studentVle.csv',chunksize=500000):
        raw_n+=len(chunk)
        c=chunk[(chunk.code_module=='BBB') & chunk.code_presentation.isin(['2013J','2014J'])].copy()
        selected_n+=len(c)
        bad=c[['date','sum_click']].isna().any(axis=1)|(c.sum_click<0)
        invalid_n+=int(bad.sum()); c=c[~bad & c.date.between(0,259)]
        parts.append(c.groupby(keys+['date'],as_index=False).sum_click.sum())
    daily=pd.concat(parts).groupby(keys+['date']).sum_click.sum().unstack('date',fill_value=0).reindex(columns=range(260),fill_value=0)
    daily=daily.reindex(pd.MultiIndex.from_frame(cohort[keys]),fill_value=0)
    daily.index=cohort.index
    train_ids=set(cohort.loc[cohort.code_presentation=='2013J','id_student'])
    repeated=(cohort.code_presentation=='2014J') & cohort.id_student.isin(train_ids)
    results={'source':{'rows_studentInfo':len(info),'unique_students':int(info.id_student.nunique()),'rows_studentVle':raw_n,'selected_vle_rows':selected_n,'invalid_vle_rows':invalid_n,'cohort_counts':cohort.groupby('code_presentation').size().to_dict(),'repeated_test_ids_excluded':int(repeated.sum()),'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in data.glob('*.csv') if f.name!='studentVle.csv'}},'seed':42}
    # Hypothesis O1. All predictors are observed by the end of day 55.
    eligible=(cohort.date_registration.le(0)) & (cohort.date_unregistration.isna() | cohort.date_unregistration.gt(55)) & ~repeated
    f=cohort[eligible].copy(); d=daily.loc[f.index,range(56)]
    f['clicks']=d.sum(axis=1); f['log_clicks']=np.log1p(f.clicks); f['active_days']=(d>0).sum(axis=1)
    f['coverage']=f.active_days/56
    f['weekly_cv']=d.T.groupby(np.arange(56)//7).sum().std(ddof=0)/(f.clicks/8).replace(0,np.nan)
    f['weekly_cv']=f.weekly_cv.fillna(0)
    f['success']=f.final_result.isin(['Pass','Distinction']).astype(int)
    train=f.code_presentation.eq('2013J'); test=f.code_presentation.eq('2014J')
    base=['log_clicks','num_of_prev_attempts','studied_credits']; ext=base+['coverage']
    predictions={}; fitted={}
    for name,cols in [('volume',base),('regularity',ext)]:
        m=model().fit(f.loc[train,cols],f.loc[train,'success']); p=m.predict_proba(f.loc[test,cols])[:,1]
        fitted[name]=m; predictions[name]=p; results[name]=metrics(f.loc[test,'success'],p); results[name]['auc_ci']=auc_boot(f.loc[test,'success'],p)
    results['O1']={'train_n':int(train.sum()),'test_n':int(test.sum()),'train_success':int(f.loc[train,'success'].sum()),'effect':coef_boot(f.loc[train,ext],f.loc[train,'success'],'coverage',.1),'auc_difference':results['regularity']['roc_auc']-results['volume']['roc_auc'],'difference_ci':auc_boot(f.loc[test,'success'],predictions['regularity'],predictions['volume'])}
    f.to_csv(out/'oulad_week8_features.csv',index=False)
    groups=[]
    for label,g in f[test].groupby(pd.cut(f.loc[test,'active_days'],[-1,14,28,42,56],labels=['0 to 14','15 to 28','29 to 42','43 to 56']),observed=False):
        groups.append({'active_days':str(label),'n':len(g),'success_rate':float(g.success.mean())})
    results['O1']['coverage_groups']=groups
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    styles=[('Pass or distinction','solid','o',1),('Fail or withdrawal','dashed','s',0)]
    for label,style,marker,y in styles:
        ix=f.index[test & f.success.eq(y)]; dd=daily.loc[ix,range(56)]
        w=dd.T.groupby(np.arange(56)//7).sum().T
        axes[0].plot(range(1,9),w.median(),color='black',ls=style,marker=marker,label=label)
        a=(dd>0).T.groupby(np.arange(56)//7).sum().T
        axes[1].plot(range(1,9),a.mean(),color='black',ls=style,marker=marker,label=label)
    axes[0].set(xlabel='Week from course start',ylabel='Median weekly clicks',xticks=range(1,9))
    axes[1].set(xlabel='Week from course start',ylabel='Mean active days per week',xticks=range(1,9),ylim=(0,7)); axes[1].legend(fontsize=8)
    savefig(out,'weekly_behavior')
    plt.figure(figsize=(8,3)); plt.bar([x['active_days'] for x in groups],[100*x['success_rate'] for x in groups],color='white',edgecolor='black',hatch='///')
    for i,g in enumerate(groups): plt.text(i,100*g['success_rate']+2,f"n = {g['n']}",ha='center',fontsize=9)
    plt.ylim(0,105);plt.xlabel('Active days during the first 56 days');plt.ylabel('Pass or distinction (%)');savefig(out,'coverage_outcomes')
    plt.figure(figsize=(7.8,3.0))
    for name,style in [('volume','dashed'),('regularity','solid')]:
        a,b,_=roc_curve(f.loc[test,'success'],predictions[name]);plt.plot(a,b,color='black',ls=style,label=f"{name.capitalize()}: area = {results[name]['roc_auc']:.3f}")
    plt.plot([0,1],[0,1],color='0.6',ls=':');plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.legend();savefig(out,'roc')
    # Hypothesis O2. Choose the first non-exam TMA deadline >= day 70 per course run.
    deadlines=ass[(ass.code_module=='BBB') & ass.code_presentation.isin(['2013J','2014J']) & ass.assessment_type.eq('TMA') & ass.date.ge(70)].sort_values('date').groupby('code_presentation').head(1)
    rows=[]; dates={}; early_count={}; banked_count={}
    for a in deadlines.itertuples():
        due=int(a.date); checkpoint=due-7; dates[a.code_presentation]={'id_assessment':int(a.id_assessment),'due_day':due,'checkpoint_day':checkpoint}
        c=cohort[cohort.code_presentation.eq(a.code_presentation)&~repeated].copy()
        c=c[c.date_registration.le(checkpoint-13)&(c.date_unregistration.isna()|c.date_unregistration.gt(checkpoint))]
        ss=sub[sub.id_assessment.eq(a.id_assessment)].set_index('id_student')
        c['submission_day']=c.id_student.map(ss.date_submitted);c['banked']=c.id_student.map(ss.is_banked).fillna(0)
        banked_count[a.code_presentation]=int(c.banked.eq(1).sum());c=c[c.banked.eq(0)]
        early_count[a.code_presentation]=int(c.submission_day.le(checkpoint).sum()); c=c[~c.submission_day.le(checkpoint)]
        c['prior_clicks']=daily.loc[c.index,range(checkpoint-13,checkpoint-6)].sum(axis=1)
        c['recent_clicks']=daily.loc[c.index,range(checkpoint-6,checkpoint+1)].sum(axis=1)
        c['log_prior']=np.log1p(c.prior_clicks)
        c['decline']=np.log1p(c.prior_clicks)-np.log1p(c.recent_clicks)
        c['late_missing']=(c.submission_day.isna()|c.submission_day.gt(due)).astype(int)
        c['due']=due;c['checkpoint']=checkpoint;rows.append(c)
    h=pd.concat(rows); train=h.code_presentation.eq('2013J');test=h.code_presentation.eq('2014J')
    bcols=['log_prior','num_of_prev_attempts','studied_credits']; xcols=bcols+['decline']
    sub_predictions={}
    for name,cols in [('submission_volume',bcols),('submission_decline',xcols)]:
        m=model().fit(h.loc[train,cols],h.loc[train,'late_missing']);p=m.predict_proba(h.loc[test,cols])[:,1]
        results[name]=metrics(h.loc[test,'late_missing'],p)
        sub_predictions[name]=p
        results[name]['auc_ci']=auc_boot(h.loc[test,'late_missing'],p)
    results['O2']={'deadlines':dates,'banked_excluded':banked_count,'early_submissions_excluded':early_count,'train_n':int(train.sum()),'test_n':int(test.sum()),'effect':coef_boot(h.loc[train,xcols],h.loc[train,'late_missing'],'decline',np.log(2))}
    results['O2']['auc_difference']=results['submission_decline']['roc_auc']-results['submission_volume']['roc_auc']
    results['O2']['difference_ci']=auc_boot(h.loc[test,'late_missing'],sub_predictions['submission_decline'],sub_predictions['submission_volume'])
    rates=[]
    ht=h[test].copy();ht['group']=np.where(ht.prior_clicks.eq(0),'No prior clicks',np.where(ht.recent_clicks.lt(ht.prior_clicks),'Declining activity','Stable or rising activity'))
    for label,g in ht.groupby('group'): rates.append({'group':label,'n':len(g),'late_missing_rate':float(g.late_missing.mean()),'late_n':int((g.submission_day>g.due).sum()),'missing_n':int(g.submission_day.isna().sum())})
    results['O2']['groups']=rates
    h.to_csv(out/'oulad_submission_features.csv',index=False)
    plt.figure(figsize=(8,3.2))
    due=dates['2014J']['due_day']
    for label,style,mark,y in [('On time','solid','o',0),('Late or missing','dashed','s',1)]:
        ix=ht.index[ht.late_missing.eq(y)]
        yy=(daily.loc[ix,range(due-27,due+1)]>0).mean()*100
        plt.plot(range(-27,1),yy,color='black',ls=style,marker=mark,markevery=4,label=f'{label} (n = {len(ix)})')
    plt.axvline(-7,color='0.5',ls=':');plt.text(-6.5,plt.ylim()[1]*.95,'Prediction point',fontsize=8,va='top')
    plt.xlabel('Days relative to assessment deadline');plt.ylabel('Learners active that day (%)');plt.legend(fontsize=8,loc='lower left');savefig(out,'deadline_behavior')
    results['O2']['test_missing']=int(ht.submission_day.isna().sum());results['O2']['test_late']=int(ht.submission_day.gt(ht.due).sum())
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()

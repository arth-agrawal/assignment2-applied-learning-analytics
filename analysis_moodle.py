"""Analyze the supplied Moodle course log.

Example: python analysis_moodle.py --data "MOODLE Data.csv" --out results_moodle
Requires pandas, numpy, scipy, scikit-learn, matplotlib.
Record-level exports are not written. Output figures and summaries contain no
learner identifiers. Course logs have no grade or completion field; continuation
means a recorded event in the next calendar week.
"""
from pathlib import Path
import argparse, json, re, hashlib, site
import numpy as np
import pandas as pd
# The workspace carries the plotting wheel for this task when it is not global.
site.addsitedir(str(Path(__file__).resolve().parents[1]/'work'/'packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def scale_fit(x):
    mu=x.mean(0); sd=x.std(0); sd[sd==0]=1
    return (x-mu)/sd,mu,sd
def logistic_fit(x,y,penalty=1.0):
    x=np.c_[np.ones(len(x)),x];w=np.zeros(x.shape[1]);reg=np.diag([0]+[penalty]*(x.shape[1]-1))
    for _ in range(80):
        z=np.clip(x@w,-30,30);p=1/(1+np.exp(-z));g=x.T@(p-y)+reg@w
        h=x.T@((p*(1-p))[:,None]*x)+reg+np.eye(x.shape[1])*1e-9
        step=np.linalg.solve(h,g);w-=step
        if np.max(np.abs(step))<1e-8:break
    return w
def auc(y,p):
    order=np.argsort(p,kind='mergesort');r=np.empty(len(order),float);r[order]=np.arange(1,len(order)+1)
    # Average ranks for tied scores.
    vals,inv,cnt=np.unique(p,return_inverse=True,return_counts=True);means=np.cumsum(cnt)-cnt/2+0.5;r=means[inv]
    n1=int(np.sum(y));n0=len(y)-n1
    return float((r[np.asarray(y)==1].sum()-n1*(n1+1)/2)/(n1*n0))
def average_precision(y,p):
    order=np.argsort(-p,kind='mergesort');ys=np.asarray(y)[order];tp=np.cumsum(ys);fp=np.cumsum(1-ys)
    return float(np.sum((tp/(tp+fp))*ys)/max(1,ys.sum()))
def binary_metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);yp=p>=.5;tp=int(np.sum((y==1)&yp));fp=int(np.sum((y==0)&yp));fn=int(np.sum((y==1)&~yp));tn=int(np.sum((y==0)&~yp))
    prec=tp/max(1,tp+fp);rec=tp/max(1,tp+fn)
    return dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()),accuracy=(tp+tn)/len(y),precision=prec,recall=rec,f1=2*prec*rec/max(1e-12,prec+rec),roc_auc=auc(y,p),average_precision=average_precision(y,p),confusion=[[tn,fp],[fn,tp]])
def boot_diff(y,p,q,groups,seed=42,n=1000):
    y=np.asarray(y);p=np.asarray(p);q=np.asarray(q);groups=np.asarray(groups);rng=np.random.default_rng(seed);v=[];users=np.unique(groups)
    for _ in range(n):
        draw=rng.choice(users,size=len(users),replace=True)
        ix=np.concatenate([np.flatnonzero(groups==user) for user in draw])
        if np.unique(y[ix]).size==2:v.append(auc(y[ix],p[ix])-auc(y[ix],q[ix]))
    return np.quantile(v,[.025,.975]).tolist()
def km_fit(x,k,seed=42,restarts=20):
    rng=np.random.default_rng(seed);best=None
    for _ in range(restarts):
        centers=[x[rng.integers(len(x))].copy()]
        for _j in range(1,k):
            dd=np.min(((x[:,None,:]-np.asarray(centers)[None,:,:])**2).sum(2),axis=1);pr=dd/dd.sum()
            centers.append(x[rng.choice(len(x),p=pr)].copy())
        centers=np.asarray(centers)
        for _j in range(100):
            lab=((x[:,None,:]-centers[None,:,:])**2).sum(2).argmin(1)
            new=np.asarray([x[lab==j].mean(0) if np.any(lab==j) else x[rng.integers(len(x))] for j in range(k)])
            if np.allclose(centers,new,atol=1e-6):centers=new;break
            centers=new
        loss=float(((x-centers[lab])**2).sum())
        if best is None or loss<best[0]:best=(loss,lab.copy(),centers.copy())
    return best[1],best[2]
def silhouette(x,lab):
    dist=np.sqrt(np.maximum(0,((x[:,None,:]-x[None,:,:])**2).sum(2)));out=[]
    for i in range(len(x)):
        own=np.where(lab==lab[i])[0];own=own[own!=i]
        if not len(own):continue
        aa=dist[i,own].mean();bb=min(dist[i,lab==c].mean() for c in np.unique(lab) if c!=lab[i]);out.append((bb-aa)/max(aa,bb))
    return float(np.mean(out))
def adjusted_rand(a,b):
    _,aa=np.unique(a,return_inverse=True);_,bb=np.unique(b,return_inverse=True)
    tab=np.zeros((aa.max()+1,bb.max()+1),dtype=np.int64)
    np.add.at(tab,(aa,bb),1)
    c2=lambda z:z*(z-1)/2
    ix=c2(tab).sum();ra=c2(tab.sum(1)).sum();cb=c2(tab.sum(0)).sum();tot=len(a)*(len(a)-1)/2
    exp=ra*cb/tot if tot else 0;mx=(ra+cb)/2
    return float((ix-exp)/(mx-exp)) if mx!=exp else 1.0

USER_PAT=re.compile(r"user with the id '(\d+)'")
NUMERIC_ID=re.compile(r"'(\d+)'")
def metrics(y,p):
    return binary_metrics(y,p)
def ci_metric(y,p,fn,seed=42,n=1000):
    y=np.asarray(y);p=np.asarray(p);g=np.random.default_rng(seed);v=[]
    for _ in range(n):
        ix=g.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))==2:v.append(fn(y[ix],p[ix]))
    return np.quantile(v,[.025,.975]).tolist()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',default='results_moodle');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    path=Path(a.data);d=pd.read_csv(path,encoding='utf-8-sig',dtype=str);raw_n=len(d);dups=int(d.duplicated().sum())
    # Moodle's export malformed the first column on rows where its quoted user id
    # was split. Recover these two cases from the matching activity description.
    ids=d['User ID'].str.extract(NUMERIC_ID,expand=False)
    malformed=ids.isna();fallback=d.Description.str.extract(USER_PAT,expand=False);recover=malformed&fallback.notna();ids.loc[recover]=fallback.loc[recover]
    rec_n=int(recover.sum());unresolved=int(ids.isna().sum())
    # A matching user id in the event description is an independent consistency check.
    described=d.Description.str.extract(USER_PAT,expand=False)
    discrepancy=int((ids.notna()&described.notna()&ids.ne(described)).sum())
    d['uid']=ids;d['ts']=pd.to_datetime(d.Time,format='%d/%m/%y, %H:%M:%S',errors='coerce')
    invalid_time=int(d.ts.isna().sum());d=d[d.uid.notna()&d.ts.notna()].copy()
    # This is a single course activity export. Resource and assessment rows refer
    # to course modules, so filtering only rows that literally say "Course" would
    # silently remove most learning events. Exclude staff-style report views.
    n_pre=int(len(d));d=d[d['Event name'].ne('Course user report viewed')].copy()
    # Count distinct event records as Moodle logs. Do not infer minutes from events.
    d['week']=d.ts.dt.to_period('W-SUN').dt.start_time
    start=d.week.min();end=d.ts.max().normalize();people=sorted(d.uid.unique())
    d['day']=d.ts.dt.normalize();d['component']=d.Component.fillna('Unknown')
    d['event_name']=d['Event name'].fillna('Unknown')
    component_counts=d.component.value_counts()
    d['is_quiz']=d.component.eq('Quiz').astype(int)
    d['is_resource']=d.component.isin(['Book','Page','File','SCORM package','Glossary']).astype(int)
    d['is_submission']=d.component.isin(['Assignment','File submissions','Online text submissions']).astype(int)
    # Sessionize per learner after sorting. A session starts after a gap over 30 minutes.
    d=d.sort_values(['uid','ts'])
    gaps=d.groupby('uid').ts.diff().dt.total_seconds().div(60)
    d['session_start']=(gaps.isna()|gaps.gt(30)).astype(int)
    d['session_id']=d.groupby('uid').session_start.cumsum()
    sessions=d.groupby(['uid','session_id'],as_index=False).agg(session_start=('ts','min'),session_end=('ts','max'),session_events=('event_name','size'))
    sessions['duration_min']=(sessions.session_end-sessions.session_start).dt.total_seconds().div(60)
    session_summary=sessions.groupby('uid').agg(sessions=('session_id','size'),median_session_min=('duration_min','median'))
    # Weekly time line, with the observed interval shown including empty weeks.
    weekly=d.groupby('week').agg(events=('uid','size'),learners=('uid','nunique')).reindex(pd.date_range(start,end,freq='W-MON'),fill_value=0)
    weekly.index.name='week'
    fig,ax=plt.subplots(figsize=(9,3.2));ax.plot(weekly.index,weekly.events,color='black',lw=1.2,label='Events');ax.set_ylabel('Recorded events per week');ax.set_xlabel('Week beginning');ax.tick_params(axis='x',rotation=30);ax2=ax.twinx();ax2.plot(weekly.index,weekly.learners,color='0.45',ls=':',lw=1.2,label='Learners');ax2.set_ylabel('Learners active that week');ax.legend(loc='upper left',frameon=False);ax2.legend(loc='upper right',frameon=False);fig.tight_layout();fig.savefig(out/'moodle_weekly_trend.png',dpi=220,bbox_inches='tight');plt.close(fig)
    # M1: non-overlapping four-week feature windows, next seven days target.
    # Course calendar begins with the Monday of the first observed event week.
    first=pd.Timestamp(start);last=pd.Timestamp(end);horizon=7;window=28
    obs_days=(last-first).days+1
    windows=[]
    for offset in range(0,obs_days-window-horizon+1,window):
        lo=first+pd.Timedelta(days=offset);hi=lo+pd.Timedelta(days=window);target_end=hi+pd.Timedelta(days=horizon)
        f=d[(d.ts>=lo)&(d.ts<hi)]
        nxt=d[(d.ts>=hi)&(d.ts<target_end)]
        # Only learners observed in the feature window can be forecast from logs.
        by=f.groupby('uid').agg(event_count=('event_name','size'),active_days=('day','nunique'),quiz_events=('is_quiz','sum'),resource_events=('is_resource','sum'),submission_events=('is_submission','sum'))
        by['coverage']=by.active_days/window;by['log_events']=np.log1p(by.event_count)
        denom=by.event_count.replace(0,np.nan)
        by['quiz_share']=by.quiz_events/denom;by['resource_share']=by.resource_events/denom;by['submission_share']=by.submission_events/denom
        returned=set(nxt.uid.unique());by['returned']=by.index.isin(returned).astype(int);by['window_start']=lo
        windows.append(by.reset_index())
    panel=pd.concat(windows,ignore_index=True);cols0=['log_events'];cols1=cols0+['coverage']
    rng=np.random.default_rng(42);all_users=np.array(sorted(panel.uid.unique()));heldout=set(rng.choice(all_users,size=max(1,round(.2*len(all_users))),replace=False))
    train=panel[~panel.uid.isin(heldout)].copy();test=panel[panel.uid.isin(heldout)].copy()
    models={};preds={};res={};scaled={}
    for label,cols in [('event_volume',cols0),('volume_and_coverage',cols1)]:
        xt,mu,sd=scale_fit(train[cols].to_numpy(float));xv=(test[cols].to_numpy(float)-mu)/sd;w=logistic_fit(xt,train.returned.to_numpy(float));p=1/(1+np.exp(-np.clip(np.c_[np.ones(len(xv)),xv]@w,-30,30)));models[label]=(w,sd);preds[label]=p;res[label]=metrics(test.returned,p)
    changed=res['volume_and_coverage']['roc_auc']-res['event_volume']['roc_auc']
    res['M1']={'features':'Four calendar weeks of events; next seven days is the outcome window.','split':'80% of learners train, 20% held out by learner identifier; no learner appears in both sets.','train_learners':int(train.uid.nunique()),'test_learners':int(test.uid.nunique()),'train_rows':len(train),'test_rows':len(test),'window_count':int(panel.window_start.nunique()),'auc_change':changed,'auc_change_ci':boot_diff(test.returned,preds['volume_and_coverage'],preds['event_volume'],test.uid),'activity_coverage_effect':float(np.exp(models['volume_and_coverage'][0][2]/models['volume_and_coverage'][1][1]*.1))}
    # M2: K Means on whole-period learner profiles. Counts are log transformed;
    # activity shares are calculated within each learner's recorded events.
    prof=d.groupby('uid').agg(events=('event_name','size'),active_days=('day','nunique'),active_weeks=('week','nunique'),quiz=('is_quiz','sum'),resource=('is_resource','sum'),submission=('is_submission','sum'))
    prof=prof.join(session_summary,how='left').fillna(0)
    weeks=max(1,len(weekly));days=max(1,(last-first).days+1)
    prof['log_events']=np.log1p(prof.events);prof['day_coverage']=prof.active_days/days;prof['week_coverage']=prof.active_weeks/weeks
    prof['quiz_share']=prof.quiz/prof.events;prof['resource_share']=prof.resource/prof.events;prof['submission_share']=prof.submission/prof.events
    profile_cols=['log_events','day_coverage','week_coverage','quiz_share','resource_share','submission_share','sessions']
    X,_,_=scale_fit(prof[profile_cols].to_numpy(float));sil={}
    for k in range(2,min(6,len(prof)-1)):
        lab,_=km_fit(X,k,seed=42+k,restarts=20)
        sizes=pd.Series(lab).value_counts()
        if sizes.min()>=8:sil[str(k)]=silhouette(X,lab)
    best=max(sil,key=sil.get) if sil else '2'
    prof['cluster'],_=km_fit(X,int(best),seed=42+int(best),restarts=20)
    repeat_labels=[km_fit(X,int(best),seed=100+j,restarts=5)[0] for j in range(10)]
    stability=[adjusted_rand(prof.cluster.to_numpy(),labs) for labs in repeat_labels]
    prof['profile_label']=prof.cluster.map(lambda z:f'Profile {z+1}')
    summary=prof.groupby('profile_label').agg(learners=('events','size'),median_events=('events','median'),median_active_days=('active_days','median'),median_active_weeks=('active_weeks','median'),median_day_coverage=('day_coverage','median'),quiz_share=('quiz_share','median'),resource_share=('resource_share','median'),submission_share=('submission_share','median'),sessions=('sessions','median')).reset_index()
    # Order labels by total interaction volume for an intuitive low to high reading order.
    order=summary.sort_values('median_events').profile_label.tolist();label_map={old:f'Profile {i+1}' for i,old in enumerate(order)};summary.profile_label=summary.profile_label.map(label_map);prof.profile_label=prof.profile_label.map(label_map)
    summary=summary.sort_values('profile_label');
    fig,ax=plt.subplots(figsize=(8.5,3.1));x=np.arange(len(summary));ax.bar(x,summary.median_active_days,color='white',edgecolor='black',hatch='///');ax.set_xticks(x,[f"{z}\n(n={n})" for z,n in zip(summary.profile_label,summary.learners)]);ax.set_xlabel('Learner activity profile, ordered by median event volume');ax.set_ylabel('Median active days in observed period');fig.tight_layout();fig.savefig(out/'moodle_profiles.png',dpi=220,bbox_inches='tight');plt.close(fig)
    res['M2']={'features':profile_cols,'selected_k':int(best),'silhouette_by_k':sil,'repeat_fit_ari_mean':float(np.mean(stability)),'repeat_fit_ari_min':float(np.min(stability)),'profile_summary':summary.to_dict(orient='records'),'note':'Clusters use only recorded behaviour; labels are descriptive and contain no learner IDs.'}
    # Event type and weekly group figures contain no individual-level data.
    top=d.component.value_counts().head(8).sort_values()
    fig,ax=plt.subplots(figsize=(8.4,3.2));ax.barh(top.index,top.values,color='white',edgecolor='black',hatch='///');ax.set_xlabel('Recorded events');ax.set_ylabel('Moodle component');fig.tight_layout();fig.savefig(out/'moodle_components.png',dpi=220,bbox_inches='tight');plt.close(fig)
    # Do not export learner IDs or row-level features.
    weekly.reset_index().to_csv(out/'moodle_weekly_aggregate.csv',index=False)
    summary.to_csv(out/'moodle_profile_aggregate.csv',index=False)
    res['source']={'rows_received':raw_n,'exact_duplicate_rows':dups,'unique_valid_learners':int(d.uid.nunique()),'malformed_id_rows_recovered_from_matching_description':rec_n,'unresolved_id_rows':unresolved,'description_id_mismatches':discrepancy,'timestamp_rows_invalid':invalid_time,'report_view_rows_excluded':n_pre-len(d),'analysis_rows':int(len(d)),'course_identifier':'Single course interaction export; Course viewed descriptions identify IT-2507, course id 1137','course_start_week_monday':str(first.date()),'last_event_date':str(last.date()),'component_counts':component_counts.to_dict(),'top_event_types':d.event_name.value_counts().head(10).to_dict(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'privacy':'Raw user identifiers and row-level learner features are omitted from exports.'}
    (out/'moodle_results.json').write_text(json.dumps(res,indent=2,default=lambda z:int(z) if isinstance(z,np.integer) else float(z) if isinstance(z,np.floating) else str(z)))
    print(json.dumps(res,indent=2,default=lambda z:int(z) if isinstance(z,np.integer) else float(z) if isinstance(z,np.floating) else str(z)))
if __name__=='__main__':main()

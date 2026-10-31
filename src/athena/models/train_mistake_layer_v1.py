from pathlib import Path as _AthenaPath
_ATHENA_ROOT = _AthenaPath(__file__).resolve().parents[3]  # repo root
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score

ROOT=Path(str(_ATHENA_ROOT))
FILES={
'XGB':ROOT/'experiments/xgboost/xgboost_v4_oos_2023_predictions.csv',
'LGB':ROOT/'experiments/lightgbm/lightgbm_v3_oos_2023_predictions.csv',
'CAT':ROOT/'experiments/catboost/catboost_v3_oos_2023_predictions.csv',
'LSTM':ROOT/'experiments/lstm/lstm_v2_validation_predictions.csv',
'TRF':ROOT/'experiments/transformer/transformer_v1_validation_predictions.csv'}
OUT=ROOT/'reports/model_comparison/mistake_layer_v1'; OUT.mkdir(parents=True,exist_ok=True)
MODELS=list(FILES); LABELS={0:'DOWN',1:'NEUTRAL',2:'UP'}

def code(s):
    n=pd.to_numeric(s,errors='coerce'); o=pd.Series(np.nan,index=s.index,dtype=float); m=n.notna(); o.loc[m]=n.loc[m]
    o.loc[~m]=s.astype(str).str.strip().str.upper().map({'DOWN':0,'NEUTRAL':1,'UP':2}).loc[~m]
    if o.isna().any(): raise ValueError('Unrecognized class values')
    o=o.astype(int)
    if not o.isin([0,1,2]).all(): raise ValueError('Bad class code')
    return o

def col(df, names, what):
    for x in names:
        if x in df: return x
    raise KeyError(f'{what}: tried {names}; available={list(df.columns)}')

def load(path):
    d=pd.read_csv(path); o=pd.DataFrame(); o['Timestamp']=pd.to_datetime(d['Timestamp'])
    o['Actual_Class']=code(d[col(d,['Actual_Class','Target_Code','Actual_Code','Target_Name'],'actual')])
    o['Predicted_Class']=code(d[col(d,['Predicted_Class','Predicted_Code','Prediction_Code','Predicted_Name'],'prediction')])
    for c in ['DOWN','NEUTRAL','UP']:
        o[f'Prob_{c}']=pd.to_numeric(d[col(d,[f'Prob_{c}',f'Probability_{c}'],f'{c} probability')],errors='raise')
    p=o[['Prob_DOWN','Prob_NEUTRAL','Prob_UP']].to_numpy(float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError(f'Invalid probabilities in {path.name}')
    return o.sort_values('Timestamp').reset_index(drop=True)

def main():
    fs={m:load(p) for m,p in FILES.items()}
    common=sorted(set(fs['XGB'].Timestamp).intersection(*[set(fs[m].Timestamp) for m in MODELS[1:]]))
    fs={m:fs[m][fs[m].Timestamp.isin(common)].sort_values('Timestamp').reset_index(drop=True) for m in MODELS}
    actual=fs['XGB'].Actual_Class.to_numpy()
    if not all(np.array_equal(actual,fs[m].Actual_Class.to_numpy()) for m in MODELS): raise ValueError('Actual target mismatch')
    n=len(actual); d=pd.DataFrame({'Timestamp':fs['XGB'].Timestamp})
    preds=[]
    for m in MODELS:
        p=fs[m].Predicted_Class.to_numpy(); preds.append(p); d[f'{m}_Pred']=p
        probs=fs[m][['Prob_DOWN','Prob_NEUTRAL','Prob_UP']].to_numpy(float)
        for i,c in enumerate(['DOWN','NEUTRAL','UP']): d[f'{m}_Prob_{c}']=probs[:,i]
        d[f'{m}_Confidence']=probs[np.arange(n),p]
        sp=np.sort(probs,axis=1); d[f'{m}_Margin']=sp[:,-1]-sp[:,-2]
    pm=np.column_stack(preds); counts=np.column_stack([(pm==c).sum(1) for c in [0,1,2]])
    base=counts.argmax(1); votes=counts.max(1)
    d['Base_Class']=base; d['Base_Votes']=votes; d['Actual_Class']=actual; d['Base_Correct']=(base==actual).astype(int); d['Base_Wrong']=1-d['Base_Correct']
    probs=np.stack([fs[m][['Prob_DOWN','Prob_NEUTRAL','Prob_UP']].to_numpy(float) for m in MODELS],axis=1); mean=probs.mean(1); sd=probs.std(1)
    for i,c in enumerate(['DOWN','NEUTRAL','UP']): d[f'Mean_Prob_{c}']=mean[:,i]; d[f'Std_Prob_{c}']=sd[:,i]
    d['Prob_Max']=mean.max(1); d['Prob_Min']=mean.min(1); sm=np.sort(mean,axis=1); d['Prob_Margin']=sm[:,-1]-sm[:,-2]
    d['All_5_Agree']=(pm.min(1)==pm.max(1)).astype(int); d['Tree_Agree']=((pm[:,0]==pm[:,1])&(pm[:,1]==pm[:,2])).astype(int); d['Sequence_Agree']=(pm[:,3]==pm[:,4]).astype(int); d['Tree_Sequence_Disagree']=(d.Tree_Agree.eq(1)&d.Sequence_Agree.eq(1)&(pm[:,2]!=pm[:,3])).astype(int)
    for c in [0,1,2]: d[f'Votes_{LABELS[c]}']=(pm==c).sum(1)
    for m,p in zip(MODELS,preds):
        ok=(p==actual).astype(int); prior=pd.Series(ok).shift(1); d[f'{m}_Correct_Prev1']=prior; d[f'{m}_Error_Streak']=pd.Series(1-ok).shift(1).groupby(pd.Series(1-ok).shift(1).eq(0).cumsum()).cumsum().fillna(0)
        for w in [5,10,20]: d[f'{m}_Accuracy_Last_{w}']=pd.Series(ok).shift(1).rolling(w,min_periods=1).mean()
    d['Base_Error_Streak']=pd.Series(d.Base_Wrong).shift(1).groupby(pd.Series(d.Base_Wrong).shift(1).eq(0).cumsum()).cumsum().fillna(0)
    for w in [5,10,20]: d[f'Base_Accuracy_Last_{w}']=d.Base_Correct.shift(1).rolling(w,min_periods=1).mean()
    d['Hour']=d.Timestamp.dt.hour; d['Minute']=d.Timestamp.dt.minute; d['Minutes_From_Open']=d.Hour*60+d.Minute-(9*60+15)
    d=d.dropna().reset_index(drop=True)
    features=[c for c in d.columns if c not in {'Timestamp','Actual_Class','Base_Class','Base_Correct','Base_Wrong'}]
    train=d[d.Timestamp<pd.Timestamp('2023-10-01')]; val=d[d.Timestamp>=pd.Timestamp('2023-10-01')]
    model=HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=500,
        max_leaf_nodes=15,
        max_depth=5,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[features],train.Base_Wrong.astype(int)); pw=model.predict_proba(val[features])[:,1]; pred=(pw>=.5).astype(int); y=val.Base_Wrong.astype(int)
    metrics={'rows':len(val),'features':len(features),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'log_loss':log_loss(y,np.column_stack([1-pw,pw])),'roc_auc':roc_auc_score(y,pw),'base_mistake_rate':y.mean(),'confusion_matrix':confusion_matrix(y,pred).tolist(),'classification_report':classification_report(y,pred,target_names=['BASE_CORRECT','BASE_WRONG'],output_dict=True,zero_division=0)}
    out=val[['Timestamp','Base_Class','Actual_Class','Base_Correct','Base_Wrong']].copy(); out['P_Base_Wrong']=pw; out['Predicted_Base_Wrong']=pred
    sel=[]
    for t in [.5,.6,.7,.8,.9]:
        a=out[pw<t]
        if len(a): sel.append({'threshold':t,'accepted_rows':len(a),'coverage':len(a)/len(out),'base_accuracy':a.Base_Correct.mean(),'base_wrong_rate':1-a.Base_Correct.mean()})
    d.to_csv(OUT/'mistake_layer_v1_dataset.csv',index=False); out.to_csv(OUT/'mistake_layer_v1_verification_predictions.csv',index=False); pd.DataFrame(sel).to_csv(OUT/'mistake_layer_v1_selective_analysis.csv',index=False)
    report={'phase':'12A','base_models':MODELS,'base_decision':'strict majority','target':'Base_Wrong','discovery':'2023-01-01 to 2023-09-30','verification':'2023-10-01 to 2023-12-31','metrics':metrics,'selective_analysis':sel,'feature_count':len(features),'dataset_rows':len(d)}
    (OUT/'mistake_layer_v1_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('='*80); print('MISTAKE LAYER V1'); print('='*80); print(f'Common rows: {n:,}'); print(f'Dataset rows: {len(d):,}'); print(f'Features: {len(features)}'); print(f'Verification rows: {len(val):,}'); print(f'Accuracy: {metrics["accuracy"]:.6f}'); print(f'Balanced accuracy: {metrics["balanced_accuracy"]:.6f}'); print(f'Log loss: {metrics["log_loss"]:.6f}'); print(f'ROC-AUC: {metrics["roc_auc"]:.6f}'); print(f'Base mistake rate: {metrics["base_mistake_rate"]:.6f}'); print('\nSelective filtering:'); print(pd.DataFrame(sel).to_string(index=False)); print(f'\nSaved to: {OUT}')

if __name__=='__main__': main()

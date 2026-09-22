#!/usr/bin/env python3
"""Full-MISA nuisance bridge for the ICLR revision (offline only).

Unlike the original temporal-core probe, this keeps the complete MISA path:
three target scales, temporal normalization, scale normalization, leader BiGRU,
cross-attention, gate, and readout. Independent nuisance steps are PREPENDED to
the target branch only; r is a multiple of 6 so pooling blocks never mix noise
and real observations. The original 30 real steps remain the most recent context.

The experiment measures both attention mass and *functional* nuisance
sensitivity Psi = E_x Var_z[f(x,z)] by repeatedly resampling nuisance after
training with the model frozen. This does not claim Theorem 1 applies to the
BiGRU; it directly tests the full trained predictor.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'package'
import sys
sys.path.insert(0,str(PKG))
from misa_sparsity import MISABiGRU, Trainer, build_features, get_device
import misa_sparsity.run_ablation as ra

DATA=PKG/'data'   # overridden by --data-dir
OUT=ROOT/'results'/'full_misa_mechanism.csv'
RAW=ROOT/'results'/'full_misa_mechanism.json'
CKPT=ROOT/'results'/'full_misa_checkpoints'
DEFAULT_SETTINGS=['BTC_P2','ETH_P2','SAND_P2']
DEFAULT_SEEDS=[42,43,44]
DEFAULT_R=[0,12]

CFG=dict(window=30,horizon=5,hidden=64,n_heads=4,dropout=.25,lambda_gate=.05,
         lr=5e-4,epochs=150,patience=20,warmup=20,batch_size=64,train_ratio=.70,val_frac=.10,n_features=7)
DATE_TAG={'P1':'20221101','P2':'20231001','P3':'20240301'}
CA_MAP={'BTC':'ETH'}          # every other asset uses BTC as the leader series


def leader(asset):
    return CA_MAP.get(asset,'BTC')


def path(asset,period): return DATA/f'{asset}_5m_{period}_{DATE_TAG[period]}.csv'


def make_windows(close, feat_tgt, feat_ca):
    log_ret=np.diff(np.log(np.maximum(close,1e-10)))
    n=min(len(feat_tgt),len(feat_ca)); X=[]; Xca=[]; y=[]; pc=[]
    w=CFG['window']; h=CFG['horizon']
    for i in range(n-w-h+1):
        X.append(feat_tgt[i:i+w]); Xca.append(feat_ca[i:i+w])
        y.append(log_ret[i+w-1:i+w+h-1].sum()); pc.append(close[i+w-1])
    return np.array(X,np.float32),np.array(Xca,np.float32),np.array(y,np.float32),np.array(pc,np.float32)


def prepare(setting:str):
    asset,period=setting.rsplit('_',1); ca=leader(asset)
    close,ft=build_features(str(path(asset,period)),CFG['n_features'])
    _,fc=build_features(str(path(ca,period)),CFG['n_features'])
    X,Xca,y,pc=make_windows(close,ft,fc)
    n=len(y); nt=int(n*CFG['train_ratio']); nv=int(n*CFG['val_frac']); F=X.shape[-1]
    st,sc=MinMaxScaler(),MinMaxScaler(); st.fit(X[:nt].reshape(-1,F)); sc.fit(Xca[:nt].reshape(-1,F))
    X=st.transform(X.reshape(-1,F)).reshape(X.shape).astype(np.float32)
    Xca=sc.transform(Xca.reshape(-1,F)).reshape(Xca.shape).astype(np.float32)
    mu=X[:nt].reshape(-1,F).mean(0); sd=X[:nt].reshape(-1,F).std(0)+1e-8
    return dict(asset=asset,period=period,F=F,X=X,Xca=Xca,y=y,pc=pc,nt=nt,nv=nv,
                mu=mu.astype(np.float32),sd=sd.astype(np.float32),price_range=float(close.max()-close.min()))


def prepend_noise(A,r,mu,sd,rng):
    if r==0: return A.copy()
    z=rng.normal(mu,sd,size=(len(A),r,A.shape[-1])).astype(np.float32)
    return np.concatenate([z,A],axis=1)


def split_with_fixed_noise(base,r,setting_seed):
    rng=np.random.default_rng(setting_seed)
    Xin=prepend_noise(base['X'],r,base['mu'],base['sd'],rng)
    nt,nv=base['nt'],base['nv']
    return dict(
      tr=(Xin[:nt],base['Xca'][:nt],base['y'][:nt]),
      val=(Xin[nt:nt+nv],base['Xca'][nt:nt+nv],base['y'][nt:nt+nv]),
      te=(Xin[nt+nv:],base['Xca'][nt+nv:],base['y'][nt+nv:]),
      real_te=base['X'][nt+nv:], xca_te=base['Xca'][nt+nv:], y_te=base['y'][nt+nv:],
      pte=base['pc'][nt+nv:],F=base['F'],mu=base['mu'],sd=base['sd'],
      price_range=base['price_range'],r=r)


def diagnostic_weights(model,X,Xca,r):
    model.eval(); dev=next(model.parameters()).device
    with torch.no_grad():
        d=model.forward_diagnostics(torch.tensor(X,device=dev),torch.tensor(Xca,device=dev))
    if r==0:
        return dict(attn_s1=0.,attn_s2=0.,attn_s3=0.,effective_attn=0.,gate_target=float(d['gate_target'].mean()) if d['gate_target'] is not None else np.nan)
    r1,r2,r3=r,r//3,r//6
    m1=d['a1'][:,:r1].sum(1)
    m2=d['a2'][:,:r2].sum(1)
    m3=d['a3'][:,:r3].sum(1)
    masses=torch.stack([m1,m2,m3],dim=1)
    eff=(masses*d['scale_weights']).sum(1)
    return dict(attn_s1=float(m1.mean()),attn_s2=float(m2.mean()),attn_s3=float(m3.mean()),
                effective_attn=float(eff.mean()),gate_target=float(d['gate_target'].mean()) if d['gate_target'] is not None else np.nan)


def nuisance_sensitivity(model,realX,Xca,mu,sd,r,K=12,seed=0,batch=1024):
    if r==0: return 0.,0.
    rng=np.random.default_rng(seed); preds=[]; model.eval(); dev=next(model.parameters()).device
    for k in range(K):
        X=prepend_noise(realX,r,mu,sd,rng)
        chunks=[]
        with torch.no_grad():
            for i in range(0,len(X),batch):
                p,_,_=model(torch.tensor(X[i:i+batch],device=dev),torch.tensor(Xca[i:i+batch],device=dev))
                chunks.append(p.cpu().numpy())
        preds.append(np.concatenate(chunks))
    P=np.stack(preds,axis=0)
    psi=float(np.var(P,axis=0,ddof=1).mean())
    yvar=float(np.var(P.mean(axis=0),ddof=1))
    return psi, yvar


def signal_sensitivity(model,realX,Xca,r,mu,sd,K=12,seed=0,batch=1024):
    """Psi_signal: prediction variance when the *real* context is resampled.

    Psi_noise alone cannot establish selective insensitivity, because a model
    that ignores its input entirely also has a small Psi_noise. This control
    permutes the whole real context across held-out windows -- both the target
    window and the leader window, jointly, so their pairing is destroyed -- while
    keeping every marginal distribution intact and leaving the injected nuisance
    untouched. A model that uses its context has Psi_signal >> Psi_noise; the
    ratio Psi_signal / Psi_noise is reported as the selectivity ratio.
    """
    rng=np.random.default_rng(seed); preds=[]; model.eval(); dev=next(model.parameters()).device
    for k in range(K):
        perm=rng.permutation(len(realX))
        Xs=realX[perm]; Xc=Xca[perm]
        X=prepend_noise(Xs,r,mu,sd,rng)
        chunks=[]
        with torch.no_grad():
            for i in range(0,len(X),batch):
                p,_,_=model(torch.tensor(X[i:i+batch],device=dev),torch.tensor(Xc[i:i+batch],device=dev))
                chunks.append(p.cpu().numpy())
        preds.append(np.concatenate(chunks))
    P=np.stack(preds,axis=0)
    return float(np.var(P,axis=0,ddof=1).mean())


def train_one(setting,r,norm,seed,K,device="cpu",save_checkpoint=True):
    base=prepare(setting)
    data_seed=100000+sum(map(ord,setting))+100*r
    sp=split_with_fixed_noise(base,r,data_seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model=MISABiGRU(feat_dim=sp['F'],hidden=CFG['hidden'],n_heads=CFG['n_heads'],dropout=CFG['dropout'],
                    lambda_gate=CFG['lambda_gate'],use_multiscale=True,use_cross_asset=True,use_sparsemax=(norm=='sparsemax'))
    tr=Trainer(model,lr=CFG['lr'],epochs=CFG['epochs'],patience=CFG['patience'],warmup=CFG['warmup'],
               batch_size=CFG['batch_size'],device=device,verbose=False)
    t0=time.time(); tr.fit(*sp['tr'],*sp['val']); seconds=time.time()-t0
    metrics=tr.evaluate(*sp['te'],sp['pte'],price_range=sp['price_range'])
    diag=diagnostic_weights(model,sp['te'][0],sp['te'][1],r)
    psi,predvar=nuisance_sensitivity(model,sp['real_te'],sp['xca_te'],sp['mu'],sp['sd'],r,K=K,seed=seed+9000)
    psi_sig=signal_sensitivity(model,sp['real_te'],sp['xca_te'],r,sp['mu'],sp['sd'],K=K,seed=seed+7000)
    if save_checkpoint:
        CKPT.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'setting': setting, 'r': r, 'normalization': norm, 'seed': seed, 'cfg': CFG},
                   CKPT/f'{setting}_r{r}_{norm}_seed{seed}.pt')
    yvar=float(np.var(sp['y_te'],ddof=1))
    return dict(setting=setting,r=r,normalization=norm,seed=seed,n_test=len(sp['y_te']),
                nmse=float(metrics['NMSE']),rmse=float(metrics['RMSE']),r2=float(metrics['R2']),
                psi_return=psi,psi_over_yvar=psi/(yvar+1e-18),prediction_variance=predvar,
                psi_signal=psi_sig,selectivity_ratio=psi_sig/(psi+1e-18),
                train_seconds=seconds,**diag)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--settings',nargs='*',default=DEFAULT_SETTINGS)
    ap.add_argument('--seeds',nargs='*',type=int,default=DEFAULT_SEEDS)
    ap.add_argument('--r',nargs='*',type=int,default=DEFAULT_R)
    ap.add_argument('--K',type=int,default=12,help='Frozen nuisance resamples per held-out context.')
    ap.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    ap.add_argument('--threads',type=int,default=6,help='CPU intra-op threads.')
    ap.add_argument('--epochs',type=int,default=CFG['epochs'])
    ap.add_argument('--patience',type=int,default=CFG['patience'])
    ap.add_argument('--warmup',type=int,default=CFG['warmup'])
    ap.add_argument('--batch-size',type=int,default=CFG['batch_size'])
    ap.add_argument('--no-checkpoints',action='store_true')
    ap.add_argument('--data-dir',default=None,
                    help='Directory holding the <ASSET>_5m_<PERIOD>_<DATE>.csv files. '
                         'Defaults to package/data, which ships only three assets.')
    ap.add_argument('--out-tag',default=None,help='Suffix for the output files.')
    args=ap.parse_args()
    global DATA,OUT,RAW,CKPT
    if args.data_dir: DATA=Path(args.data_dir).expanduser()
    if args.out_tag:
        OUT=ROOT/'results'/f'full_misa_mechanism_{args.out_tag}.csv'
        RAW=ROOT/'results'/f'full_misa_mechanism_{args.out_tag}.json'
        CKPT=ROOT/'results'/f'full_misa_checkpoints_{args.out_tag}'
    torch.set_num_threads(args.threads)
    device=get_device(args.device)
    CFG.update(epochs=args.epochs,patience=args.patience,warmup=args.warmup,batch_size=args.batch_size)
    print(f'device={device} epochs={CFG["epochs"]} patience={CFG["patience"]} warmup={CFG["warmup"]} batch={CFG["batch_size"]}',flush=True)
    for r in args.r:
        if r%6: raise ValueError('Use r multiples of 6 so pooled noise blocks do not mix with real steps.')
    existing=[]
    if RAW.exists(): existing=json.load(open(RAW))
    index={(z['setting'],z['r'],z['normalization'],z['seed']):z for z in existing}
    for setting in args.settings:
        for r in args.r:
            for seed in args.seeds:
                for norm in ['sparsemax','softmax']:
                    key=(setting,r,norm,seed)
                    if key in index: continue
                    print('RUN',key,flush=True)
                    row=train_one(setting,r,norm,seed,args.K,device=device,save_checkpoint=not args.no_checkpoints); existing.append(row); index[key]=row
                    RAW.write_text(json.dumps(existing,indent=2),encoding='utf-8')
                    import pandas as pd
                    pd.DataFrame(existing).to_csv(OUT,index=False)
                    print(f"  NMSE={row['nmse']:.5f} eff_attn={row['effective_attn']:.4f} Psi/yvar={row['psi_over_yvar']:.3e} time={row['train_seconds']:.1f}s",flush=True)
    import pandas as pd
    df=pd.DataFrame(existing); df.to_csv(OUT,index=False)
    if not df.empty:
        s=(df.groupby(['setting','r','normalization'])[['nmse','effective_attn','psi_return','psi_over_yvar']].agg(['mean','std']))
        print('\nSUMMARY\n',s.to_string())

if __name__=='__main__': main()

"""Fruit Ripeness Mini Project 2: six reproducible classification experiments.

Run from a notebook by importing this module, or run directly:
  python SRN1_SRN2_SRN3.py --zip ip_miniproject.zip --work-dir work
The first run may download ImageNet weights for ResNet50 and MViT from PyTorch/timm.
"""
from __future__ import annotations
import argparse, os, random, shutil, zipfile
from pathlib import Path
from collections import Counter
import numpy as np

CLASSES = ["Unripe", "Semi-ripe", "Ripe", "Overripe"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError: pass


def extract_dataset(zip_path, extract_dir):
    """Extract ZIP safely; return actual extracted root and print its tree."""
    zip_path, extract_dir = Path(zip_path), Path(extract_dir)
    if not zip_path.is_file(): raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    # prevent ZIP path traversal
    base = extract_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (extract_dir / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
        zf.extractall(extract_dir)
    print(f"Extracted {zip_path.name} to {extract_dir.resolve()}")
    print_directory_tree(extract_dir)
    return locate_raw_root(extract_dir)


def print_directory_tree(root, max_depth=4):
    root = Path(root); print(f"Dataset tree (to depth {max_depth}): {root}")
    for current, dirs, files in os.walk(root):
        rel = Path(current).relative_to(root); depth = len(rel.parts)
        dirs[:] = sorted(d for d in dirs if d != "__MACOSX")
        if depth > max_depth: dirs[:] = []; continue
        print("  " * depth + (rel.name if depth else root.name) + f"/ ({len(files)} files)")


def locate_raw_root(root):
    root = Path(root)
    candidates = [p for p in root.rglob("raw") if p.is_dir()]
    for candidate in candidates:
        names = {p.name.casefold() for p in candidate.iterdir() if p.is_dir()}
        expected = {c.casefold() for c in CLASSES}
        if expected.issubset(names): return candidate
    # Also accept a class-folder root directly.
    names = {p.name.casefold() for p in root.iterdir() if p.is_dir()} if root.exists() else set()
    if {c.casefold() for c in CLASSES}.issubset(names): return root
    raise FileNotFoundError("Could not locate raw image root containing folders Unripe, Semi-ripe, Ripe, Overripe")


def load_samples(raw_root):
    """Read original files only from raw/<class>; unreadable images are skipped."""
    import cv2
    raw_root = Path(raw_root); samples = []
    for cls in CLASSES:
        folder = next((p for p in raw_root.iterdir() if p.is_dir() and p.name.casefold() == cls.casefold()), None)
        if folder is None: raise FileNotFoundError(f"Missing class folder {cls} under {raw_root}")
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTS: continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"WARNING: unreadable image skipped: {path}"); continue
            samples.append({"path": path, "label": CLASS_TO_ID[cls], "class_name": cls, "image": image})
    if not samples: raise ValueError(f"No readable raw images found in {raw_root}")
    print(f"Loaded {len(samples)} readable original images; counts: {Counter(s['class_name'] for s in samples)}")
    print("Fruit varieties found in filenames:", sorted({s['path'].name.split('_')[0] for s in samples}))
    return samples


def make_split(samples, test_size=.20, seed=SEED):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(samples)); y = np.array([s['label'] for s in samples])
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    train, test = [samples[i] for i in train_idx], [samples[i] for i in test_idx]
    train_ids, test_ids = {str(s['path'].resolve()) for s in train}, {str(s['path'].resolve()) for s in test}
    assert not train_ids.intersection(test_ids)
    print("Train counts:", Counter(s['class_name'] for s in train))
    print("Test counts: ", Counter(s['class_name'] for s in test))
    print(f"Split: {len(train)} train / {len(test)} test, seed={seed}; originals only")
    return train, test


def show_samples(samples, n=8, seed=SEED):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed); chosen = rng.choice(samples, min(n, len(samples)), replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6));
    for ax, s in zip(axes.flat, chosen):
        ax.imshow(__import__('cv2').cvtColor(s['image'], __import__('cv2').COLOR_BGR2RGB)); ax.set_title(s['class_name']); ax.axis('off')
    for ax in axes.flat[len(chosen):]: ax.axis('off')
    plt.tight_layout(); plt.show()


def sift_descriptors(samples, max_dim=640):
    import cv2
    sift = cv2.SIFT_create(nfeatures=500)
    results = []
    for s in samples:
        img = s['image']; h,w = img.shape[:2]; scale=min(1.0,max_dim/max(h,w))
        if scale < 1: img=cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _, desc=sift.detectAndCompute(gray,None)
        results.append(desc if desc is not None else np.empty((0,128),np.float32))
    return results


def fit_bovw(train_desc, vocab_size=64, seed=SEED, max_desc_per_image=300):
    from sklearn.cluster import MiniBatchKMeans
    pieces=[]
    rng=np.random.default_rng(seed)
    for desc in train_desc:
        if len(desc): pieces.append(desc[rng.choice(len(desc),min(max_desc_per_image,len(desc)),replace=False)])
    if not pieces: raise ValueError("SIFT found no descriptors in training images")
    all_desc=np.vstack(pieces)
    k=min(vocab_size,len(all_desc))
    model=MiniBatchKMeans(n_clusters=k,random_state=seed,batch_size=min(2048,max(256,k*4)),n_init=3)
    model.fit(all_desc)
    return model


def bovw_transform(descs, vocab):
    out=[]
    for d in descs:
        hist=np.zeros(vocab.n_clusters,dtype=np.float32)
        if len(d): hist += np.bincount(vocab.predict(d),minlength=vocab.n_clusters)
        norm=np.linalg.norm(hist)
        if norm: hist/=norm
        out.append(hist)
    return np.vstack(out)


def harris_features(samples, size=(128,128)):
    import cv2
    features=[]
    for s in samples:
        gray=cv2.cvtColor(s['image'],cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,size,interpolation=cv2.INTER_AREA).astype(np.float32)/255.
        response=cv2.cornerHarris(gray,2,3,.04)
        threshold=max(float(response.max())*.01,1e-8)
        ys,xs=np.where(response>threshold)
        # Fixed 4x4 spatial grid of normalized corner counts, plus global Harris statistics.
        grid=np.zeros((4,4),np.float32)
        if len(xs):
            gx=np.minimum(xs*4//size[0],3); gy=np.minimum(ys*4//size[1],3)
            np.add.at(grid,(gy,gx),1)
        grid/=max(grid.sum(),1)
        positive=np.maximum(response,0)
        stats=np.array([len(xs)/(size[0]*size[1]), positive.mean(), positive.std(), positive.max(), np.quantile(positive,.9)],np.float32)
        # Append HSV color histogram as complementary fruit surface/color information.
        hsv=cv2.cvtColor(s['image'],cv2.COLOR_BGR2HSV)
        color=cv2.calcHist([hsv],[0,1],None,[8,4],[0,180,0,256]).flatten().astype(np.float32)
        color/=max(np.linalg.norm(color),1e-8)
        features.append(np.concatenate([grid.flatten(),stats,color]))
    return np.vstack(features)


def cosine_prototype_predict(x_train,y_train,x_test):
    from sklearn.preprocessing import normalize
    prototypes=[]
    for c in range(len(CLASSES)):
        rows=x_train[y_train==c]
        if len(rows)==0: raise ValueError(f"No training examples for class {CLASSES[c]}")
        prototypes.append(rows.mean(axis=0))
    p=normalize(np.vstack(prototypes)); x=normalize(x_test)
    return (x @ p.T).argmax(axis=1)


def evaluate(name,y_true,y_pred):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
    return {"Experiment":name,"Accuracy":accuracy_score(y_true,y_pred),"Macro Precision":precision_score(y_true,y_pred,average='macro',zero_division=0),"Macro Recall":recall_score(y_true,y_pred,average='macro',zero_division=0),"Macro F1":f1_score(y_true,y_pred,average='macro',zero_division=0),"Confusion Matrix":confusion_matrix(y_true,y_pred,labels=np.arange(4))}


def classic_experiments(train,test,progress=True):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from tqdm.auto import tqdm
    ytr=np.array([s['label'] for s in train]); yte=np.array([s['label'] for s in test]); results=[]
    print("Extracting SIFT descriptors...")
    trd=sift_descriptors(tqdm(train,desc='SIFT train',disable=not progress)); ted=sift_descriptors(tqdm(test,desc='SIFT test',disable=not progress))
    vocab=fit_bovw(trd)
    xtr=bovw_transform(trd,vocab); xte=bovw_transform(ted,vocab)
    svm=make_pipeline(StandardScaler(),SVC(kernel='rbf',class_weight='balanced',random_state=SEED))
    svm.fit(xtr,ytr); results.append(evaluate('SIFT + SVM',yte,svm.predict(xte)))
    results.append(evaluate('SIFT + Cosine Similarity',yte,cosine_prototype_predict(xtr,ytr,xte)))
    print("Extracting Harris corner features...")
    htr=harris_features(tqdm(train,desc='Harris train',disable=not progress)); hte=harris_features(tqdm(test,desc='Harris test',disable=not progress))
    hmodel=make_pipeline(StandardScaler(),SVC(kernel='rbf',class_weight='balanced',random_state=SEED)); hmodel.fit(htr,ytr)
    results.append(evaluate('Harris + SVM',yte,hmodel.predict(hte)))
    # Scaling parameters learned on training data only; cosine uses class prototypes in that scaled space.
    scaler=StandardScaler().fit(htr); htr_s=scaler.transform(htr); hte_s=scaler.transform(hte)
    results.append(evaluate('Harris + Cosine Similarity',yte,cosine_prototype_predict(htr_s,ytr,hte_s)))
    return results


def _torch_image_data(samples):
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    class FruitDataset(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self,i):
            s=samples[i]; im=Image.open(s['path']).convert('RGB')
            return transform(im),int(s['label'])
    return FruitDataset()


def _train_head(backbone, head, train, test, device, epochs=8, batch_size=32, lr=1e-3, model_name='model'):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    tr_loader=DataLoader(_torch_image_data(train),batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=(device.type=='cuda'))
    te_loader=DataLoader(_torch_image_data(test),batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=(device.type=='cuda'))
    backbone.to(device).eval(); head.to(device).train()
    optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=1e-4)
    loss_fn=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        head.train(); total=0.; seen=0
        for xb,yb in tr_loader:
            xb,yb=xb.to(device),yb.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): feats=backbone(xb)
            logits=head(feats); loss=loss_fn(logits,yb); loss.backward(); optimizer.step()
            total+=loss.item()*len(yb); seen+=len(yb)
        print(f"{model_name} epoch {epoch+1}/{epochs}: train loss={total/max(seen,1):.4f}")
    backbone.eval(); head.eval(); ys=[]; preds=[]
    with torch.no_grad():
        for xb,yb in te_loader:
            logits=head(backbone(xb.to(device)))
            if logits.shape[1]!=4: raise RuntimeError(f"Expected four output classes, received {logits.shape[1]}")
            preds.extend(logits.argmax(1).cpu().numpy()); ys.extend(yb.numpy())
    return evaluate(model_name,np.array(ys),np.array(preds)),backbone,head


def run_resnet(train,test,device=None,epochs=8,batch_size=32):
    import torch
    from torch import nn
    try:
        from torchvision.models import resnet50, ResNet50_Weights
        model=resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception as exc:
        raise RuntimeError("Could not load pretrained ResNet50 ImageNet weights. Check torchvision installation and network/cache access.") from exc
    for p in model.parameters(): p.requires_grad=False
    in_features=model.fc.in_features; model.fc=nn.Identity()
    head=nn.Linear(in_features,4)
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _train_head(model,head,train,test,device,epochs,batch_size,model_name='ResNet50 (frozen backbone)')


def run_mvit(train,test,device=None,epochs=8,batch_size=16,model_name='mvitv2_small.fb_in1k'):
    import torch
    from torch import nn
    try: import timm
    except ImportError as exc: raise RuntimeError("MViT requires timm. Install with: pip install timm") from exc
    if 'mvit' not in model_name.casefold(): raise ValueError("The requested architecture must be an MViT; refusing architecture substitution")
    try: backbone=timm.create_model(model_name,pretrained=True,num_classes=0)
    except Exception as exc: raise RuntimeError(f"Could not load pretrained MViT '{model_name}'. Confirm this exact model is available in your timm version and weights can be downloaded.") from exc
    architecture_id = (backbone.__class__.__name__ + " " + backbone.__class__.__module__).casefold()
    if "mvit" not in architecture_id and "multiscalevit" not in architecture_id:
        raise RuntimeError(f"Loaded module {type(backbone).__module__}.{type(backbone).__name__} is not identifiable as MViT; stopping")
    for p in backbone.parameters(): p.requires_grad=False
    features=int(backbone.num_features); head=nn.Linear(features,4)
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _train_head(backbone,head,train,test,device,epochs,batch_size,model_name=f"MViT ({model_name})")


def plot_results(results):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import ConfusionMatrixDisplay
    table=pd.DataFrame([{k:v for k,v in r.items() if k!='Confusion Matrix'} for r in results])
    print(table.to_string(index=False))
    fig,ax=plt.subplots(figsize=(10,5)); ax.bar(table['Experiment'],table['Accuracy'],color='#4086a8'); ax.set_ylim(0,1); ax.set_ylabel('Accuracy'); ax.set_title('Held-out test accuracy'); ax.tick_params(axis='x',rotation=25); plt.tight_layout(); plt.show()
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for ax,r in zip(axes.flat,results):
        ConfusionMatrixDisplay(r['Confusion Matrix'],display_labels=CLASSES).plot(ax=ax,colorbar=False,values_format='d'); ax.set_title(r['Experiment']); ax.tick_params(axis='x',rotation=30)
    plt.tight_layout(); plt.show()
    return table


def run_all(zip_path,work_dir='work',epochs=8,run_deep=True):
    seed_everything(); raw=extract_dataset(zip_path,Path(work_dir)/'dataset_extracted'); samples=load_samples(raw)
    show_samples(samples); train,test=make_split(samples)
    results=classic_experiments(train,test)
    if run_deep:
        import torch
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:',device,'GPU:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'not available')
        rr,_,_=run_resnet(train,test,device,epochs); results.append(rr)
        mr,_,_=run_mvit(train,test,device,epochs); results.append(mr)
    plot_results(results)
    return results,train,test


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',required=True,help='Path to ip_miniproject.zip')
    parser.add_argument('--work-dir',default='work'); parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--skip-deep',action='store_true',help='Run four traditional experiments only')
    args=parser.parse_args(); run_all(args.zip,args.work_dir,args.epochs,not args.skip_deep)

if __name__=='__main__': main()

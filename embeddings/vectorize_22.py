"""22 text representations for FakeNewsNet JSON/JSONL.
Python 3.10-3.12 recommended. Optional methods report SKIPPED if unavailable.
Usage: python vectorize_22.py --input fakenewsnet-unbalanced.json --column text --out vectors --max-docs 500
       python vectorize_22.py --input data.json --methods tfidf,sbert,bge
       ./.venv312/Scripts/python.exe vectorize_22.py --input ../fakenewsnet-unbalanced.json --methods word2vec,fasttext,doc2vec --max-docs 500
Install baseline: pip install numpy pandas scipy scikit-learn joblib
Extras: pip install gensim sentence-transformers transformers torch nltk
For 3.14, native dependency availability varies; use 3.11 venv.
"""
import argparse, importlib, json, re, traceback
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.preprocessing import normalize
from joblib import dump


METHODS = ['bow','binary_bow','ngram','char_ngram','tfidf','hashing','bm25','log_entropy','ppmi',
           'word2vec','glove','fasttext','doc2vec','lsa','nmf','lda','hdp',
           'transformer','sbert','simcse','e5','bge']

def load_texts(path, column):
    p=Path(path)
    with p.open(encoding='utf-8-sig') as f:
        start=f.read(2048).lstrip(); f.seek(0)
        if start.startswith('['):
            rows=json.load(f)
        else:
            rows=[json.loads(line) for line in f if line.strip()]
    return [str(r.get(column) or '') for r in rows]

def toks(text): return re.findall(r"(?u)\b\w+\b", text.lower())
def save_matrix(out, name, x, model=None):
    p=out/name; p.mkdir(exist_ok=True,parents=True)
    if sparse.issparse(x):
        sparse.save_npz(p/'vectors.npz', x.tocsr()); shape=x.shape; fmt='sparse_npz'
    else:
        x=np.asarray(x,dtype=np.float32)
        np.save(p/'vectors.npy',x); shape=x.shape; fmt='dense_npy'
    if model is not None:
        try: dump(model,p/'model.joblib')
        except Exception: pass
    (p/'metadata.json').write_text(json.dumps({'method':name,'shape':list(shape),'format':fmt},indent=2),encoding='utf-8')
    return shape

def word_vectors(texts, name, dim, seed):
    from gensim.models import Word2Vec, FastText, Doc2Vec
    from gensim.models.doc2vec import TaggedDocument
    sentences=[toks(t) for t in texts]
    if name=='word2vec':
        model=Word2Vec(sentences=sentences,vector_size=dim,min_count=1,workers=1,epochs=10,seed=seed)
        x=np.stack([np.mean([model.wv[w] for w in s],axis=0) if s else np.zeros(dim) for s in sentences]); return x,model
    if name=='fasttext':
        model=FastText(sentences=sentences,vector_size=dim,min_count=1,workers=1,epochs=10,seed=seed)
        x=np.stack([np.mean([model.wv[w] for w in s],axis=0) if s else np.zeros(dim) for s in sentences]); return x,model
    docs=[TaggedDocument(words=s,tags=[i]) for i,s in enumerate(sentences)]
    model=Doc2Vec(documents=docs,vector_size=dim,min_count=1,workers=1,epochs=20,seed=seed)
    return np.stack([model.dv[i] for i in range(len(docs))]),model

def glove_vectors(texts, path, dim):
    """Requires user-supplied pre-trained GloVe txt; averages word vectors."""
    if not path: raise ValueError('GloVe requires --glove-file /path/to/glove.6B.100d.txt')
    vocab={w for t in texts for w in toks(t)}; vectors={}
    with open(path,encoding='utf-8') as f:
        for line in f:
            parts=line.rstrip().split(' ')
            if parts[0] in vocab:
                try: vectors[parts[0]]=np.asarray(parts[1:],dtype=np.float32)
                except ValueError: pass
    if not vectors: raise ValueError('No GloVe vectors matched input tokens')
    size=len(next(iter(vectors.values())))
    x=np.stack([np.mean([vectors[w] for w in toks(t) if w in vectors],axis=0) if any(w in vectors for w in toks(t)) else np.zeros(size) for t in texts])
    return x,None

def embed_hf(texts, model_id, batch_size, max_chars):
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(model_id)
    # Char truncation is only a safety bound; model tokenizer may truncate further.
    return model.encode([t[:max_chars] for t in texts],batch_size=batch_size,show_progress_bar=True,convert_to_numpy=True),None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True); ap.add_argument('--column',default='text'); ap.add_argument('--out',default='vectors_22')
    ap.add_argument('--methods',default='all',help='comma-separated method IDs or all')
    ap.add_argument('--max-docs',type=int,default=0); ap.add_argument('--max-features',type=int,default=20000)
    ap.add_argument('--dim',type=int,default=100); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--glove-file',default=None)
    ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--max-chars',type=int,default=3000)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    texts=load_texts(args.input,args.column)
    if args.max_docs: texts=texts[:args.max_docs]
    if not texts: raise ValueError('No documents')
    selected=METHODS if args.methods=='all' else [s.strip() for s in args.methods.split(',')]
    unknown=set(selected)-set(METHODS)
    if unknown: raise ValueError(f'Unknown methods: {unknown}; available: {METHODS}')
    report=[]
    for name in selected:
        print(f'\n[{name}] starting',flush=True)
        try:
            model=None
            if name=='bow':
                model=CountVectorizer(max_features=args.max_features); x=model.fit_transform(texts)
            elif name=='binary_bow':
                model=CountVectorizer(max_features=args.max_features,binary=True); x=model.fit_transform(texts)
            elif name=='ngram':
                model=CountVectorizer(ngram_range=(1,2),max_features=args.max_features); x=model.fit_transform(texts)
            elif name=='char_ngram':
                model=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=args.max_features); x=model.fit_transform(texts)
            elif name=='tfidf':
                model=TfidfVectorizer(ngram_range=(1,2),max_features=args.max_features); x=model.fit_transform(texts)
            elif name=='hashing':
                model=HashingVectorizer(n_features=args.max_features,alternate_sign=False); x=model.transform(texts)
            elif name=='bm25':
                # BM25 term-weight features, not retrieval ranking. Sparse matrix.
                cv=CountVectorizer(max_features=args.max_features); counts=cv.fit_transform(texts).tocsr().astype(float)
                n=counts.shape[0]; df=np.asarray((counts>0).sum(axis=0)).ravel()
                idf=np.log(1+(n-df+0.5)/(df+0.5)); lengths=np.asarray(counts.sum(axis=1)).ravel(); avg=max(lengths.mean(),1e-9)
                k1=1.5; b=0.75; coo=counts.tocoo()
                vals=idf[coo.col]*coo.data*(k1+1)/(coo.data+k1*(1-b+b*lengths[coo.row]/avg))
                x=sparse.csr_matrix((vals,(coo.row,coo.col)),shape=counts.shape); model={'vectorizer':cv,'idf':idf,'k1':k1,'b':b,'avgdl':avg}
            elif name=='log_entropy':
                cv=CountVectorizer(max_features=args.max_features); counts=cv.fit_transform(texts).tocsr().astype(float)
                totals=np.asarray(counts.sum(axis=0)).ravel(); coo=counts.tocoo()
                probs=coo.data/np.maximum(totals[coo.col],1e-12)
                h=np.bincount(coo.col,weights=probs*np.log(np.maximum(probs,1e-12)),minlength=counts.shape[1])
                weights=1+h/max(np.log(max(counts.shape[0],2)),1e-9)
                x=counts.copy(); x.data=np.log1p(x.data); x=x.multiply(weights).tocsr(); model={'vectorizer':cv,'global_weights':weights}
            elif name=='ppmi':
                # Word-word cooccurrence PPMI -> SVD word embeddings -> mean document embedding.
                cv=CountVectorizer(max_features=min(args.max_features,3000),binary=True)
                a=cv.fit_transform(texts).astype(float); co=(a.T@a).tocsr(); co.setdiag(0); co.eliminate_zeros()
                total=co.sum(); row=np.asarray(co.sum(axis=1)).ravel(); c=co.tocoo()
                vals=np.maximum(np.log((c.data*total+1e-12)/(row[c.row]*row[c.col]+1e-12)),0)
                pp=sparse.csr_matrix((vals,(c.row,c.col)),shape=co.shape)
                k=min(args.dim,max(1,pp.shape[1]-1)); svd=TruncatedSVD(n_components=k,random_state=args.seed)
                w=svd.fit_transform(pp); x=normalize(a@w); model={'vectorizer':cv,'svd':svd,'word_embeddings':w}
            elif name in ('word2vec','fasttext','doc2vec'):
                x,model=word_vectors(texts,name,args.dim,args.seed)
            elif name=='glove':
                x,model=glove_vectors(texts,args.glove_file,args.dim)
            elif name=='lsa':
                tf=TfidfVectorizer(max_features=args.max_features); a=tf.fit_transform(texts)
                k=min(args.dim,max(1,min(a.shape)-1)); svd=TruncatedSVD(n_components=k,random_state=args.seed)
                x=svd.fit_transform(a); model={'tfidf':tf,'svd':svd}
            elif name=='nmf':
                tf=TfidfVectorizer(max_features=args.max_features); a=tf.fit_transform(texts)
                k=min(args.dim,max(1,min(a.shape))); nmf=NMF(n_components=k,init='nndsvda',max_iter=1000,random_state=args.seed)
                x=nmf.fit_transform(a); model={'tfidf':tf,'nmf':nmf}
            elif name=='lda':
                cv=CountVectorizer(max_features=args.max_features); a=cv.fit_transform(texts)
                k=min(args.dim,max(1,a.shape[1])); lda=LatentDirichletAllocation(n_components=k,random_state=args.seed,learning_method='batch')
                x=lda.fit_transform(a); model={'count_vectorizer':cv,'lda':lda}
            elif name=='hdp':
                from gensim.corpora import Dictionary
                from gensim.models import HdpModel
                tokenized=[toks(t) for t in texts]; dictionary=Dictionary(tokenized); dictionary.filter_extremes(no_below=2,no_above=0.95,keep_n=args.max_features)
                corpus=[dictionary.doc2bow(t) for t in tokenized]
                if len(dictionary)==0: raise ValueError('HDP vocabulary empty')
                model=HdpModel(corpus=corpus,id2word=dictionary)
                topics=[model[c] for c in corpus]; d=max((i for doc in topics for i,_ in doc),default=-1)+1
                x=np.zeros((len(texts),d),dtype=np.float32)
                for j,doc in enumerate(topics):
                    for i,v in doc: x[j,i]=v
            elif name in ('transformer','sbert','simcse','e5','bge'):
                ids={'transformer':'bert-base-uncased',
                     'sbert':'sentence-transformers/all-MiniLM-L6-v2',
                     'simcse':'princeton-nlp/sup-simcse-bert-base-uncased',
                     'e5':'intfloat/e5-small-v2',
                     'bge':'BAAI/bge-small-en-v1.5'}
                if name=='transformer':
                    # Raw BERT last hidden state, attention-mask mean pooling; not fine-tuned classifier.
                    import torch
                    from transformers import AutoTokenizer,AutoModel
                    tokenizer=AutoTokenizer.from_pretrained(ids[name]); encoder=AutoModel.from_pretrained(ids[name]); encoder.eval()
                    vectors=[]
                    for i in range(0,len(texts),args.batch_size):
                        inp=tokenizer([t[:args.max_chars] for t in texts[i:i+args.batch_size]],padding=True,truncation=True,max_length=512,return_tensors='pt')
                        with torch.no_grad(): h=encoder(**inp).last_hidden_state
                        mask=inp['attention_mask'].unsqueeze(-1)
                        vectors.append(((h*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)).numpy())
                    x=np.vstack(vectors); model=None
                else:
                    data=['passage: '+t for t in texts] if name=='e5' else texts
                    x,model=embed_hf(data,ids[name],args.batch_size,args.max_chars)
            shape=save_matrix(out,name,x,model)
            print('OK',shape,flush=True); report.append({'method':name,'status':'ok','shape':list(shape)})
        except Exception as exc:
            traceback.print_exc()
            report.append({
                'method': name,
                'status': 'failed',
                'error': f'{type(exc).__name__}: {exc}'})
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nReport:',out/'report.json')

if __name__=='__main__': main()

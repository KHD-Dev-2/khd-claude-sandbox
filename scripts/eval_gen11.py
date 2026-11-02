import json,re,math,unicodedata,os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
subprocess_import_error=None

# --- model ---
class Block(nn.Module):
    def __init__(self,d,heads,ff):
        super().__init__();self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d,bias=False);self.proj=nn.Linear(d,d,bias=False);self.ln2=nn.LayerNorm(d);self.fc1=nn.Linear(d,ff);self.fc2=nn.Linear(ff,d);self.heads=heads;self.drop=nn.Dropout(0.1);self.d=d
    def forward(self,x):
        b,t,d=x.shape;h=self.ln1(x);q,k,v=self.qkv(h).chunk(3,dim=-1);q=q.view(b,t,self.heads,d//self.heads).transpose(1,2);k=k.view(b,t,self.heads,d//self.heads).transpose(1,2);v=v.view(b,t,self.heads,d//self.heads).transpose(1,2);att=(q@k.transpose(-2,-1))/math.sqrt(d//self.heads);mask=torch.tril(torch.ones(t,t,device=x.device)).view(1,1,t,t);att=F.softmax(att.masked_fill(mask==0,float('-inf')),dim=-1);x=x+self.drop(self.proj((att@v).transpose(1,2).contiguous().view(b,t,d)));return x+self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)))))
class TinyGPT(nn.Module):
    def __init__(self,c):
        super().__init__();self.wte=nn.Embedding(c['vocab_size'],c['d_model']);self.wpe=nn.Embedding(c['context'],c['d_model']);self.loops=c.get('loops',1);self.blocks=nn.ModuleList([Block(c['d_model'],c['heads'],c['d_ff']) for _ in range(c['layers'])]) if self.loops==1 else nn.ModuleList();self.block=Block(c['d_model'],c['heads'],c['d_ff']) if self.loops>1 else None;self.loop_emb=nn.Embedding(self.loops,c['d_model']) if self.loops>1 else None;self.lnf=nn.LayerNorm(c['d_model']);self.head=nn.Linear(c['d_model'],c['vocab_size'],bias=False);self.context=c['context']
    def forward(self,idx):
        x=self.wte(idx)+self.wpe(torch.arange(idx.shape[1],device=idx.device))
        if self.loops>1:
            for loop in range(self.loops):x=self.block(x+self.loop_emb(torch.full((idx.shape[0],idx.shape[1]),loop,dtype=torch.long,device=idx.device)))
        else:
            for b in self.blocks:x=b(x)
        return self.head(self.lnf(x))

def digitize(t):return re.sub(r'[0-9]+(?:[.,][0-9]+)?',lambda m:' '.join(m.group(0).replace(',',' , ').replace('.',' . ')),t)
def norm(t):
    t=unicodedata.normalize('NFD',str(t));t=''.join(c for c in t if unicodedata.category(c)!='Mn');return re.sub(r'\s+',' ',t.lower()).strip()
def numbers(t):
    t=t.replace(',','.')
    return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',t)]
def score(pred,item):
    if item['kind']=='text':
        return norm(item['expected']) in norm(pred)
    nums=numbers(pred)
    if not nums:return False
    target=float(item['expected']);tol=max(0.02,abs(target)*item.get('tol',0.02))
    return any(abs(n-target)<=tol for n in nums[-4:])

DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ROOT=Path('/kaggle/input');files=list(ROOT.rglob('checkpoint.pt'));print('checkpoints',[str(x) for x in files],flush=True)
BASE=next(p for p in ROOT.iterdir() if p.is_dir())
GEN11=next(p.parent for p in ROOT.rglob('checkpoint.pt') if 'gen11' in str(p).lower())
models=[('gen11',GEN11,'khd-gen11')]
bench=[json.loads(x) for x in next(ROOT.rglob('bench.jsonl')).read_text(encoding='utf-8').splitlines() if x.strip()]
result={'status':'COMPLETE','bench_cases':len(bench),'models':{},'predictions':[]}
for name,base,label in models:
    ckpt=torch.load(base/'checkpoint.pt',map_location='cpu');config=ckpt.get('config') or json.loads((base/'config.json').read_text());model=TinyGPT(config).to(DEV);model.load_state_dict(ckpt['model']);model.eval()
    if (base/'tokenizer.json').exists():tok=__import__('tokenizers').Tokenizer.from_file(str(base/'tokenizer.json'))
    else:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        v=json.loads((base/'vocab.json').read_text());vocab={t:i for i,t in enumerate(v)};tok=Tokenizer(WordLevel(vocab=vocab,unk_token='<unk>'));tok.pre_tokenizer=Whitespace()
    bos=tok.token_to_id('<bos>') or 0;eos=tok.token_to_id('<eos>');ctx=config['context'];correct=0
    for i,it in enumerate(bench):
        q=it['question'];prompt=('Question : '+q+'\nRéponse :')
        if name in ('gen6','gen7','gen11'):prompt=digitize(prompt)
        ids=[bos]+tok.encode(prompt).ids
        with torch.no_grad():
            for _ in range(48):
                x=torch.tensor([ids[-ctx:]],device=DEV);logits=model(x)[0,-1];nid=int(torch.argmax(logits));
                if eos is not None and nid==eos:break
                ids.append(nid)
        pred=tok.decode(ids[len([bos]+tok.encode(prompt).ids):],skip_special_tokens=True);ok=score(pred,it);correct+=int(ok)
        result['predictions'].append({'model':name,'id':it['id'],'domain':it['domain'],'question':q,'expected':it['expected'],'prediction':pred[:300],'ok':ok})
        if (i+1)%50==0:print(name,i+1,'correct',correct,flush=True)
    result['models'][name]={'correct':correct,'total':len(bench),'accuracy':round(correct/len(bench),4),'params':sum(p.numel() for p in model.parameters())}
    print('DONE_MODEL',name,result['models'][name],flush=True)
Path('/kaggle/working/khd-eval-gen11-result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(result['models'],indent=2))

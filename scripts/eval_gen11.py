import json,re,math,unicodedata,os,sys,hashlib
from pathlib import Path

# Extractor v1 is FROZEN: digitize/norm/numbers/score are unchanged and stay the primary score until the Lead adopts v2.
# Extractor v2 only adds undigitize(): digit-tokenized models decode 12.5 as "1 2 . 5", which v1 reads as 1, 2, 5.
# Pure functions come before any torch import so --selftest and --rescore run on CPU without torch.
EXTRACTOR='v1-frozen+v2-undigitize'
DIGIT_MODELS=('gen6','gen7','gen11')

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

# --- extractor v2 (diagnostic until adopted) ---
_SPLIT=re.compile(r'(?<![\d.,])\d(?: +\d)*(?: +[.,] +\d(?: +\d)*)?(?![\d])')
def undigitize(t):
    """Inverse of digitize() on decoded text: '1 2 . 5' -> '12.5', '0 , 0 2 5' -> '0,025'. Single digits are unchanged."""
    return _SPLIT.sub(lambda m:re.sub(r' +','',m.group(0)),t)
def tokens(t):return re.findall(r'\w+|[^\w\s]',norm(t))
def numbers_v2(t):
    # Same as numbers() but a digit glued to a letter is not a number: 'V2 = 2,5' must not yield 2.
    return [float(x) for x in re.findall(r'(?<![A-Za-z_])-?\d+(?:\.\d+)?',t.replace(',','.'))]
def as_number(s):
    s=str(s).strip().rstrip('.').strip()
    return float(s.replace(',','.')) if re.fullmatch(r'-?\d+(?:[.,]\d+)?',s) else None
def _v2_item(item):
    # A text answer that is only a number (D2-style: '0,4817.', '157.') is scored as a number at its STATED precision
    # (half a unit of its last digit), so it stays exact like v1 intended; v1 needed the exact string spacing.
    n=as_number(item['expected']) if item['kind']=='text' else None
    if n is None:return item
    s=str(item['expected']).strip().rstrip('.').replace(',','.');dec=len(s.split('.')[1]) if '.' in s else 0
    return {**item,'kind':'numeric','expected':n,'abs_tol':0.5*10**-dec}
def _v2_num(nums,it):
    target=float(it['expected']);tol=it['abs_tol'] if 'abs_tol' in it else max(0.02,abs(target)*it.get('tol',0.02))
    return [abs(n-target)<=tol+1e-12 for n in nums]
def _v2_text(p,item):
    # Whole-token match instead of substring: v1 counts 'A' as found in any prediction containing the letter a ('Médiane').
    e=tokens(str(item['expected']).rstrip('.'));pt=tokens(p)
    return bool(e) and any(pt[i:i+len(e)]==e for i in range(len(pt)-len(e)+1))
def score_v2(pred,item,digit_split):
    """v1 rule (any of the last 4 numbers within tolerance) on correctly re-joined numbers."""
    p=undigitize(pred) if digit_split else pred;it=_v2_item(item)
    return _v2_text(p,it) if it['kind']=='text' else any(_v2_num(numbers_v2(p)[-4:],it))
def score_v2_strict(pred,item,digit_split):
    """Same as v2 but only the LAST number counts (v1/v2 accept any of the last 4, which lets operands match by chance)."""
    p=undigitize(pred) if digit_split else pred;it=_v2_item(item)
    return _v2_text(p,it) if it['kind']=='text' else any(_v2_num(numbers_v2(p)[-1:],it))
def vocab_digit_split(tokens):
    """True when the vocabulary holds single digits but no multi-digit or decimal number token (digit-by-digit corpus)."""
    tokens=set(tokens)
    return all(str(d) in tokens for d in range(10)) and not any(re.fullmatch(r'\d{2,}|\d+[.,]\d+',t) for t in tokens)

# --- statistics ---
def wilson(k,n,z=1.96):
    if n==0:return (0.0,0.0)
    p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (round(max(0.0,c-h),4),round(min(1.0,c+h),4))
def mcnemar_exact(b,c):
    """Two-sided exact McNemar (binomial on discordant pairs, p=0.5)."""
    n=b+c
    if n==0:return 1.0
    return min(1.0,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n)

def sha256_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rescore(result_path,bench_path,digit_split=None):
    """Re-score saved predictions with v1 and v2, without any model or GPU. digit_split=None uses the DIGIT_MODELS name rule."""
    res=json.loads(Path(result_path).read_text(encoding='utf-8'))
    bench={x['id']:x for x in (json.loads(l) for l in Path(bench_path).read_text(encoding='utf-8').splitlines() if l.strip())}
    out={'result':str(result_path),'bench_sha256':sha256_file(bench_path),'bench_cases':len(bench),'extractor':EXTRACTOR,'models':{},'per_case':{}}
    for p in res['predictions']:
        it=bench.get(p['id'])
        if it is None:continue
        ds=(p['model'] in DIGIT_MODELS) if digit_split is None else digit_split
        v1=score(p['prediction'],it);v2=score_v2(p['prediction'],it,ds);st=score_v2_strict(p['prediction'],it,ds)
        m=out['models'].setdefault(p['model'],{'n':0,'v1':0,'v2':0,'v2_strict':0,'recorded_ok':0,'v1_mismatch_recorded':0,'v1_to_v2_gained':0,'v1_to_v2_lost':0,'digit_split':ds})
        m['n']+=1;m['v1']+=v1;m['v2']+=v2;m['v2_strict']+=st;m['recorded_ok']+=bool(p.get('ok'));m['v1_mismatch_recorded']+=(v1!=bool(p.get('ok')))
        m['v1_to_v2_gained']+=(v2 and not v1);m['v1_to_v2_lost']+=(v1 and not v2)
        out['per_case'].setdefault(p['model'],{})[p['id']]={'v1':v1,'v2':v2,'v2_strict':st}
    for m in out['models'].values():
        m['wilson_v1']=wilson(m['v1'],m['n']);m['wilson_v2']=wilson(m['v2'],m['n']);m['wilson_v2_strict']=wilson(m['v2_strict'],m['n'])
        m['mcnemar_v1_vs_v2_p']=mcnemar_exact(m['v1_to_v2_gained'],m['v1_to_v2_lost'])
    return out

def compare(cand,base,key='v2'):
    """Paired exact McNemar between two rescored models on the cases both answered. b = cand right & base wrong."""
    ids=sorted(set(cand)&set(base));b=sum(cand[i][key] and not base[i][key] for i in ids);c=sum(base[i][key] and not cand[i][key] for i in ids)
    return {'cases':len(ids),'metric':key,'cand':sum(cand[i][key] for i in ids),'base':sum(base[i][key] for i in ids),'b':b,'c':c,'p_exact':mcnemar_exact(b,c)}

def selftest():
    num=lambda e:{'kind':'number','expected':e}
    # The bug: what a digit-tokenized model decodes for 12.5 is not read as 12.5 by v1.
    decoded='Réponse : 1 2 . 5 %'
    assert numbers(decoded)==[1.0,2.0,5.0]
    assert not score(decoded,num(12.5)) and score_v2(decoded,num(12.5),True)
    assert undigitize(digitize('0,025 et 300 sur 12000'))=='0,025 et 300 sur 12000'
    assert undigitize(' '.join(digitize('0,025 et 300 sur 12000').split()))=='0,025 et 300 sur 12000'
    assert undigitize('x = 7')=='x = 7' and undigitize('a 1 0 0 b')=='a 100 b'
    # v2 changes nothing for models that were not digit-tokenized.
    assert score_v2('Réponse : 12.5',num(12.5),False) and score_v2('2,5 %',num(2.5),False)
    # Chance match that v1/v2 accept and v2_strict refuses: target 3, the digit 3 appears among the last 4 numbers.
    assert score('3 1 4',num(3)) and not score_v2_strict('3 1 4 0',num(3),False)
    # D2 answers are text '0,4817.': a digit-tokenized model can never match the string in v1, even when exactly right.
    d2={'kind':'text','expected':'0,4817.'}
    assert not score('RR = 0 , 4 8 1 7 .',d2) and score_v2('RR = 0 , 4 8 1 7 .',d2,True) and not score_v2('RR = 8 , 5 .',d2,True)
    # Substring chance match in v1: expected 'A' is "found" in 'Médiane'.
    letter={'kind':'text','expected':'A'}
    assert score('Médiane = 2 7 .',letter) and not score_v2('Médiane = 2 7 .',letter,True) and score_v2('Réponse : A',letter,True)
    assert score_v2('6 4 0 J .',{'kind':'text','expected':'640 J.'},True) and score_v2('O ( log n )',{'kind':'text','expected':'O(log n)'},False)
    # Stated precision stays exact: '157.' is not matched by 159 (the 2 % numeric tolerance would have accepted it).
    assert not score_v2('Réponse : 1 5 9 .',{'kind':'text','expected':'157.'},True) and score_v2('Réponse : 1 5 7 .',{'kind':'text','expected':'157.'},True)
    # A digit glued to a letter is a name, not a number.
    assert score('V2 = 2 , 5 L .',num(2)) and not score_v2('V2 = 2 , 5 L .',num(2),True)
    assert vocab_digit_split([str(d) for d in range(10)]+['Réponse',':']) and not vocab_digit_split([str(d) for d in range(10)]+['12'])
    assert mcnemar_exact(4,5)==1.0 and abs(mcnemar_exact(0,3)-0.25)<1e-12 and abs(mcnemar_exact(10,2)-0.03857421875)<1e-12
    assert wilson(8,199)==(0.0205,0.0773)
    print('SELFTEST_OK')

if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='--selftest':
    selftest();sys.exit(0)
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='--rescore':
    # python eval_gen11.py --rescore <result.json> <bench.jsonl> [<baseline_result.json>]
    cand=rescore(sys.argv[2],sys.argv[3]);report={'candidate':{k:v for k,v in cand.items() if k!='per_case'}}
    if len(sys.argv)>4:
        base=rescore(sys.argv[4],sys.argv[3]);report['baseline']={k:v for k,v in base.items() if k!='per_case'}
        bname=next(iter(base['per_case']));report['paired']={f'{m}_vs_{bname}':{k:compare(cand['per_case'][m],base['per_case'][bname],k) for k in ('v1','v2','v2_strict')} for m in cand['per_case']}
    print(json.dumps(report,indent=2,ensure_ascii=False));sys.exit(0)

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

DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ROOT=Path('/kaggle/input');files=list(ROOT.rglob('checkpoint.pt'));print('checkpoints',[str(x) for x in files],flush=True)
BASE=next(p for p in ROOT.iterdir() if p.is_dir())
GEN11=next(p.parent for p in ROOT.rglob('checkpoint.pt') if 'gen11' in str(p).lower())
models=[('gen11',GEN11,'khd-gen11')]
bench_path=next(ROOT.rglob('bench.jsonl'))
bench=[json.loads(x) for x in bench_path.read_text(encoding='utf-8').splitlines() if x.strip()]
result={'status':'COMPLETE','bench_cases':len(bench),'bench_sha256':sha256_file(bench_path),'extractor':EXTRACTOR,'models':{},'predictions':[]}
for name,base,label in models:
    ckpt=torch.load(base/'checkpoint.pt',map_location='cpu');config=ckpt.get('config') or json.loads((base/'config.json').read_text());model=TinyGPT(config).to(DEV);model.load_state_dict(ckpt['model']);model.eval()
    if (base/'tokenizer.json').exists():tok=__import__('tokenizers').Tokenizer.from_file(str(base/'tokenizer.json'))
    else:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        v=json.loads((base/'vocab.json').read_text());vocab={t:i for i,t in enumerate(v)};tok=Tokenizer(WordLevel(vocab=vocab,unk_token='<unk>'));tok.pre_tokenizer=Whitespace()
    bos=tok.token_to_id('<bos>') or 0;eos=tok.token_to_id('<eos>');ctx=config['context'];correct=0;correct_v2=0;correct_v2_strict=0
    prompt_digitized=name in DIGIT_MODELS;vocab_split=vocab_digit_split(tok.get_vocab().keys())
    # The prompt rule is unchanged (frozen); a mismatch with the vocabulary is only reported, never silently fixed.
    if prompt_digitized!=vocab_split:print('WARNING_DIGIT_MISMATCH',name,'prompt_digitized',prompt_digitized,'vocab_digit_split',vocab_split,flush=True)
    for i,it in enumerate(bench):
        q=it['question'];prompt=('Question : '+q+'\nRéponse :')
        if prompt_digitized:prompt=digitize(prompt)
        ids=[bos]+tok.encode(prompt).ids
        with torch.no_grad():
            for _ in range(48):
                x=torch.tensor([ids[-ctx:]],device=DEV);logits=model(x)[0,-1];nid=int(torch.argmax(logits));
                if eos is not None and nid==eos:break
                ids.append(nid)
        pred=tok.decode(ids[len([bos]+tok.encode(prompt).ids):],skip_special_tokens=True);ok=score(pred,it);correct+=int(ok)
        ok_v2=score_v2(pred,it,vocab_split);ok_st=score_v2_strict(pred,it,vocab_split);correct_v2+=int(ok_v2);correct_v2_strict+=int(ok_st)
        result['predictions'].append({'model':name,'id':it['id'],'domain':it['domain'],'question':q,'expected':it['expected'],'prediction':pred[:300],'ok':ok,'ok_v2':ok_v2,'ok_v2_strict':ok_st})
        if (i+1)%50==0:print(name,i+1,'correct',correct,'v2',correct_v2,flush=True)
    result['models'][name]={'correct':correct,'total':len(bench),'accuracy':round(correct/len(bench),4),'params':sum(p.numel() for p in model.parameters()),
        'correct_v2':correct_v2,'correct_v2_strict':correct_v2_strict,'wilson_v1':wilson(correct,len(bench)),'wilson_v2':wilson(correct_v2,len(bench)),
        'prompt_digitized':prompt_digitized,'vocab_digit_split':vocab_split}
    print('DONE_MODEL',name,result['models'][name],flush=True)
Path('/kaggle/working/khd-eval-gen11-result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(result['models'],indent=2))

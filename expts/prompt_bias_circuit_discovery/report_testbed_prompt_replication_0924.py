"""Report all fixed prompt/temperature conditions; no selection or threshold tuning."""
import json,math,re,statistics
from pathlib import Path
from collections import Counter
from expts.prompt_bias_circuit_discovery.explicit_final_letter import explicit_final_letter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('results/prompt_bias_v2/testbed_audit_0924/prompt_replication')

def wilson(k,n):
    p=k/n;z=statistics.NormalDist().inv_cdf(.975);den=1+z*z/n
    c=(p+z*z/(2*n))/den;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-r,c+r

def main():
    records=json.loads((ROOT/'generations.json').read_text());m=json.loads((ROOT/'manifest.json').read_text())
    parsing=[]
    for record in records:
        for sample in record['rollouts']:
            declared,mode=explicit_final_letter(sample['text'])
            assert declared is not None, sample['seed']
            parsing.append(dict(seed=sample['seed'],llm_parser_answer=sample['answer'],declared_answer=declared,method=mode,disagreement=sample['answer']!=declared))
            sample['llm_parser_answer']=sample['answer'];sample['answer']=declared
    (ROOT/'answer_parsing_audit.json').write_text(json.dumps(parsing,indent=2))
    keys={(r['qid'],r['variant'],r['prompt_format'],r['temperature'],r['arm']):r for r in records}
    rows=[]
    for case in m['cases']:
        for fmt in ['published','local']:
            for temp in [.6,.7]:
                ctrl=keys[case['qid'],case['variant'],fmt,temp,'control'];intv=keys[case['qid'],case['variant'],fmt,temp,case['variant']]
                target=intv['target'];k0=sum(x['answer']==target for x in ctrl['rollouts']);k1=sum(x['answer']==target for x in intv['rollouts']);n0=len(ctrl['rollouts']);n1=len(intv['rollouts']);p0=k0/n0;p1=k1/n1;d=p1-p0
                l0,u0=wilson(k0,n0);l1,u1=wilson(k1,n1)
                rows.append(dict(qid=case['qid'],variant=case['variant'],prompt_format=fmt,temperature=temp,k0=k0,n0=n0,k1=k1,n1=n1,p0=p0,p1=p1,effect=d,published_effect=case['published_switch'],effect_ci95=[d-math.hypot(p1-l1,u0-p0),d+math.hypot(u1-p1,p0-l0)]))
    all_rolls=[r for x in records for r in x['rollouts']]
    direct=[]
    for r in all_rolls:
        final=r['text'].rsplit('</think>',1)[-1].strip()
        match=re.fullmatch(r'\(?([AB])\)?[.!]?',final)
        if match:direct.append(match.group(1)==r['answer'])
    assert len({r['seed'] for r in all_rolls})==len(all_rolls)
    out=dict(rows=rows,n_generations=len(all_rolls),answer_source='Deterministic final declared letter; original LLM-parser fields preserved in generations.json',llm_parser_disagreements=sum(x['disagreement'] for x in parsing),declared_answer_coverage=len(parsing),finish_reasons=dict(Counter(r['finish_reason'] for r in all_rolls)),missing_think_end=sum('</think>' not in r['text'] for r in all_rolls),unparsed=sum(r['answer'] not in ['A','B'] for r in all_rolls),single_letter_regex_checked=len(direct),single_letter_regex_disagrees=sum(not x for x in direct),distinct_sampling_seeds=len({r['seed'] for r in all_rolls}),mean_effect_by_format_temperature={f'{fmt}_{temp}':statistics.mean(x['effect'] for x in rows if x['prompt_format']==fmt and x['temperature']==temp) for fmt in ['published','local'] for temp in [.6,.7]},interpretation='Descriptive fixed three-case diagnostic, not a population yield estimate. All conditions reported; no mask training or tuning.')
    (ROOT/'summary.json').write_text(json.dumps(out,indent=2))
    plt.rcParams.update({'font.size':15,'axes.labelsize':15,'xtick.labelsize':14,'ytick.labelsize':14})
    fig,axes=plt.subplots(1,3,figsize=(17,5.5),sharey=True)
    for ax,case in zip(axes,m['cases']):
        for fmt,off,color,label in [('published',-.05,'#0072B2','Published prompt'),('local',.05,'#D55E00','Earlier local prompt')]:
            rs=[r for r in rows if r['qid']==case['qid'] and r['prompt_format']==fmt]
            y=np.array([r['effect'] for r in rs]);bounds=np.array([r['effect_ci95'] for r in rs]);ax.errorbar(np.array([.6,.7])+off/4,y,yerr=[y-bounds[:,0],bounds[:,1]-y],marker='o',capsize=4,color=color,label=label)
        ax.axhline(case['published_switch'],ls='--',color='gray',label='Released effect');ax.axhline(0,color='black',lw=.8);ax.set_title(case['qid'][:8]+'…');ax.set_xticks([.6,.7]);ax.set_xlim(.56,.74);ax.set_xlabel('Temperature');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Hinted minus control agreement');axes[0].legend(fontsize=12,loc='upper left');fig.tight_layout();dest=Path('notes/images/testbed_audit_0924/prompt_replication.png');fig.savefig(dest,dpi=160);plt.close(fig)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()

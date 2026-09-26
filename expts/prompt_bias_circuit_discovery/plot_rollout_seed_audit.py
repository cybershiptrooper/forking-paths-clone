"""Plot the historical admission seed-overlap audit.

  .venv/bin/python -m expts.prompt_bias_circuit_discovery.plot_rollout_seed_audit
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    folder=Path("results/prompt_bias_v2/rethink_0924/seed_overlap")
    summary=json.loads((folder/"summary.json").read_text())
    rows=json.loads((folder/"admission_338_reanalysis.json").read_text())
    plt.rcParams.update({"font.size":15,"axes.titlesize":17,"axes.labelsize":15,
                         "xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14})
    fig,axes=plt.subplots(2,2,figsize=(16,12),constrained_layout=True)
    blue,orange,gray="#0072B2","#D55E00","#777777"
    ax=axes[0,0]
    ax.broken_barh([(41.5,16)],(2-.25,.5),facecolors=orange)
    ax.broken_barh([(6.5,64)],(1-.25,.5),facecolors=gray)
    ax.broken_barh([(41.5,16)],(1-.25,.5),facecolors=orange)
    ax.broken_barh([(6.5,35),(57.5,13)],(-.25,.5),facecolors=blue)
    ax.set(yticks=[2,1,0],yticklabels=["Screen: 16 draws","Confirmation: 64 draws","Retained: 48 draws"],
           xlabel="Child RNG seed",xlim=(3,74),ylim=(-.6,2.8),xticks=[7,20,35,42,57,70],
           title="Screen seeds repeat in confirmation")
    ax.text(49.5,2.38,"42–57",ha="center",fontsize=15)
    ax.legend(handles=[Patch(color=orange,label="Reused seeds"),Patch(color=blue,label="Retained seeds")],loc="upper left",bbox_to_anchor=(0,-.17),ncol=2)
    ax=axes[0,1]
    tests=summary["summaries"]["retained"]["tests"]["mcnemar"]
    pos=set(tests["positive_bh"]["uids"])
    neg=set(tests["small_effect_new_p1_range"]["uids"])
    for uids,color,marker,label,size in [(neg,gray,"o",f"Low-effect candidates (n={len(neg)})",35),(pos,blue,"^",f"Effect-selected inputs (n={len(pos)})",65)]:
        selected=[r["retained"] for r in rows if r["uid"] in uids]
        ax.scatter([r["p0"] for r in selected],[r["p1"] for r in selected],s=size,color=color,marker=marker,alpha=.8,label=label)
    ax.plot([0,1],[0,1],"--",color="black",alpha=.4)
    ax.set(xlim=(-.025,1.025),ylim=(-.025,1.025),xlabel="Control admit rate p₀",ylabel="Intervention admit rate p₁",
           title="Retained bank supports 37 matched pairs")
    ax.text(.03,.97,"Exact gender; intervention-rate gap ≤0.10",transform=ax.transAxes,va="top",fontsize=14, bbox={"facecolor":"white","alpha":.95,"edgecolor":"none","pad":3})
    ax.legend(loc="lower right")
    ax=axes[1,0]
    selected=[r for r in rows if r["historical_selection"]=="positives"]
    ax.scatter([r["historical_64"]["delta"] for r in selected],[r["retained"]["delta"] for r in selected],color=blue,s=45,alpha=.75)
    ax.plot([.05,.65],[.05,.65],"--",color="black",alpha=.4)
    ax.axhline(.1,color=gray,linewidth=1)
    ax.set(xlabel="Effect estimate in historical 64 draws",ylabel="Effect estimate in retained 48 draws",xlim=(.05,.65),ylim=(.05,.65),
           title="Original 63 effects remain positive")
    ax.text(.08,.61,"All 63 retain an estimated effect ≥0.10\nMean effect: 0.296 → 0.285",va="top",fontsize=15)
    ax=axes[1,1]
    testall=summary["summaries"]["retained"]["tests"]
    labels=["Fisher\nBH","Paired\nBH","Paired\nBY","Paired\nHolm"]
    values=[testall["fisher"]["positive_bh"]["n"],tests["positive_bh"]["n"],tests["positive_by"]["n"],tests["positive_holm"]["n"]]
    bars=ax.bar(labels,values,color=[gray,blue,blue,blue])
    ax.bar_label(bars,padding=4,fontsize=17)
    ax.set(ylabel="Inputs passing positive-effect criterion",ylim=(0,49),title="Significance depends on test and correction")
    ax.text(.5,.96,"338 tests; effect ≥0.10; adjusted p <0.05",transform=ax.transAxes,ha="center",va="top",fontsize=14)
    for ax in axes.flat:
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(alpha=.12,axis="y")
    out=Path("notes/images/monitorability_rethink/seed_overlap_reanalysis.png")
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160)
    print(out)


if __name__=="__main__":
    main()

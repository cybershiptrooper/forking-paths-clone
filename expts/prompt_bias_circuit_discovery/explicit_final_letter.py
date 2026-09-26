"""Read explicitly declared A/B answers after </think>, without judging reasoning.

Returns None when the output does not supply an unambiguous supported format.
The final marked declaration wins if the model visibly corrects itself.
"""
import re


def explicit_final_letter(text):
    final=text.rsplit('</think>',1)[-1].strip().replace('**','').replace('`','')
    simple=re.fullmatch(r'\s*\(?([AB])\)?[.!]?\s*',final)
    if simple:return simple.group(1),'single_letter'
    patterns=[
        r'\\boxed\s*\{\s*(?:\\(?:text|mathrm)\s*\{\s*)?([AB])\s*\}',
        r'(?i:\b(?:final\s+)?answer(?:\s+is)?)\s*[:：]?\s*\(?([AB])\b',
        r'(?m)^\s*(?:#+\s*)?([AB])\s*\)',
        r'(?m)^\s*([AB])\s*$',
    ]
    matches=[(m.start(),m.group(1)) for pattern in patterns for m in re.finditer(pattern,final)]
    if matches:return max(matches,key=lambda x:x[0])[1],'explicit_marker'
    return None,'unresolved'

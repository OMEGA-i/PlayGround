"""Summarize independently recorded gates; never promote downstream production evidence."""
import argparse
import json
from pathlib import Path


def matrix(root, robots):
    rows=[]
    for robot in robots:
        row={'robot':robot}
        for layer in range(1,6):
            p=root/f'layer{layer}'/robot/'result.json'
            row[f'layer{layer}']=json.loads(p.read_text())['status'] if p.exists() else 'not_run'
        for layer,required in ((3,(1,2)),(4,(1,2,3)),(5,(1,2,3,4))):
            failed=[i for i in required if row[f'layer{i}']!='passed']
            if failed:
                row[f'layer{layer}']='blocked_by_'+','.join(map(str,failed))
        rows.append(row)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    args=p.parse_args();rows=matrix(args.root,['g1','a3','h2','sih2300'])
    (args.root/'matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))

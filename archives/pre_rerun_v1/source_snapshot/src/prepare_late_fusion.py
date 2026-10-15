"""Prepare train-only native CQT/rhythm normalization for representation-level fusion."""
import argparse
import json
from pathlib import Path

import numpy as np

from data.aligned_sequences import digest,load_split
from data.native_sequences import NativePairSource,compute_native_statistics
from features_cqt import ROOT,sha256


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    split_doc,metadata,ids,labels,_=load_split(args.metadata,args.split)
    train_ids=split_doc['parts']['train']
    source=NativePairSource(ids,labels,args.cqt_dir,args.rhythm_dir,args.metadata,args.split,args.manifest)
    cm,cs,cc,rm,rs,rc,cframes,rwindows=compute_native_statistics(source,train_ids)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:
        np.savez(f,cqt_mean=cm,cqt_std=cs,cqt_counts=cc,
                 rhythm_mean=rm,rhythm_std=rs,rhythm_counts=rc,
                 cqt_frames=np.asarray(cframes),rhythm_windows=np.asarray(rwindows),
                 mode=np.asarray('native_two_branch_v1'),
                 source_signature=np.asarray(source.source_signature),
                 manifest_sha256=np.asarray(source.manifest_sha256),
                 train_ids_sha256=np.asarray(digest('\n'.join(train_ids).encode())),
                 split_sha256=np.asarray(sha256(args.split)),
                 metadata_sha256=np.asarray(sha256(args.metadata)))
    print(json.dumps({'output':str(args.output),'train_pieces':len(train_ids),
                      'cqt_frames':cframes,'rhythm_windows':rwindows},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cqt-dir',type=Path,default=ROOT/'datasets/features/cqt')
    p.add_argument('--rhythm-dir',type=Path,default=ROOT/'datasets/features/rhythm_windows')
    p.add_argument('--metadata',type=Path,default=ROOT/'datasets/metadata/new_clean_data.json')
    p.add_argument('--split',type=Path,default=ROOT/'datasets/splits/cqt_rhythm_common_v1.json')
    p.add_argument('--manifest',type=Path,default=None)
    p.add_argument('--output',type=Path,default=ROOT/'datasets/features/cqt_rhythm_native_normalization.npz')
    main(p.parse_args())

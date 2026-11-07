"""Current SMILES -> SI workflow; competition endpoint columns are forbidden."""
import argparse
import json
from threadpoolctl import threadpool_limits
from src.current.data import ROOT,prepare,sha


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/current.json')
    parser.add_argument('--stage',choices=['all','prepare','train','features'],default='all')
    parser.add_argument('--worker-repeat',type=int,choices=[0,1,2])
    parser.add_argument('--final-only',action='store_true')
    parser.add_argument('--predict-artifact')
    parser.add_argument('--test',default='data/raw/kaggle/test.csv')
    parser.add_argument('--out',default='reports/current/prediction.csv')
    args=parser.parse_args(argv)
    config=json.loads((ROOT/args.config).read_text())
    with threadpool_limits(limits=4):
        if args.predict_artifact:
            from src.current.workflow import predict
            predict(args.predict_artifact,args.test,args.out)
            return
        train_path=ROOT/'data/raw/kaggle/train_v2.csv'
        expected=config.get('expected_train_sha256')
        if expected and train_path.exists() and sha(train_path)!=expected:
            raise ValueError('Unreviewed competition train version; compare official files and update the dataset/configuration explicitly')
        if args.stage in ('all','prepare'):
            prepare(config)
        if args.stage=='features':
            from src.current.workflow import feature_inputs
            _,_,_,_,metadata,_=feature_inputs(config)
            print(json.dumps(metadata,indent=2))
        if args.stage in ('all','train'):
            from src.current.workflow import run
            run(config,args.worker_repeat,args.final_only)


if __name__=='__main__':
    main()

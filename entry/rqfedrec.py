import argparse
import configparser

from flalgorithm.rqfedrec import RQFedRec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ml100k/rqfedrec.ini')
    args= parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    Testfed = RQFedRec(config)
    Testfed.fit()

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', help='define the model for training (factorization / sequence)')
    args = parser.parse_args()
    model = args.model
    os.system('export OPENBLAS_NUM_THREADS=1')

    if model == 'factorization':
        from mrecsys.factorization import trainer
        trainer.run()
    elif model == 'sequence':
        from mrecsys.sequence import trainer
        trainer.run()
    else:
        raise ValueError("Unknown Model Type")

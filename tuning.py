import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='define the model for tuning (factorization / sequence)')
    args = parser.parse_args()
    model = args.model

    if model == 'factorization':
        from mrecsys.factorization import tuner
        tuner.run()
    elif model == 'sequence':
        from mrecsys.sequence import tuner
        tuner.run()
    else:
        raise ValueError("Unknown Model Type")

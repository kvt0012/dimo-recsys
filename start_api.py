import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', help='define the model for starting api (factorization / sequence)')
    args = parser.parse_args()
    model = args.model

    if model == 'factorization':
        from mrecsys.factorization.service import api
        api.run()
    elif model == 'sequence':
        from mrecsys.sequence.service import api
        api.run()
    else:
        raise ValueError("Unknown Model Type")

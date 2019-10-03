import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', help='define the model for starting api (factorization / sequence)')
    args = parser.parse_args()
    target = args.target

    if target == 'factorization':
        from mrecsys.factorization.service import api
        api.run()
    elif target == 'sequence':
        from mrecsys.sequence.service import api
        api.run()
    else:
        raise ValueError("Unknown Model Type")

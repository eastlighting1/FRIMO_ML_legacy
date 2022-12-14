from summa import keywords
import argparse

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--summ_data',
                    type=str,
                    default=False,
                    help='summarized data')

args = parser.parse_args()

print(keywords.keywords(args.summ_data))
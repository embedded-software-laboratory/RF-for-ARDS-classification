import argparse

from learning.sk_Learner import SKLearner
import json


def read_options():
    with open("options.json", "r") as file:
        return json.load(file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("database")
    argparser.add_argument("filter")
    args = argparser.parse_args()
    database = args.database
    filter = args.filter
    options = read_options()
    learner = SKLearner(options)

    learner.tune_hyperparameters(database, filter)

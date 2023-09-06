from learning.sk_Learner import SKLearner
import json

def read_options():
    with open("options.json", "r") as file:
        return json.load(file)
if __name__ == '__main__' :
    options = read_options()
    learner = SKLearner(options)
    learner.tune_hyperparameters()
from middlewares.to_pull import pull
from middlewares.pull_to_preprocess import pull_preprocess
from middlewares.preprocess_to_train import preprocess_train
from middlewares.train_to_test import train_test

stage_one = pull()
stage_two = pull_preprocess(stage_one)
stage_three = preprocess_train(stage_two)
stage_four = train_test(stage_three)

print("Pipeline done")
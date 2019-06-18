# Billing items in shopping using OpenCV and Tensorflow


To log (optional):
tensorboard --logdir tf_files/training_summaries &

Below command trains a model by transfer learning:
python retrain.py --bottleneck_dir=bottlenecks --how_many_training_steps=500 --model_dir=models --summaries_dir=training_summaries --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=billing_items

Run source.py
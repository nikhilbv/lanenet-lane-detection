import glog as log

from data_provider import lanenet_data_feed_pipline
from config import global_config

CFG = global_config.cfg
log.info("CFG: {}".format(CFG))


def main():
  dataset_dir = "./data/training_data_example"
  train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=dataset_dir, flags='train'
    )

  train_images, train_binary_labels, train_instance_labels = train_dataset.inputs(
            CFG.TRAIN.BATCH_SIZE, 1
        )

  log.info("train_images, train_binary_labels, train_instance_labels: {},{},{}".format(train_images, train_binary_labels, train_instance_labels))


if __name__=='__main__':
  main()

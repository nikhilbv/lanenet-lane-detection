**1. Test - **
python tools/test_lanenet.py --weights_path ./model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt  --image_path ./data/tusimple_test_image/0.jpg
python tools/test_lanenet.py --weights_path ./model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt  --image_path /aimldl-cod/practice/nikhil/sample-images/*.jpg

**2. Data Preparation - **
python data_provider/lanenet_data_feed_pipline.py --dataset_dir ./data/training_data_example --save_dir ./data/training_data_example/tfrecords
python data_provider/lanenet_data_feed_pipline.py --dataset_dir /aimldl-dat/data-public/tusimple/train_set/training --tfrecords_dir /aimldl-dat/data-public/tusimple/train_set/training/tfrecords

**3. Generate tusimple dataset - **
python tools/generate_tusimple_dataset.py --src_dir path/to/your/unzipped/file
python tools/generate_tusimple_dataset.py --src_dir /aimldl-dat/data-public/tusimple/train_set

**4. Train - **
python tools/train_lanenet.py --net vgg --dataset_dir data/training_data_example -m 0
python tools/train_lanenet.py --net vgg --dataset_dir ./data/training_data_example -m 0 1>1.output.log 2>1.error.log
python tools/train_lanenet.py --net vgg --dataset_dir ./data/training_data_example -m 0 1>logs/lanenet-$(date -d now +'%d%m%y_%H%M%S').log 2>&1
python tools/train_lanenet.py --net vgg --dataset_dir /aimldl-dat/data-public/tusimple/train_set/training -m 0 1>logs/lanenet-$(date -d now +'%d%m%y_%H%M%S').log 2>&1

**5. Evaluate - **
python tools/evaluate_lanenet_on_tusimple.py --image_dir ROOT_DIR/TUSIMPLE_DATASET/test_set/clips --weights_path ./model/tusimple_lanenet_vgg/tusimple_lanenet.ckpt --save_dir ROOT_DIR/TUSIMPLE_DATASET/test_set/test_output


## Debug

find . -iname "*.py" -type f -exec grep -inH --color="auto" "device" {} \;
./tools/train_lanenet.py:228:    with tf.device('/gpu:1'):
./tools/train_lanenet.py:550:            with tf.device('/gpu:{:d}'.format(i)):
./tools/train_lanenet.py:614:    sess_config = tf.ConfigProto(device_count={'GPU': CFG.TRAIN.GPU_NUM}, allow_soft_placement=True)

```json
{
  "TRAIN": {
    "EPOCHS": 80010,
    "DISPLAY_STEP": 1,
    "VAL_DISPLAY_STEP": 1000,
    "MOMENTUM": 0.9,
    "LEARNING_RATE": 0.0005,
    "GPU_MEMORY_FRACTION": 0.95,
    "TF_ALLOW_GROWTH": true,
    "BATCH_SIZE": 4,
    "VAL_BATCH_SIZE": 4,
    "CLASSES_NUMS": 2,
    "IMG_HEIGHT": 256,
    "IMG_WIDTH": 512,
    "EMBEDDING_FEATS_DIMS": 4,
    "CROP_PAD_SIZE": 32,
    "CPU_MULTI_PROCESS_NUMS": 6,
    "MOVING_AVERAGE_DECAY": 0.9999,
    "GPU_NUM": 2
  },
  "TEST": {
    "GPU_MEMORY_FRACTION": 0.8,
    "TF_ALLOW_GROWTH": true,
    "BATCH_SIZE": 2
  },
  "POSTPROCESS": {
    "MIN_AREA_THRESHOLD": 100,
    "DBSCAN_EPS": 0.35,
    "DBSCAN_MIN_SAMPLES": 1000
  }
}
```

I0727
W0727
E0727

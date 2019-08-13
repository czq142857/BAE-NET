import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np

from model import IMSEG

import tensorflow as tf
import h5py

flags = tf.app.flags
flags.DEFINE_integer("epoch", 0, "Epoch to train [0]")
flags.DEFINE_integer("iteration", 0, "Iteration to train. Either epoch or iteration need to be zero [0]")
flags.DEFINE_integer("pretrain_iters", 2000, "Iteration for supervised training [1000]")
flags.DEFINE_integer("retrain_iters", 4, "Set to positive number N for doing one supervised PASS (training all shapes in supervision_list) every N iterations. Set to 0 for fully supervised training. Set to negative number -N for doing one supervised PASS every N epochs [4]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string("dataset", "03001627_vox", "The name of dataset")
flags.DEFINE_integer("real_size", 32, "output point-value voxel grid size in training [64]")
flags.DEFINE_integer("points_per_shape", 8192, "num of points per shape [32768]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("supervision_list", "obj_train_list.txt", "A list of objects for supervised training")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("recon", False, "(in testing mode) True for outputing reconstructed shape with colored segmentation [False]")
flags.DEFINE_boolean("pointcloud", False, "(in testing mode) True for outputing point cloud with colored segmentation [False]")
flags.DEFINE_boolean("mesh", False, "(in testing mode) True for outputing mesh with colored segmentation [False]")
flags.DEFINE_boolean("iou", False, "(in testing mode) True for outputing IOU for test shapes [False]")
flags.DEFINE_boolean("enhance_vertical", False, "True for applying data enhancement by moving model in vertical direction [False]")
flags.DEFINE_boolean("supervised", False, "True for supervised training, False for unsupervised [False]")
flags.DEFINE_boolean("L1reg", False, "True for adding L1 regularization at layer 3 [False]")
FLAGS = flags.FLAGS


ID2name = {
'02691156': 'airplane',
'02773838': 'bag',
'02954340': 'cap',
'02958343': 'car',
'03001627': 'chair',
'03261776': 'earphone',
'03467517': 'guitar',
'03624134': 'knife',
'03636649': 'lamp', # lamp - missing one part
'03642806': 'laptop',
'03790512': 'motorbike',
'03797390': 'mug',
'03948459': 'pistol',
'04099429': 'rocket',
'04225987': 'skateboard',
'04379243': 'table' # table - missing one part
}

ID2Partnum = {'02691156': 4,
'02773838': 2,
'02954340': 2,
'02958343': 4,
'03001627': 4,
'03261776': 3,
'03467517': 3,
'03624134': 2,
'03636649': 4,
'03642806': 2,
'03790512': 6,
'03797390': 2,
'03948459': 3,
'04099429': 3,
'04225987': 3,
'04379243': 3 }



def main(_):
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	#run_config = tf.ConfigProto(gpu_options=gpu_options)
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		imseg = IMSEG(
				sess,
				FLAGS.real_size,
				FLAGS.points_per_shape,
				FLAGS.supervised,
				FLAGS.L1reg,
				supervision_list = FLAGS.supervision_list,
				is_training = FLAGS.train,
				dataset_name=FLAGS.dataset,
				checkpoint_dir=FLAGS.checkpoint_dir,
				sample_dir=FLAGS.sample_dir,
				data_dir=FLAGS.data_dir)

		if FLAGS.train:
			imseg.train(FLAGS)
		else:
			if FLAGS.recon:
				imseg.test_dae(FLAGS) #output reconstructed shape with colored segmentation
			if FLAGS.pointcloud:
				imseg.test_pointcloud(FLAGS) #output point cloud with colored segmentation
			if FLAGS.mesh:
				imseg.test_obj(FLAGS) #output mesh with colored segmentation
			if FLAGS.iou:
				imseg.test_pcSeg(FLAGS) #output IOU for test shapes

if __name__ == '__main__':
	tf.app.run()

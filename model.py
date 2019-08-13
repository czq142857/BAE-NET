import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import h5py

from ops import *
from utils import *

class IMSEG(object):
	def __init__(self, sess, real_size, points_per_shape, supervised, L1reg, supervision_list, is_training = False, z_dim=128, ef_dim=32, gf_dim=256, dataset_name='default', checkpoint_dir=None, sample_dir=None, data_dir='./data'):
		"""
		Args:
			too lazy to explain
		"""
		self.sess = sess

		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16*2)
		#3-- (64, 32*32*32)
		self.real_size = real_size #output point-value voxel grid size in training
		self.points_per_shape = points_per_shape #training batch size (virtual, batch_size is the real batch_size)
		
		self.batch_size = self.points_per_shape
		
		self.input_size = 64 #input voxel grid size

		self.z_dim = z_dim
		self.ef_dim = ef_dim
		self.gf_dim = gf_dim
		self.L1reg = L1reg

		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.data_dir = data_dir
		
		data_hdf5_name = self.data_dir+'/'+self.dataset_name+'.hdf5'
		if os.path.exists(data_hdf5_name):
			self.data_dict = h5py.File(data_hdf5_name, 'r')
			data_points_int = self.data_dict['points_'+str(self.real_size)][:]
			self.data_points = (data_points_int+0.5)/self.real_size-0.5
			self.data_values = self.data_dict['values_'+str(self.real_size)][:]
			self.data_voxels = self.data_dict['voxels'][:]
			if self.points_per_shape!=self.data_points.shape[1]:
				print("error: points_per_shape!=data_points.shape")
				exit(0)
			if self.input_size!=self.data_voxels.shape[1]:
				print("error: input_size!=data_voxels.shape")
				exit(0)
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)
		
		if supervised:
			# load whole set
			allset_name = self.dataset_name[:8] + "_vox"
			allset_txt_name = self.data_dir+'/'+allset_name+'.txt'
			if allset_name==self.dataset_name:
				allset_points = self.data_points
				allset_values = self.data_values
				allset_voxels = self.data_voxels
			else:
				allset_hdf5_name = self.data_dir+'/'+allset_name+'.hdf5'
				if os.path.exists(allset_hdf5_name):
					allset_dict = h5py.File(allset_hdf5_name, 'r')
					allset_points_int = allset_dict['points_'+str(self.real_size)][:]
					allset_points = (allset_points_int+0.5)/self.real_size-0.5
					allset_values = allset_dict['values_'+str(self.real_size)][:]
					allset_voxels = allset_dict['voxels'][:]
					if self.points_per_shape!=allset_points.shape[1]:
						print("error: points_per_shape!=data_points.shape")
						exit(0)
					if self.input_size!=allset_voxels.shape[1]:
						print("error: input_size!=data_voxels.shape")
						exit(0)
				else:
					print("error: cannot load "+allset_hdf5_name)
					exit(0)
			
			# load training point cloud
			ref_txt_name = self.data_dir+'/'+supervision_list
			if os.path.exists(ref_txt_name):
				self.ref_b_points, self.ref_b_values, self.ref_b_point_num, self.gf_split, self.ref_idx, _, self.ref_obj_name = parse_txt_list(ref_txt_name, self.data_dir+"/points", allset_txt_name)
				self.ref_points = allset_points[self.ref_idx]
				self.ref_values = allset_values[self.ref_idx]
				self.ref_voxels = allset_voxels[self.ref_idx]
				'''
				#output obj
				fout = open("ref.obj", 'w')
				for i in range(self.ref_b_point_num[0]):
					fout.write("v "+str(self.ref_b_points[0,i,0])+" "+str(self.ref_b_points[0,i,1])+" "+str(self.ref_b_points[0,i,2])+"\n")
				fout.close()
				
				fout = open("vox.obj", 'w')
				for i in range(len(self.ref_points[0])):
					if self.ref_values[0,i,0]>0:
						fout.write("v "+str(self.ref_points[0,i,0])+" "+str(self.ref_points[0,i,1])+" "+str(self.ref_points[0,i,2])+"\n")
				fout.close()
				exit(0)
				'''
			else:
				print("error: cannot load "+ref_txt_name)
				exit(0)
			
			# load testing point cloud
			testset_name = self.dataset_name[:8] + "_test_vox"
			test_txt_name = self.data_dir+'/'+testset_name+'.txt'
			if os.path.exists(test_txt_name):
				self.test_b_points, self.test_b_values, self.test_b_point_num, _, self.test_idx, self.labels_unique, _ = parse_txt_list(test_txt_name, self.data_dir+"/points", allset_txt_name)
				self.test_points = allset_points[self.test_idx]
				self.test_values = allset_values[self.test_idx]
				self.test_voxels = allset_voxels[self.test_idx]
			else:
				print("error: cannot load "+test_txt_name)
				exit(0)
			
			#attention: for table category we only use 2 branches: top, leg
			#original: top, leg, other support
			if "04379243" in self.dataset_name:
				self.gf_split = 2
			#attention: for lamp category we only use 3 branches: base, pole, lampshade
			#original: base, pole, canopy, lampshade. therefore switch place 2<->3
			if "03636649" in self.dataset_name:
				self.gf_split = 3
				temp = np.copy(self.test_b_values)
				self.test_b_values[:,:,2] = temp[:,:,3]
				self.test_b_values[:,:,3] = temp[:,:,2]
				temp = np.copy(self.labels_unique)
				self.labels_unique[2] = temp[3]
				self.labels_unique[3] = temp[2]
			
		else:
			self.gf_split = 8
			self.ref_points = []
			allset_name = self.dataset_name[:8] + "_vox"
			allset_txt_name = self.data_dir+'/'+allset_name+'.txt'
			if allset_name==self.dataset_name:
				allset_points = self.data_points
				allset_values = self.data_values
				allset_voxels = self.data_voxels
			else:
				allset_hdf5_name = self.data_dir+'/'+allset_name+'.hdf5'
				if os.path.exists(allset_hdf5_name):
					allset_dict = h5py.File(allset_hdf5_name, 'r')
					allset_points_int = allset_dict['points_'+str(self.real_size)][:]
					allset_points = (allset_points_int+0.5)/self.real_size-0.5
					allset_values = allset_dict['values_'+str(self.real_size)][:]
					allset_voxels = allset_dict['voxels'][:]
					if self.points_per_shape!=allset_points.shape[1]:
						print("error: points_per_shape!=data_points.shape")
						exit(0)
					if self.input_size!=allset_voxels.shape[1]:
						print("error: input_size!=data_voxels.shape")
						exit(0)
				else:
					print("error: cannot load "+allset_hdf5_name)
					exit(0)
			ref_txt_name = self.data_dir+'/'+supervision_list
			if os.path.exists(ref_txt_name):
				self.ref_idx, self.ref_obj_name = parse_txt_list_unsupervised(ref_txt_name, allset_txt_name)
				self.ref_voxels = allset_voxels[self.ref_idx]
		
		
		
		if not is_training:
			self.real_size = 64 #output point-value voxel grid size in testing
			self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
			self.batch_size = self.test_size*self.test_size*self.test_size #do not change
			
			#get coords
			dima = self.test_size
			dim = self.real_size
			self.aux_x = np.zeros([dima,dima,dima],np.uint8)
			self.aux_y = np.zeros([dima,dima,dima],np.uint8)
			self.aux_z = np.zeros([dima,dima,dima],np.uint8)
			multiplier = int(dim/dima)
			multiplier2 = multiplier*multiplier
			multiplier3 = multiplier*multiplier*multiplier
			for i in range(dima):
				for j in range(dima):
					for k in range(dima):
						self.aux_x[i,j,k] = i*multiplier
						self.aux_y[i,j,k] = j*multiplier
						self.aux_z[i,j,k] = k*multiplier
			self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
						self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
			self.coords = (self.coords+0.5)/dim-0.5
			self.coords = np.reshape(self.coords,[multiplier3,self.batch_size,3])
		
		self.build_model()

	def build_model(self):
		self.vox3d = tf.placeholder(shape=[1,self.input_size,self.input_size,self.input_size,1], dtype=tf.float32, name="vox3d")
		self.z_vector = tf.placeholder(shape=[1,self.z_dim], dtype=tf.float32, name="z_vector")
		self.point_coord = tf.placeholder(shape=[None,3], dtype=tf.float32, name="point_coord")
		self.point_value = tf.placeholder(shape=[None,1], dtype=tf.float32, name="point_value")
		self.branch_coord = tf.placeholder(shape=[None,3], dtype=tf.float32, name="branch_coord")
		self.branch_value = tf.placeholder(shape=[None,self.gf_split], dtype=tf.float32, name="branch_value")
		
		self.E = self.encoder(self.vox3d, phase_train=True, reuse=False)
		self.G_, self.G = self.generator(self.point_coord, self.E, phase_train=True, reuse=False)
		self.Gsuper_, self.Gsuper = self.generator(self.branch_coord, self.E, phase_train=True, reuse=True)
		
		self.sE = self.encoder(self.vox3d, phase_train=False, reuse=True)
		self.sG_, self.sG = self.generator(self.point_coord, self.sE, phase_train=False, reuse=True)
		self.bG, self.zG = self.generator(self.point_coord, self.z_vector, phase_train=False, reuse=True)
		
		self.loss = tf.reduce_mean(tf.square(self.point_value - self.G))
		
		if self.L1reg:
			regularizer = tf.contrib.layers.l1_regularizer(scale=0.000001)
			reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			#print("\n\n\nreg_variables")
			#print(reg_variables)
			#print("\n\n\n")
			reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
			self.loss += reg_term
		
		self.loss_supervised = self.loss + tf.reduce_mean(tf.square(self.branch_value - self.Gsuper_))
		
		self.saver = tf.train.Saver(max_to_keep=10)
		
		
	def generator(self, points, z, phase_train=True, reuse=False):
		
		batch_size = tf.shape(points)[0]
		zs = tf.tile(z, [batch_size,1])
		pointz = tf.concat([points,zs],1)
		print("pointz",pointz.shape)
		
		
		with tf.variable_scope("simple_net") as scope:
			if reuse:
				scope.reuse_variables()
			
			#level 1
			h1 = lrelu(linear(pointz, self.gf_dim*4, 'h1'))
			
			#level 2
			h2 = lrelu(linear(h1, self.gf_dim, 'h2'))
			
			#level 2_2
			#uncomment the following line to get the 4-layer model
			#h2 = lrelu(linear(h2, self.gf_dim, 'h2_2'))
		
			#level 3
			h3 = tf.nn.sigmoid(linear(h2, self.gf_split, 'h3', add_reg=(self.L1reg and not reuse) ))
			
			#Sometimes it is beneficial to let the initial value of the output equal to 0 (rather than 0.5).
			#Uncomment the following line to move the initial output value from 0.5 to 0.
			#h3 = h3*2-1
		
		return h3, tf.reduce_max(h3, axis=1, keepdims=True)
	
	def encoder(self, inputs, phase_train=True, reuse=False):
		with tf.variable_scope("encoder") as scope:
			if reuse:
				scope.reuse_variables()
			
			d_1 = conv3d(inputs, shape=[4, 4, 4, 1, self.ef_dim], strides=[1,2,2,2,1], scope='conv_1')
			d_1 = lrelu(batch_norm(d_1, phase_train))

			d_2 = conv3d(d_1, shape=[4, 4, 4, self.ef_dim, self.ef_dim*2], strides=[1,2,2,2,1], scope='conv_2')
			d_2 = lrelu(batch_norm(d_2, phase_train))
			
			d_3 = conv3d(d_2, shape=[4, 4, 4, self.ef_dim*2, self.ef_dim*4], strides=[1,2,2,2,1], scope='conv_3')
			d_3 = lrelu(batch_norm(d_3, phase_train))

			d_4 = conv3d(d_3, shape=[4, 4, 4, self.ef_dim*4, self.ef_dim*8], strides=[1,2,2,2,1], scope='conv_4')
			d_4 = lrelu(batch_norm(d_4, phase_train))

			d_5 = conv3d(d_4, shape=[4, 4, 4, self.ef_dim*8, self.z_dim], strides=[1,1,1,1,1], scope='conv_5', padding="VALID")
			d_5 = tf.nn.sigmoid(d_5)
		
			return tf.reshape(d_5,[1,self.z_dim])
	
	def train(self, config):
		ae_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
		ae_optim_supervised = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss_supervised)
		self.sess.run(tf.global_variables_initializer())
		
		batch_idxs = len(self.data_points)
		batch_index_list = np.arange(batch_idxs)
		batch_ridxs = len(self.ref_points)
		batch_rindex_list = np.arange(batch_ridxs)
		
		print("\n\n----------net summary----------")
		print("training samples   ", batch_idxs)
		print("supervised samples ", batch_ridxs)
		print("network branch	 ", self.gf_split)
		print("-------------------------------\n\n")
		
		counter = 0
		start_time = time.time()
		
		if config.supervised:
			pretrain_iters = config.pretrain_iters
			retrain_iters = config.retrain_iters
			retrain_epochs = 1
			if retrain_iters<0:
				retrain_epochs = -retrain_iters
				retrain_iters = batch_idxs
		else:
			retrain_epochs = 100000000
			pretrain_iters = 0
			retrain_iters = 100000000
		
		if counter==0 and retrain_iters>0:
			# supervised training
			# use a few examples (1/2/3)
			for iter in range(pretrain_iters):
				np.random.shuffle(batch_rindex_list)
				for ridx in range(batch_ridxs):
					dxb = batch_rindex_list[ridx]
					_, errAE = self.sess.run([ae_optim_supervised, self.loss_supervised],
						feed_dict={
							self.vox3d: self.ref_voxels[dxb:dxb+1],
							self.branch_coord: self.ref_b_points[dxb,:self.ref_b_point_num[dxb]],
							self.branch_value: self.ref_b_values[dxb,:self.ref_b_point_num[dxb]],
							self.point_coord: self.ref_points[dxb],
							self.point_value: self.ref_values[dxb],
						})
					if (ridx%10==9):
						print("Iter: [%6d] time: %4.4f, loss: %.8f" % (iter, time.time() - start_time, errAE))
			#self.save(config.checkpoint_dir, 0)
		
		
		# ------data enhancement hyper-params------
		# apply data enhancement if config.enhance_vertical == True
		if config.enhance_vertical:
			assert self.real_size==32
			random_range = 8
			mul = int(self.input_size/self.real_size)
		
		
		# -------- training --------
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/batch_idxs)
		for epoch in range(0, training_epoch+1):
			
			# unsupervised training
			if retrain_iters>0:
				np.random.shuffle(batch_index_list)
				avg_loss = 0
				avg_num = 0
				for idx in range(batch_idxs):
					dxb = batch_index_list[idx]
					if config.enhance_vertical:
						# ------data enhancement Y axis------
						batch_voxel_origin = self.data_voxels[dxb:dxb+1]
						batch_point = self.data_points[dxb]
						batch_value = self.data_values[dxb]
						offset = np.random.randint(-random_range,random_range+1)
						batch_voxel = np.zeros([1,self.input_size,self.input_size,self.input_size,1],np.uint8)
						batch_voxel[:,:,max(0,0+offset*mul):min(self.input_size,self.input_size+offset*mul),:,:] = batch_voxel_origin[:,:,max(0,0+offset*mul)-offset*mul:min(self.input_size,self.input_size+offset*mul)-offset*mul,:,:]
						batch_point = batch_point+float(offset)/self.real_size
						# ------end of data enhancement------
						_, errAE = self.sess.run([ae_optim, self.loss],
							feed_dict={
								self.vox3d: batch_voxel,
								self.point_coord: batch_point,
								self.point_value: batch_value,
							})
					else:
						_, errAE = self.sess.run([ae_optim, self.loss],
							feed_dict={
								self.vox3d: self.data_voxels[dxb:dxb+1],
								self.point_coord: self.data_points[dxb],
								self.point_value: self.data_values[dxb],
							})
					avg_loss += errAE
					avg_num += 1
					
					if (idx==batch_idxs-1):
						print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f" % (epoch, training_epoch, idx, batch_idxs, time.time() - start_time, avg_loss/avg_num))
					
					if epoch%retrain_epochs==retrain_epochs-1 and idx%retrain_iters==retrain_iters-1:
						np.random.shuffle(batch_rindex_list)
						for ridx in range(batch_ridxs):
							dxb = batch_rindex_list[ridx]
							self.sess.run(ae_optim_supervised,
								feed_dict={
									self.vox3d: self.ref_voxels[dxb:dxb+1],
									self.branch_coord: self.ref_b_points[dxb,:self.ref_b_point_num[dxb]],
									self.branch_value: self.ref_b_values[dxb,:self.ref_b_point_num[dxb]],
									self.point_coord: self.ref_points[dxb],
									self.point_value: self.ref_values[dxb],
								})
			# supervised training
			else:
				np.random.shuffle(batch_rindex_list)
				avg_loss = 0
				avg_num = 0
				for ridx in range(batch_ridxs):
					dxb = batch_rindex_list[ridx]
					_, errAE = self.sess.run([ae_optim_supervised, self.loss_supervised],
						feed_dict={
							self.vox3d: self.ref_voxels[dxb:dxb+1],
							self.branch_coord: self.ref_b_points[dxb,:self.ref_b_point_num[dxb]],
							self.branch_value: self.ref_b_values[dxb,:self.ref_b_point_num[dxb]],
							self.point_coord: self.ref_points[dxb],
							self.point_value: self.ref_values[dxb],
						})
					avg_loss += errAE
					avg_num += 1
					
					if (ridx==batch_ridxs-1):
						print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f" % (epoch, training_epoch, ridx, batch_ridxs, time.time() - start_time, avg_loss/avg_num))
			
			if epoch%int(training_epoch/8)==0 and epoch>0:
				self.save(config.checkpoint_dir, epoch)
				if config.supervised:
					self.test_pcSeg(config,epoch,True)
		
		if training_epoch%int(training_epoch/8)!=0:
			self.save(config.checkpoint_dir, training_epoch)
			if config.supervised:
				self.test_pcSeg(config,epoch,True)
	
	
	
	# -------- quantitative evaluation --------
	def test_pcSeg(self, FLAGS, epoch=None, inplace=False):
		if not inplace:
			could_load, checkpoint_counter = self.load(self.checkpoint_dir)
			if could_load:
				print(" [*] Load SUCCESS")
				epoch = checkpoint_counter
			else:
				print(" [!] Load failed...")
				return
		
		num_of_test_shapes = self.test_voxels.shape[0]
		shape_mIOU = [None] * num_of_test_shapes
		for t in range(num_of_test_shapes):
			batch_voxels = self.test_voxels[t:t+1]
			b_point_num = self.test_b_point_num[t]
			branch_coord = self.test_b_points[t,:b_point_num]
			branch_value = self.test_b_values[t,:b_point_num]
			
			z_out = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			model_out = self.sess.run(self.bG,
				feed_dict={
					self.z_vector: z_out,
					self.point_coord: branch_coord,
				})
			pred_part_labels = np.argmax(model_out, axis=1).astype(np.int32)
			
			#evaluation
			gtLables = np.argmax(branch_value, axis=1).astype(np.int32)
			part_ious = [0.0] * len(self.labels_unique)
			for i in range(len(self.labels_unique)):
				if (np.sum(gtLables==i) == 0) and (np.sum(pred_part_labels==i) == 0): # part is not present, no prediction as well
					part_ious[i] = 1.0
				else:
					part_ious[i] = np.sum(( gtLables==i ) & ( pred_part_labels==i )) / float(np.sum(   ( gtLables==i ) | ( pred_part_labels==i ) ))
			
			shape_mIOU[t] = np.mean(part_ious)
		
		#write numbers
		#with open( os.path.join(self.checkpoint_dir, self.model_dir, FLAGS.dataset+'_epoch_'+str(epoch)+'_numbers.txt') , 'w' )  as outfile:
		#	for t in range(num_of_test_shapes):
		#		outfile.write(str(shape_mIOU[t])+"\n")
		
		cate_mIOU = np.round(np.mean(shape_mIOU)*1000.0)/10
		with open( os.path.join(self.checkpoint_dir, self.model_dir, FLAGS.dataset+'_epoch_'+str(epoch)+'_average.txt') , 'w' )  as outfile:
			outfile.write(str(cate_mIOU))
		
		print()
		print(self.data_dir)
		print(cate_mIOU)
		print()
	
	
	
	#output colored implicit field
	def test_dae(self, config):
		import mcubes
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		color_list = ["255 0 0","0 255 0","0 0 255","255 255 0","255 0 255","0 255 255","180 180 180", "100 100 100", "255 128 128","128 255 128","128 128 255","255 255 128","255 128 255","128 255 255"]
		
		dima = self.test_size
		dim = self.real_size
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		
		for t in range(min(len(self.ref_voxels),16)):
			model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2,self.gf_split],np.float32)
			batch_voxels = self.ref_voxels[t:t+1]
			z_out = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						model_out = self.sess.run(self.bG,
							feed_dict={
								self.z_vector: z_out,
								self.point_coord: self.coords[minib],
							})
						model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1,:] = np.reshape(model_out, [self.test_size,self.test_size,self.test_size,self.gf_split])
			
			thres = 0.4
			vertices_num = 0
			triangles_num = 0
			vertices_list = []
			triangles_list = []
			vertices_num_list = [0]
			for split in range(self.gf_split):
				vertices, triangles = mcubes.marching_cubes(model_float[:,:,:,split], thres)
				vertices_num += len(vertices)
				triangles_num += len(triangles)
				vertices_list.append(vertices)
				triangles_list.append(triangles)
				vertices_num_list.append(vertices_num)
			
			#output ply
			fout = open(config.sample_dir+"/"+str(t)+"_vox.ply", 'w')
			fout.write("ply\n")
			fout.write("format ascii 1.0\n")
			fout.write("element vertex "+str(vertices_num)+"\n")
			fout.write("property float x\n")
			fout.write("property float y\n")
			fout.write("property float z\n")
			fout.write("property uchar red\n")
			fout.write("property uchar green\n")
			fout.write("property uchar blue\n")
			fout.write("element face "+str(triangles_num)+"\n")
			fout.write("property uchar red\n")
			fout.write("property uchar green\n")
			fout.write("property uchar blue\n")
			fout.write("property list uchar int vertex_index\n")
			fout.write("end_header\n")
			
			for split in range(self.gf_split):
				vertices = (vertices_list[split])/self.real_size-0.5
				for i in range(len(vertices)):
					color = color_list[split]
					fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
			
			for split in range(self.gf_split):
				triangles = triangles_list[split] + vertices_num_list[split]
				for i in range(len(triangles)):
					color = color_list[split]
					fout.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
			
			#output separated files for different parts
			if t==-1:
				vertices, triangles = mcubes.marching_cubes(batch_voxels[0,:,:,:,0], thres)
				#output input vox ply
				fout1 = open(config.sample_dir+"/"+str(t)+"_input.ply", 'w')
				fout1.write("ply\n")
				fout1.write("format ascii 1.0\n")
				fout1.write("element vertex "+str(len(vertices))+"\n")
				fout1.write("property float x\n")
				fout1.write("property float y\n")
				fout1.write("property float z\n")
				fout1.write("property uchar red\n")
				fout1.write("property uchar green\n")
				fout1.write("property uchar blue\n")
				fout1.write("element face "+str(len(triangles))+"\n")
				fout1.write("property uchar red\n")
				fout1.write("property uchar green\n")
				fout1.write("property uchar blue\n")
				fout1.write("property list uchar int vertex_index\n")
				fout1.write("end_header\n")
				color = "180 180 180"
				vertices = (vertices)/self.real_size-0.5
				for i in range(len(vertices)):
					fout1.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
				for i in range(len(triangles)):
					fout1.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
				fout1.close()
				
				for split in range(self.gf_split):
					vertices = (vertices_list[split])/self.real_size-0.5
					triangles = triangles_list[split]
					#output part ply
					fout1 = open(config.sample_dir+"/"+str(t)+"_vox_"+str(split)+".ply", 'w')
					fout1.write("ply\n")
					fout1.write("format ascii 1.0\n")
					fout1.write("element vertex "+str(len(vertices))+"\n")
					fout1.write("property float x\n")
					fout1.write("property float y\n")
					fout1.write("property float z\n")
					fout1.write("property uchar red\n")
					fout1.write("property uchar green\n")
					fout1.write("property uchar blue\n")
					fout1.write("element face "+str(len(triangles))+"\n")
					fout1.write("property uchar red\n")
					fout1.write("property uchar green\n")
					fout1.write("property uchar blue\n")
					fout1.write("property list uchar int vertex_index\n")
					fout1.write("end_header\n")
					for i in range(len(vertices)):
						color = color_list[split]
						fout1.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
					for i in range(len(triangles)):
						color = color_list[split]
						fout1.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
					fout1.close()
			
			fout.close()
			
			print("[sample]")
	
	#output colored point cloud
	def test_pointcloud(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		color_list = ["255 0 0","0 255 0","0 0 255","255 255 0","255 0 255","0 255 255","180 180 180", "100 100 100", "255 128 128","128 255 128","128 128 255","255 255 128","255 128 255","128 255 255"]
		
		for t in range(min(len(self.data_voxels),32)):
			batch_voxels = self.ref_voxels[t:t+1]
			b_point_num = self.ref_b_point_num[t]
			branch_coord = self.ref_b_points[t,:b_point_num]
			branch_value = self.ref_b_values[t,:b_point_num]
			z_out = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			model_out = self.sess.run(self.bG,
				feed_dict={
					self.z_vector: z_out,
					self.point_coord: branch_coord,
				})
				
			label_gt = np.argmax(branch_value, axis=1)
			label_out = np.argmax(model_out, axis=1)
			label_max = np.max(model_out, axis=1)
			
			#output ply
			fout = open(config.sample_dir+"/"+str(t)+"_gt.ply", 'w')
			fout.write("ply\n")
			fout.write("format ascii 1.0\n")
			fout.write("element vertex "+str(b_point_num)+"\n")
			fout.write("property float x\n")
			fout.write("property float y\n")
			fout.write("property float z\n")
			fout.write("property uchar red\n")
			fout.write("property uchar green\n")
			fout.write("property uchar blue\n")
			fout.write("end_header\n")
			for i in range(b_point_num):
				color = color_list[label_gt[i]]
				fout.write(str(branch_coord[i,0])+" "+str(branch_coord[i,1])+" "+str(branch_coord[i,2])+" "+color+"\n")
			fout.close()
			
			
			#output ply
			fout = open(config.sample_dir+"/"+str(t)+"_out.ply", 'w')
			fout.write("ply\n")
			fout.write("format ascii 1.0\n")
			fout.write("element vertex "+str(b_point_num)+"\n")
			fout.write("property float x\n")
			fout.write("property float y\n")
			fout.write("property float z\n")
			fout.write("property uchar red\n")
			fout.write("property uchar green\n")
			fout.write("property uchar blue\n")
			fout.write("end_header\n")
			for i in range(b_point_num):
				color = color_list[label_out[i]]
				fout.write(str(branch_coord[i,0])+" "+str(branch_coord[i,1])+" "+str(branch_coord[i,2])+" "+color+"\n")
			fout.close()
			'''
			#output ply
			fout = open(config.sample_dir+"/"+str(t)+"_out.ply", 'w')
			fout.write("ply\n")
			fout.write("format ascii 1.0\n")
			fout.write("element vertex "+str(len(self.ref_points[t]))+"\n")
			fout.write("property float x\n")
			fout.write("property float y\n")
			fout.write("property float z\n")
			fout.write("property uchar red\n")
			fout.write("property uchar green\n")
			fout.write("property uchar blue\n")
			fout.write("end_header\n")
			for i in range(len(self.ref_points[t])):
				if label_max[i]>0.1:
					color = color_list[label_out[i]]
					fout.write(str(self.ref_points[t,i,0])+" "+str(self.ref_points[t,i,1])+" "+str(self.ref_points[t,i,2])+" "+color+"\n")
				else:
					fout.write("0 0 0 0 0 0\n")
			fout.close()
			'''
			
			
			print("[sample]")
	
	#output colored mesh
	def test_obj(self, config):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		color_list = ["255 0 0","0 255 0","0 0 255","255 255 0","255 0 255","0 255 255","180 180 180", "100 100 100", "255 128 128","128 255 128","128 128 255","255 255 128","255 128 255","128 255 255"]
		
		
		
		for t in range(min(len(self.data_voxels),16)):
			obj_name = self.ref_obj_name[t]
			obj_dir = "F:\\ShapeNetCore.v2\\ShapeNetCore.v2\\"+self.dataset_name[:8]+"\\"+obj_name+"\\models\\model_normalized.obj"
			
			vertices, triangles = load_obj(obj_dir)
			
			'''
			#if shapenetV1:
			vertices_new = np.copy(vertices)
			vertices_new[:,0] = vertices[:,2]
			vertices_new[:,1] = vertices[:,1]
			vertices_new[:,2] = -vertices[:,0]
			vertices = vertices_new
			'''
			
			batch_voxels = self.ref_voxels[t:t+1]
			z_out = self.sess.run(self.sE,
				feed_dict={
					self.vox3d: batch_voxels,
				})
			
			vertices_minibatch = []
			vertices_ = vertices
			while len(vertices_)>8192:
				vertices_minibatch.append(vertices_[:8192])
				vertices_ = vertices_[8192:]
			vertices_minibatch.append(vertices_)
			
			out_minibatch = []
			for minib in range(len(vertices_minibatch)):
				out = self.sess.run(self.bG,
					feed_dict={
						self.z_vector: z_out,
						self.point_coord: vertices_minibatch[minib],
					})
				out_minibatch.append(out)
			model_out = np.concatenate(out_minibatch, axis=0)
			
			label_out = np.argmax(model_out, axis=1)
			label_max = np.max(model_out, axis=1)
			
			#output ply
			fout = open(config.sample_dir+"/"+str(t)+"_mesh.ply", 'w')
			fout.write("ply\n")
			fout.write("format ascii 1.0\n")
			fout.write("element vertex "+str(len(vertices))+"\n")
			fout.write("property float x\n")
			fout.write("property float y\n")
			fout.write("property float z\n")
			fout.write("property uchar red\n")
			fout.write("property uchar green\n")
			fout.write("property uchar blue\n")
			fout.write("element face "+str(len(triangles))+"\n")
			fout.write("property uchar red\n")
			fout.write("property uchar green\n")
			fout.write("property uchar blue\n")
			fout.write("property list uchar int vertex_index\n")
			fout.write("end_header\n")
			
			for i in range(len(vertices)):
				color = color_list[label_out[i]]
				fout.write(str(vertices[i,0])+" "+str(vertices[i,1])+" "+str(vertices[i,2])+" "+color+"\n")
			
			for i in range(len(triangles)):
				labels = model_out[triangles[i,0]] + model_out[triangles[i,1]] + model_out[triangles[i,2]]
				
				color = color_list[np.argmax(labels)]
				fout.write(color+" 3 "+str(triangles[i,0])+" "+str(triangles[i,1])+" "+str(triangles[i,2])+"\n")
			
			
			fout.close()
			
			print("[sample]")
	
	
	
	
	@property
	def model_dir(self):
		return "{}_{}".format(
				self.dataset_name, self.input_size)
			
	def save(self, checkpoint_dir, step):
		model_name = "IMSEG.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

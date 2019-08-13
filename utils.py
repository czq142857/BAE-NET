import numpy as np 
import math


def sqdist(p1,p2):
	return np.sum(np.square(p1-p2))
def midpoint(p1,p2):
	return (p1+p2)/2

def my_simple_subdiv(vertices, triangles, threshold):
	threshold2 = threshold*threshold
	
	index_p1_p2 = []
	index_pmid = []
	
	for i in range(len(triangles)):
		pi1 = triangles[i][0]
		pi2 = triangles[i][1]
		pi3 = triangles[i][2]
		p1 = vertices[pi1]
		p2 = vertices[pi2]
		p3 = vertices[pi3]
		
		if sqdist(p1,p2)>threshold2 or sqdist(p1,p3)>threshold2 or sqdist(p2,p3)>threshold2:
			#subdiv
			current_len = len(vertices)
			current_counter = 0
			
			if (pi1,pi2) in index_p1_p2:
				pi12 = index_pmid[index_p1_p2.index((pi1,pi2))]
			elif (pi2,pi1) in index_p1_p2:
				pi12 = index_pmid[index_p1_p2.index((pi2,pi1))]
			else:
				vertices.append(midpoint(p1,p2))
				pi12 = current_len+current_counter
				current_counter +=1
				index_p1_p2.append((pi1,pi2))
				index_pmid.append(pi12)
			
			if (pi3,pi2) in index_p1_p2:
				pi23 = index_pmid[index_p1_p2.index((pi3,pi2))]
			elif (pi2,pi3) in index_p1_p2:
				pi23 = index_pmid[index_p1_p2.index((pi2,pi3))]
			else:
				vertices.append(midpoint(p3,p2))
				pi23 = current_len+current_counter
				current_counter +=1
				index_p1_p2.append((pi3,pi2))
				index_pmid.append(pi23)
			
			if (pi3,pi1) in index_p1_p2:
				pi13 = index_pmid[index_p1_p2.index((pi3,pi1))]
			elif (pi1,pi3) in index_p1_p2:
				pi13 = index_pmid[index_p1_p2.index((pi1,pi3))]
			else:
				vertices.append(midpoint(p3,p1))
				pi13 = current_len+current_counter
				current_counter +=1
				index_p1_p2.append((pi1,pi3))
				index_pmid.append(pi13)
			
			
			triangles[i][1]=pi12
			triangles[i][2]=pi13
			triangles.append([pi2,pi23,pi12])
			triangles.append([pi3,pi13,pi23])
			triangles.append([pi12,pi23,pi13])
	
	return vertices, triangles


def load_obj(shape_name):
	fin = open(shape_name,'r')
	lines = fin.readlines()
	fin.close()
	
	vertices = []
	triangles = []
	
	for i in range(len(lines)):
		line = lines[i].split()
		if len(line)==0:
			continue
		if line[0] == 'v':
			x = float(line[1])
			y = float(line[2])
			z = float(line[3])
			vertices.append([x,y,z])
		if line[0] == 'f':
			x = int(line[1].split("/")[0])
			y = int(line[2].split("/")[0])
			z = int(line[3].split("/")[0])
			triangles.append([x-1,y-1,z-1])
	
	vertices = np.array(vertices, np.float32)
	
	
	#remove isolated points
	triangles_ = np.array(triangles, np.int32).reshape([-1])
	vertices_ = vertices[triangles_]
	
	
	#normalize diagonal=1
	x_max = np.max(vertices_[:,0])
	y_max = np.max(vertices_[:,1])
	z_max = np.max(vertices_[:,2])
	x_min = np.min(vertices_[:,0])
	y_min = np.min(vertices_[:,1])
	z_min = np.min(vertices_[:,2])
	
	x_mid = (x_max+x_min)/2
	y_mid = (y_max+y_min)/2
	z_mid = (z_max+z_min)/2
	
	x_scale = x_max - x_min
	y_scale = y_max - y_min
	z_scale = z_max - z_min
	
	scale = math.sqrt(x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)
	
	'''
	#normalize max=1
	x_max = np.max(vertices_[:,0])
	y_max = np.max(vertices_[:,1])
	z_max = np.max(vertices_[:,2])
	x_min = np.min(vertices_[:,0])
	y_min = np.min(vertices_[:,1])
	z_min = np.min(vertices_[:,2])
	
	x_mid = (x_max+x_min)/2
	y_mid = (y_max+y_min)/2
	z_mid = (z_max+z_min)/2
	
	x_scale = x_max - x_min
	y_scale = y_max - y_min
	z_scale = z_max - z_min
	
	scale = max( max(x_scale, y_scale), z_scale)
	'''
	
	vertices = [ppp for ppp in vertices]
	print(len(vertices), len(triangles))
	if len(triangles)<100000:
		vertices, triangles = my_simple_subdiv(vertices, triangles, 0.02*scale)
		print(len(vertices), len(triangles))
	if len(triangles)<100000:
		vertices, triangles = my_simple_subdiv(vertices, triangles, 0.02*scale)
		print(len(vertices), len(triangles))
	if len(triangles)<100000:
		vertices, triangles = my_simple_subdiv(vertices, triangles, 0.02*scale)
		print(len(vertices), len(triangles))
	vertices = np.array(vertices, np.float32)
	triangles = np.array(triangles, np.int32)
	
	
	vertices[:,0] = (vertices[:,0]-x_mid)/scale
	vertices[:,1] = (vertices[:,1]-y_mid)/scale
	vertices[:,2] = (vertices[:,2]-z_mid)/scale
	
	return vertices, triangles



#.txt format  --  X,Y,Z, normalX,normalY,normalZ, label
def parse_txt_points(shape_name,gf_split,labels_unique):
	#open file & read points
	file = open(shape_name, 'r')
	lines = file.readlines()
	file.close()
	
	points = []
	labels = []
	for i in range(len(lines)):
		line = lines[i].split()
		points.append([float(line[2]),float(line[1]),-float(line[0])])
		labels.append(int(float(line[6])))
	
	point_num = len(labels)
	shape_points = np.array(points, np.float32)
	shape_values = np.zeros([point_num,gf_split], np.float32)
	
	
	#fill labels for each branch
	for i in range(point_num):
		k = labels_unique.index(labels[i])
		shape_values[i,k] = 1
	
	return shape_points, shape_values, point_num


def get_list_of_labels(txt_name):
	#open file & read points
	file = open(txt_name, 'r')
	lines = file.readlines()
	file.close()
	labels = []
	for i in range(len(lines)):
		line = lines[i].split()
		labels.append(int(float(line[6])))
	labels_unique = list(np.unique(labels))
	return labels_unique


def parse_txt_list(ref_txt_name, data_dir, data_txt_name):
	#open file & read points
	ref_file = open(ref_txt_name, 'r')
	ref_names = [line.strip() for line in ref_file]
	ref_file.close()
	data_file = open(data_txt_name, 'r')
	data_names = [line.strip() for line in data_file]
	data_file.close()
	
	num_shapes = len(ref_names)
	point_num_max = 3000
	
	labels = []
	for i in range(num_shapes):
		shape_name = data_dir+"/"+ref_names[i]+".txt"
		labels += get_list_of_labels(shape_name)
	labels_unique = list(np.unique(labels))
	labels_unique = sorted(labels_unique)
	gf_split = len(labels_unique)
	
	ref_points = np.zeros([num_shapes,point_num_max,3], np.float32)
	ref_values = np.zeros([num_shapes,point_num_max,gf_split], np.float32)
	ref_point_num = np.zeros([num_shapes], np.int32)
	idx = np.zeros([num_shapes], np.int32)
	
	for i in range(num_shapes):
		shape_name = data_dir+"/"+ref_names[i]+".txt"
		shape_idx = data_names.index(ref_names[i])
		shape_points, shape_values, point_num = parse_txt_points(shape_name,gf_split,labels_unique)
		
		ref_points[i,:point_num,:] = shape_points
		ref_values[i,:point_num,:] = shape_values
		ref_point_num[i] = point_num
		idx[i] = shape_idx
	
	return ref_points, ref_values, ref_point_num, gf_split, idx, labels_unique, ref_names

def parse_txt_list_unsupervised(ref_txt_name, data_txt_name):
	#open file & read points
	ref_file = open(ref_txt_name, 'r')
	ref_names = [line.strip() for line in ref_file]
	ref_file.close()
	data_file = open(data_txt_name, 'r')
	data_names = [line.strip() for line in data_file]
	data_file.close()
	
	num_shapes = len(ref_names)
	idx = np.zeros([num_shapes], np.int32)
	
	for i in range(num_shapes):
		shape_idx = data_names.index(ref_names[i])
		idx[i] = shape_idx
	
	return idx, ref_names


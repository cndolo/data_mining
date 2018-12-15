import sys, getopt
import random
import numpy as np
import math
from PIL import Image 

w_file = ""
r_file = ""
k = 0
size = 0
normal = list()  # store max values for initial center calculation
iterations = 0
running = True

##Read CL Arguments i.e image to cluster, and file to write to
def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hi:o:k:",["ifile=","ofile=","k="])
    except getopt.GetoptError:
        print ("k_means.py -i <inputfile> -o <outputfile> -k <value>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == ('-h' or "--help"):
            print ("k_means.py -i <inputfile> -o <outputfile> -k<value>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
           global r_file 
           r_file = arg
        elif opt in ("-o", "--ofile"):
           global  w_file 
           w_file = arg
        elif opt in ("-k", "--k"):
            global k
            k = int(arg)
            
"""
    Reads image and converst is to list of rgb pixels
    r_file: file to read
"""
def load_image(r_file):
    image = Image.open(r_file)
    global size
    size = image.width, image.height
    data = to_rgb(image)
    return data
"""
    image: image to be saved in w_file
"""
def save_image(image,w_file):
    out= "PNG"
    #image.show()
    image.save(w_file,out)
"""
    get RGB value of pixel (x,y) of image
"""
def get_rgb(image, x, y):
    return image.getpixel((x,y))
"""
    set RGB value of pixel(x,y) of image
"""
def set_rgb(image, x, y, r_g_b):
     r,g,b = r_g_b
     image.putpixel((x, y), (r, g, b))
"""
    image: image to be transformed to RGB list
    return: image as RGB list
"""
def to_rgb(image):
    x_w, y_w = image.width, image.height
    data = list()
    for x in range(0, x_w):
        for y in range(0, y_w):
            vector_data = list(get_rgb(image, x, y))
            data.append(vector_data)
    return data
"""
    r_file: Image that has been clustered
    cluster: 
    return: 
"""
def from_rgb(r_file,cluster):
    image = Image.open(r_file)
    w, h = image.width, image.height
    col = get_colours()
    pos = 0
    for x in range(0,w):
        for y in range(0,h):
            c_id = cluster[pos]
            set_rgb(image,x,y,col[c_id])
            pos+=1
    return image
"""
    Returns evenly spaced distinct RGB coklours for each cluster
"""
def get_colours():
    
    max_value = 16581375 #255**3
    global k
    interval = int(max_value / k)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

"""
    takes 2 points of equal length as input
    returns metric distance between the 2 points
"""
def euc_dist(p1,p2):
    
    sq_dist = 0
    for flag in range(len(p1)):
        sq_dist += math.pow(p1[flag] - p2[flag], 2)
    euc_dist = math.sqrt(sq_dist)
    return euc_dist;
"""
    returns a centeroid in the range of max values specified by normal
    normal: a list or tuple of metric maximum values
    cluster_id: the cluster id for the centeroid to be returned
    return: the tuple of centeroid, cluster id
"""
def random_centeroid(normal, cluster_id):
    random_center = tuple(
        random.randint(0, value) for value in normal)
    return random_center, cluster_id

"""
   data: to be clustered data set
   k: number of partitions
   max_iter: maximum number of runs for the algorithm
   max_cent_dist: maximum distance between old and new centroid cluster
"""

def k_means(data, k, max_iter, max_cent_dist):

    cluster_ids = list()  # list of cluster_ids


    for d in range(len(data[0])):
        global normal
        normal.append(0)

    for d in data:
        cluster_ids.append(0)  # assign each data point to cluster 0 initially
        for val in range(len(data[0])):
            normal[val] = max(normal[val], d[val])

    # chose k points as center
    cent_new = list()
    cent_old = list()
    for cluster_id in range(0, k):
        cent_new.append(random_centeroid(normal, cluster_id))

    global running
    running = True
    iter = 0
    while running:
        running = False
        assign(data,cluster_ids,cent_new)
        results = update(data,cluster_ids,cent_old, cent_new, max_iter,max_cent_dist)
    return results;


"""
    Function iteratively assigns points to the closest centeroid
    data: data set to be clustered
    cluster_id:
    cent_new:
    
"""
def assign(data,cluster_ids,cent_new):
    
    for index in range(len(data)): ##point index
        point = data[index]  #datapoint
        current_id = cluster_ids[index] 
        curr_cent, c_id = cent_new[current_id] #current_center, clusteR_id
        dist_min = euc_dist(point, curr_cent)
        for center, c_id in cent_new:
            new_dist = euc_dist(point, center)
            if new_dist < dist_min:  # data point is closer to other center
                cluster_ids[index] = c_id  # assign new cluster_id
                dist_min = new_dist # new minimal distance
"""
    data: data set to be clusters
    cluster_ids:
    cent_old:
    cent_new:
    max_iter: Maximum number of runs for the algorithm
    max_cent_dist: Maximum distance between old and new centeroid
"""

def update(data,cluster_ids,cent_old,cent_new,max_iter,max_cent_dist):
    cent_old += cent_new
    #finding average for each vector value
    total = list()
    for i in range(k):
        ctr = 0
        zero = [0] * len(data[0])
        total.append((zero, ctr))

    for point in range(len(data)):
        dp = data[point]
        cluster_id = cluster_ids[point]
        # vector to sum for center calculation
        tot_1, ctr = total[cluster_id]
        for i, value in enumerate(dp):
            tot_1[i] += value
        ctr += 1
        total[cluster_id] = tot_1, ctr

    for cluster_id in range(k):
        tot_2, ctr = total[cluster_id]
        if ctr == 0:
            centeroid = random_centeroid(normal, cluster_id)
        else:
            centeroid = (tuple(value / ctr for value in tot_2), cluster_id)
        cent_new.append(centeroid)
    
    dist = 0
    distances = list()
    for i in range(k):
        oc, ocid = cent_old[i]
        nc, ncid = cent_new[i]
        old_dist = euc_dist(oc, nc)
        distances.append((i, old_dist))
        dist = max(dist, old_dist)

        # check for exit conditions
        if dist > max_cent_dist:
            global running
            running = True

        global iterations
        iterations += 1
        if iterations >= max_iter:
            running = False

    return cluster_ids


if __name__ == "__main__":
    
    main(sys.argv[1:])
    data = load_image(r_file)
    clustered = k_means(data, k, 10, 20) 
    new_im = from_rgb(r_file,clustered)
    save_image(new_im, w_file)



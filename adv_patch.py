import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import os
import csv
import shutil
import matplotlib.pyplot as plt
import random

import skimage.io as io
from skimage.transform import resize,rescale
from sklearn.cluster import DBSCAN,KMeans
from scipy.sparse import coo_matrix

colors = [[0,255,0], #green
          [255,0,0], #red
          [255,255,0], #yellow
          [255,0,255], #magenta
          [0,255,255], #cyan
          [0,0,255], #blue
          [0,0,0], #black
          [46,139,87], #seagreen
          [0,206,209], #darkturquoise
          [127,255,122], #aquamarine
          [174,238,238], #paleturquoise
          [255,255,210], #lightgoldenrodyellow
          [205,133,63] #peru
]

matplot_colors =['green','red','yellow','magenta','cyan','blue','black','seagreen','darkturquoise','aquamarine','paleturquoise','lightgoldenrodyellow','peru']

# Prepare image to network input format
def prep(im):
    if len(im.shape)==3:
        return np.transpose(im,[2,0,1]).reshape((1,3,112,112))*2-1     #像素值转换到[-1,1]之间了--为什么不是[0,1]之间？？？
    elif len(im.shape)==4:
        return np.transpose(im,[0,3,1,2]).reshape((im.shape[0],3,112,112))*2-1

# Print distance matrix
def write_csv(name_list, loss, csv_file_name):
    """
    write to csv file
    :param name_list: list of pic names
    :param loss: loss
    :param csv_file_name: csv file name
    """
    header = [""]+name_list

    num_img = len(name_list)
    with open(csv_file_name, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)

        row = ["Positive"]
        for i in range(num_img):
            row.append('%1.3f'%loss[i])
        f_csv.writerow(row)

def calcu_cam(last_conv, norm_grad_conv):
    norm_grad_conv = norm_grad_conv.transpose(0, 2, 3, 1)  # [?, 7,7,512]  --[?, 512,7,7] 维度置换成->[?, 7,7,512]

    # 归一化
    # grads_val_0_1 = (norm_grad_conv_run - np.min(norm_grad_conv_run)) / (np.max(norm_grad_conv_run) - np.min(norm_grad_conv_run))
    # norm_grad_input_0_1 = (norm_grad_input_run - np.min(norm_grad_input_run)) / (np.max(norm_grad_input_run) - np.min(norm_grad_input_run))
    grads_val_0_1 = np.zeros_like(norm_grad_conv, dtype=np.float32)

    cam = np.zeros((norm_grad_conv.shape[0],112,112), dtype=np.float32)  # [?,112,112]
    for i in range(norm_grad_conv.shape[0]):
        grad_conv_max = np.max(np.abs(norm_grad_conv[i,:,:,:]))
        grads_val_0_1[i,:,:,:] = np.abs(norm_grad_conv[i,:,:,:]) / grad_conv_max  # [?,7,7,512]

        weights = np.mean(grads_val_0_1[i,:,:,:], axis=(0,1) )  # [512]

        # Taking a weighted average
        cam_small = np.zeros_like(last_conv[i,0,:,:])
        for j, w in enumerate(weights):
            cam_small += w * np.abs(last_conv[i, j,:,:])

        # Passing through ReLU
        cam_small = cam_small / np.max(cam_small)  # [7,7]
        cam_resized = resize(cam_small, (112, 112))  # [resize,resize] -- 与输入的图片大小一致

        # Converting grayscale to 3-D
        cam[i,:,:] = cam_resized

    return cam, grads_val_0_1


def cacu_norm_grad(grad_input):
    """
    normalization the gradient
    :param grad_input: gradient [?, 3, 112,112]
    :return: normalized gradient [?, 112,112]
    """
    grad_input = grad_input.sum(axis=1) # [?, 112,112]
    norm_grad_input_0_1 = np.zeros_like(grad_input, dtype=np.float32)

    for i in range(grad_input.shape[0]):
        max = np.max(np.abs(grad_input[i, :, :]))
        norm_grad_input_0_1[i, :, :] = np.abs(grad_input[i, :, :]) / max  # [?,112,112]

    return norm_grad_input_0_1

def dbscan(eps, min_samples, img_mask, img_grad):
    """
    dbscan implementation
    :param eps: eps of dbscan
    :param min_samples: minimum samples in a cluster
    :param img_mask: threshold mask, 1 for the points beyond the threshold
    :param img_grad: gradient of the image
    :return: the list of clusters sorted by weight (label, weight, center ,cluster_mask)
    """
    data = np.argwhere(img_mask == 1)
    labels_db = DBSCAN(eps=eps,min_samples=min_samples).fit(data)
    labels = labels_db.labels_ # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    cluster_sorted = sortClusterByWeight(data, img_grad, img_mask, labels, n_clusters)
    new_cluster_sorted = []
    for label, weight, cluster in cluster_sorted:
        points = np.argwhere(cluster == 1.0)
        center_point = np.round(np.mean(points, axis=0)).astype(np.uint8)
        new_cluster_sorted.append((label, weight, center_point, cluster))
    return new_cluster_sorted


def sortClusterByWeight(data, img_grad, img_mask, labels, n_clusters):
    """
    sort the cluster by weight
    :param data: the input points
    :param img_grad: gradient of the image
    :param img_mask: threshold mask, 1 for the points beyond the threshold
    :param labels: the result of cluster algorithm (dbscan or kmeans)
    :param n_clusters: number of clusters
    :return: the list of clusters sorted by weight (label, weight, cluster_mask)
    """
    cluster_sorted = []
    for i in range(n_clusters):
        one_cluster = data[labels == i]
        mask_data = np.ones(one_cluster.shape[0])
        row = one_cluster[:, 0]
        col = one_cluster[:, 1]
        cluster_mask = coo_matrix((mask_data, (row, col)), shape=img_mask.shape).toarray()
        weight = np.sum(cluster_mask * img_grad)

        # insert into list, sorted by weight
        inserted = False
        for j in range(len(cluster_sorted)):
            w, _, _ = cluster_sorted[j]
            if weight <= w:
                continue
            else:
                cluster_sorted.insert(j, (i, weight,  cluster_mask))
                inserted = True
                break

        if not inserted:
            cluster_sorted.append((i, weight, cluster_mask))
    return cluster_sorted


def kmeans(max_cluster, img_mask, img_grad):
    """
    kmeans algorithm
    :param max_cluster: number of cluster
    :param img_mask: threshold mask, 1 for the points beyond the threshold
    :param img_grad: gradient of the image
    :return: the list of clusters sorted by weight (label, weight, center ,cluster_mask)
    """
    data = np.argwhere(img_mask == 1)
    if max_cluster * 10 > len(data):
        max_cluster = int(len(data) / 10)
    km = KMeans(n_clusters=max_cluster).fit(data)
    labels = km.labels_
    center_points = km.cluster_centers_
    cluster_sorted = sortClusterByWeight(data, img_grad, img_mask, labels, max_cluster)
    new_cluster_sorted = []

    for label, weight, cluster in cluster_sorted:
        center = (int(center_points[label, 0]), int(center_points[label, 1]))
        new_cluster_sorted.append((label, weight, center, cluster))

    return new_cluster_sorted

def randomcluster(max_cluster, img_grad):
    """
    choose cluster randomly
    :param max_cluster: number of cluster
    :param img_grad: gradient of the image
    :return: the sorted list of weight and the list of clusters sorted by weight
    """
    w, h = img_grad.shape
    a_cluster = np.zeros((w,h), dtype=np.int32)
    cluster = []
    for i in range(max_cluster):
        center = (random.randint(0,w-1), random.randint(0,h-1))
        cluster.append((i,0,center,a_cluster))
    return cluster
#
def get_patch(mask, center, radius):
    """
    get a square patch
    :param mask: mask size
    :param center: center point
    :param radius: length of the square would be 2*radius + 1
    :return: the patch mask, 1 for the points in the patch
    """
    _,w,h = mask.shape
    patch = np.zeros((1, w, h), dtype=np.int32)
    w_left = center[0]-radius if center[0]-radius > 0 else 0
    w_right = center[0]+radius + 1 if center[0]+radius + 1 <= w  else w
    h_bot = center[1]-radius if center[1]-radius > 0 else 0
    h_top = center[1] + radius + 1 if center[1] + radius + 1 <= h else h
    patch[:,w_left:w_right,h_bot:h_top]=1
    return patch

def singlePatchAttack(args, img, clusters, session, positive_embedding, adv_filename, grad_input, positive, loss):
    """
    Single Patch
    :param args: command arguments
    :param img: the image
    :param clusters: gradient clusters (label, weight, center, cluster_mask)
    :param session: session
    :param positive_embedding: positive embedding
    :param adv_filename: the adversarial image name
    :param grad_input: gradient to input image (tensor)
    :param positive: positive image (tf.placeholder)
    :param loss: loss (tensor)
    :return:
    """
    img = img.transpose(2,0,1)
    n_cluster = min(len(clusters),args.max_cluster)
    if n_cluster == 0:
        print('%s: no cluster '%adv_filename)
        return

    image_input = tf.get_default_graph().get_tensor_by_name('image_input:0')  # (?, 3, 112, 112)
    keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
    is_train = tf.get_default_graph().get_tensor_by_name('training_mode:0')
    embedding = tf.get_default_graph().get_tensor_by_name('embedding:0')  # (?, 512)

    moment_val = 0.9
    step_val = 1.0 / args.gradient_step
    c, w,h = img.shape

    grad = np.zeros_like(img)
    img_grad = np.stack([grad for i in range(n_cluster)])
    cluster_imgs = np.stack([img for i in range(n_cluster)])
    r = 1
    success = False
    success_cluster_no = []
    success_loss = []
    while not success:
        print("radius = %d"%r)
        #create patches for all clusters
        patch_list = []
        for _, _, center, _ in clusters:
            patch_mask = get_patch(img, center, radius=r)
            patch_list.append(patch_mask)
            if len(patch_list) >= n_cluster:
                break;
        patches = np.stack(patch_list)

        moments = np.zeros([n_cluster, c, w, h], dtype=np.float32)
        for step in range(args.gradient_step):
            #try each cluster
            grad_mask = img_grad * patches
            cluster_imgs = cluster_imgs + grad_mask
            fdict = {image_input: cluster_imgs, keep_prob: 1.0, is_train: False, positive: positive_embedding}
            grads_np, embedding_np, loss_np = session.run([grad_input, embedding, loss], feed_dict=fdict)
            moments = moments * moment_val + grads_np * (1. - moment_val)
            img_grad = step_val * np.sign(moments)
            loss_np = -1 * loss_np

            print('step %d: %s '%(step, loss_np))

            # test whether success
            for i in range(n_cluster):

                if loss_np[i] < 0.2:
                    success = True
                    success_cluster_no.append(i)
                    success_loss.append(loss_np[i])
                    success_img = cluster_imgs[i,:,:,:].transpose(1,2,0)
                    output_img = np.round((success_img + 1) * 255 / 2).astype(np.uint8)
                    filename = adv_filename + '_cluster_%d.jpg'%(i)
                    io.imsave(filename, output_img)
            if success:
                break
        r=r+1
    write_csv(success_cluster_no, success_loss, adv_filename+'_r%d.csv'%(r-1))

def get_next_generation(adv_imgs, patch_list, losses, max_num):
    """
    caculate the adversarial images for next generation
    :param adv_imgs: adversarial images. ndarry
    :param patch_list: patch list
    :param losses: loss, ndarray
    :param max_num: max number of candidates
    :return adv image list, patch_list
    """
    adv_imgs_list, loss_list, patch_list = sort_by_loss(adv_imgs,losses,patch_list)
    index = int(len(loss_list) * 0.2) * -1
    prob_list = cacu_probability(losses[:index])
    adv_img_list ,new_patch_list = extend_max(adv_imgs_list[:index], patch_list[:index], prob_list, max_num)
    return adv_img_list, new_patch_list


def remove_overlap(patch_list, losses, adv_imgs):
    """
    remove overlap patches and sort by loss
    :param patch_list: list of patches
    :param losses: loss of adversarial images with these patches, ndarray
    :param adv_imgs: adversarial images, ndarray
    :return: patch list sorted by loss
    """
    adv_img_list, losses_list, patch_list = sort_by_loss(adv_imgs, losses, patch_list)

    removed = []
    for i in range(len(patch_list)):
        if i not in removed:
            for j in range(i+1, len(patch_list)):
                if j not in removed:
                    sum  = np.sum(np.multiply(patch_list[i], patch_list[j]))
                    if sum != 0:
                        removed.append(j)

    removed.sort(reverse=True)
    for k in removed:
        patch_list.pop(k)
        losses_list.pop(k)
        adv_img_list.pop(k)

    prob_list = cacu_probability(losses_list)
    return patch_list, prob_list, adv_img_list


def sort_by_loss(adv_imgs, losses, patch_list):
    """
    sort adv_imgs and patch_list by losses
    :param adv_imgs: adversarial images, ndarray
    :param losses:  losses, ndarray
    :param patch_list: patch list
    :return: sorted adv image list , loss list and patch list
    """
    adv_img_list = adv_imgs.tolist()
    losses_list = losses.tolist()
    # buble sort
    for j in range(len(losses_list) - 1):
        for i in range(len(losses_list) - j - 1):
            if losses_list[i] < losses_list[i + 1]:
                temp = losses_list[i + 1]
                losses_list[i + 1] = losses_list[i]
                losses_list[i] = temp
                temp = patch_list[i + 1]
                patch_list[i + 1] = patch_list[i]
                patch_list[i] = temp
                temp = adv_img_list[i + 1]
                adv_img_list[i + 1] = adv_img_list[i]
                adv_img_list[i] = temp
    return adv_img_list, losses_list, patch_list


def cacu_probability(losses_list):
    """
    caculate the probabilties for roulette
    :param loss_np: ndarray of losses
    :return: probability list
    """
    sum = 0
    for loss in losses_list:
        sum += 1 + loss

    # roulette
    prob_list = []
    prob = 0
    for loss in losses_list:
        prob = prob + (1+loss) / sum
        prob_list.append(prob)
    return prob_list


def extend_max(adv_img_list, patch_list, prob_list, max_num):
    """
    extend adv imgs to max_cluster
    :param adv_img_list: adversarial images list
    :param patch_list: patch of adversarial image
    :param prob_list: the probablity list of these images
    :param max_num: maximum number of candidates
    :return: adversarial image list and patch list of maximum number
    """
    while len(adv_img_list) < max_num:
        index = roulette(prob_list)
        adv_img_list.append(adv_img_list[index])
        patch_list.append(patch_list[index])
    return adv_img_list, patch_list

def roulette(prob_list):
    """
    return the index from prod_list with roulette
    :param prob_list: probability list
    :return: index
    """
    r = random.random()
    for i in range(len(prob_list)):
        if r<=prob_list[i]:
            return i


def crossover(patch_list, patch_list_0, prob_0):
    """
    cross over
    :param patch_list:
    :param patch_list_0:
    :param prob_0:
    :return: the patch list for next generation
    """
    new_patch_list = []
    patch_exist = False
    for patch in patch_list:
        patch_num = len(patch_list_0)
        while patch_num > -1 * patch_num:
            index = roulette(prob_0)
            if np.sum(np.multiply(patch, patch_list_0[index])) == 0:
                #no overlap
                cross_patch = np.add(patch, patch_list_0[index])
                break
            patch_num -=1
        if patch_num > -1 * patch_num:
            new_patch_list.append(cross_patch)
            patch_exist = True
        else:
            new_patch_list.append(np.zeros_like(patch))
    if patch_exist:
        return new_patch_list
    else:
        return []

def create_patch(size, clusters, img, n_cluster):
    """
    prepare data for 0 generation
    :param size: size of patch
    :param clusters: list of (label, weight, center, cluster_mask)
    :param img: input image
    :param n_cluster: number of clusters
    :return: patch masks for each cluster
    """
    # create patches for all clusters
    patch_list = []
    for i in range(n_cluster):
        _, _, center, _ = clusters[i]
        patch_mask = get_patch(img, center, radius=size)
        if np.sum(patch_mask) == (2 * size + 1) * (2 * size + 1):
            patch_list.append(patch_mask)
    return patch_list


def generation(args, adv_imgs_np, patches_np,  positive_embedding, model_dic, session):
    """
    one generation
    :param args: command args
    :param adv_imgs: adversarial images
    :param patches: patches for these images
    :param positive_embedding: positive embedding
    :param model_dics: model tensors
    :param session: session
    :return:
    """
    moments_np = np.zeros_like(adv_imgs_np)

    for step in range(args.gradient_step):
        # try each cluster
        fdict = {model_dic['image_input']: adv_imgs_np,
                 model_dic['keep_prob']: 1.0,
                 model_dic['is_train']: False,
                 model_dic['positive_embedding']: positive_embedding,
                 model_dic['patches']: patches_np,
                 model_dic['moments']: moments_np}
        adv_imgs_np, loss_np, moments_np = session.run([model_dic['adv_imgs'], model_dic['loss'], model_dic['new_moments']], feed_dict=fdict)

    return loss_np, adv_imgs_np

def multiPatchAttack(args, img, clusters, session, positive_embedding, adv_filename, model_dic):
    """
    Multiply Patches
    :param args: command arguments
    :param img: the image
    :param clusters: gradient clusters (label, weight, center, cluster_mask)
    :param session: session
    :param positive_embedding: positive embedding
    :param adv_filename: the adversarial image name
    :param model_dic: model tensors
    :return:
    """
    img = img.transpose(2,0,1)
    n_cluster = min(len(clusters),args.max_cluster)
    if n_cluster == 0:
        print('%s: no cluster '%adv_filename)
        return

    patch_list = create_patch(args.size, clusters, img, n_cluster)
    patches = np.stack(patch_list)
    adv_imgs = np.stack([img for i in range(len(patch_list))])
    losses, adv_imgs = generation(args, adv_imgs, patches, positive_embedding, model_dic, session)
    num_patch = 1
    print('==============%s===================='%(adv_filename))
    print('------------------%d----------------------'%(num_patch))
    print('%s ' % ( -1 * losses))
    patch_list_0, prob_0, adv_img_list = remove_overlap(patch_list, losses, adv_imgs)
    adv_img_list, patch_list = extend_max(adv_img_list, patch_list_0, prob_0, args.max_cluster)
    adv_imgs = np.stack(adv_img_list)
    success_loss = []
    name_list = []
    success = False
    while not success and num_patch < 20:
        new_patch_list = crossover(patch_list, patch_list_0, prob_0)
        if len(new_patch_list) == 0:
            # no new patches
            break

        patches = np.stack(new_patch_list)
        losses, new_adv_imgs = generation(args, adv_imgs, patches, positive_embedding, model_dic, session)
        num_patch+=1
        print('------------------%d----------------------'%(num_patch))
        print('%s ' % (-1 * losses))
        # test whether success
        for i in range(args.max_cluster):
            loss = -1 * losses[i]
            if  loss< 0.2:
                success = True
                success_loss.append(loss)
                success_img = adv_imgs[i, :, :, :].transpose(1, 2, 0)
                output_img = np.round((success_img + 1) * 255 / 2).astype(np.uint8)
                filename = adv_filename + '_cluster_%d_%d.jpg' % (i,num_patch)
                name_list.append(filename)
                io.imsave(filename, output_img)

        adv_img_list, patch_list = get_next_generation(new_adv_imgs, patch_list, losses, args.max_cluster)
        adv_imgs = np.stack(adv_img_list)

    if success:
        write_csv(name_list, success_loss, adv_filename + '_r%d.csv' % (args.size))

def find_adv_image(args, grad_input, dir, img, name_list, session, positive_embedding, model_dic):
    """
    find adversarial images
    :param args: command args
    :param grad_input: gradients to input image
    :param dir: output directory
    :param img:
    :param name_list:
    :param session:
    :param positive_embedding:
    :param model_dic:
    """
    new_img = img.transpose([0, 2, 3, 1])
    for i in range(1, 20):
        threshold = i / 20.0
        filter = np.where(grad_input > threshold, 1, 0)
        percentage = np.mean(filter)
        if percentage > args.max_percent:
            continue
        if percentage < args.min_percent:
            break

        if not os.path.exists(dir):
            os.makedirs(dir)

        expand = np.zeros_like(filter)
        exp_filter = np.stack([expand, filter, expand], axis=3)
        img_mask = np.where(exp_filter == 1, exp_filter, new_img)
        for j in range(new_img.shape[0]):
            output_img =np.round((img_mask[j,:,:,:] + 1) * 255 / 2).astype(np.uint8)
            filename = os.path.join(dir, '%s_thre_%d.jpg' % (name_list[j], i))
            io.imsave(filename, output_img)
            if args.cluster_method == 'dbscan':
                cluster_list = dbscan(args.eps, args.min_samples, filter[j,:,:], grad_input[j,:,:])
            elif args.cluster_method == 'kmeans':
                cluster_list = kmeans(args.max_cluster, filter[j, :, :], grad_input[j, :, :])
            else:
                cluster_list = randomcluster(args.max_cluster, grad_input[j, :, :])

            #ouput top 10
            index = 0
            cluster_names = []
            cluster_height = []
            cluster_color = []
            center_list = []
            cluster_img = np.round((new_img[j, :, :, :] + 1) * 255 / 2).astype(np.uint8)
            for _, weight, center_point, cluster_mask in cluster_list:
                if index >(len(matplot_colors) - 1):
                    break

                center_list.append(center_point)
                cluster_names.append('(%d,%d)'%(center_point[0],center_point[1]))
                cluster_height.append(weight)
                cluster_color.append(matplot_colors[index])

                b_channel = np.ones(cluster_mask.shape, dtype=np.uint8) * colors[index][0]
                g_channel = np.ones(cluster_mask.shape, dtype=np.uint8) * colors[index][1]
                r_channel = np.ones(cluster_mask.shape, dtype=np.uint8) * colors[index][2]
                color = np.stack([b_channel,g_channel,r_channel],axis=2)
                cluster_mask3 = np.stack([cluster_mask,cluster_mask,cluster_mask], axis=2)
                cluster_img = np.where(cluster_mask3 == 1, color, cluster_img)

                #center point
                cluster_img[center_point[0],center_point[1],:] = [255,255,255]
                index = index + 1

            filename = os.path.join(dir, '%s_thre_%d_cluster.jpg' % (name_list[j], i))
            io.imsave(filename, cluster_img)

            plt.clf()
            plt.barh(range(len(cluster_height)), cluster_height, color=cluster_color)
            plt.yticks(range(len(cluster_names)), cluster_names)
            plt.xlabel('Weight')
            plt.title('Weight of Clusters in the Image')
            plt.savefig(os.path.join(dir, '%s_thre_%d_cluster_fig.jpg' % (name_list[j], i)))

            adv_filename = os.path.join(dir, '%s_thre_%d' % (name_list[j], i))
            #singlePatchAttack(args, new_img[j,:,:,:], cluster_list, session, positive_embedding, adv_filename, grad_input, positive, loss)
            multiPatchAttack(args,new_img[j,:,:,:], cluster_list, session, positive_embedding, adv_filename, model_dic)

def load_model(args):
    """
    load model from file
    :param args: command args
    :return: model dictionary and session
    """
    sess = tf.Session()
    # Embedding model
    with tf.gfile.GFile(args.model, "rb") as f:
        graph_def = tf.GraphDef()  # 新建GraphDef文件，用于临时载入模型中的图
        graph_def.ParseFromString(f.read())  # GraphDef加载模型中的图--据说可能失败，因为没有保存variable
    tf.import_graph_def(graph_def,  # 在当前默认图中加载GraphDef中的图
                        input_map=None,
                        return_elements=None,
                        name="")
    writer = tf.summary.FileWriter("log")
    writer.add_graph(sess.graph)

    moment_val = 0.9
    step_val = 1.0 / args.gradient_step

    image_input = tf.get_default_graph().get_tensor_by_name('image_input:0')  # (?, 3, 112, 112)
    keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
    is_train = tf.get_default_graph().get_tensor_by_name('training_mode:0')
    embedding = tf.get_default_graph().get_tensor_by_name('embedding:0')  # (?, 512)
    stage4_unit3_conv2 = tf.get_default_graph().get_tensor_by_name('stage4_unit3_conv2:0')  # (?, 512, 7, 7)
    positive = tf.placeholder(tf.float32, shape=[512], name='positive_embedding')
    patches = tf.placeholder(tf.int32, shape=[None,1,112,112], name='patches')
    moments = tf.placeholder(tf.float32, shape=[None,3,112,112], name='moments')

    loss = tf.math.multiply(tf.reduce_sum(tf.multiply(embedding, positive), axis=1, name='loss'), -1)  # (?,)
    grad_input = tf.gradients(loss, image_input, name="grad_input")[0]
    new_moments = tf.math.add(tf.math.multiply(moments ,moment_val), tf.math.multiply(grad_input, (1. - moment_val)))
    img_grad = tf.math.multiply(tf.sign(new_moments), step_val)
    grad_mask = tf.multiply(img_grad, tf.cast(patches, dtype=tf.float32), name='grad_mask')
    adv_imgs = tf.add(image_input, grad_mask, name='adv_imgs')
    grad_conv = tf.gradients(loss, stage4_unit3_conv2)[0]


    model_dic= {'image_input':image_input,
                'keep_prob':keep_prob,
                'is_train':is_train,
                'embedding':embedding,
                'stage4_unit3_conv2':stage4_unit3_conv2,
                'positive_embedding':positive,
                'loss':loss,
                'grad_input':grad_input,
                'grad_conv':grad_conv,
                'patches':patches,
                'adv_imgs':adv_imgs,
                'moments':moments,
                'new_moments':new_moments}
    return model_dic, sess


def main(args):
    tf.disable_eager_execution()

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)

    model_dic, sess = load_model(args)

    dirs = os.listdir(args.input_path)
    for folder in dirs:
        path = os.path.join(args.input_path, folder)
        if os.path.isdir(path):
            output_path = os.path.join(args.output_path, folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            print('========== input %s ============'%(folder))
            positive_img_names = tf.gfile.Glob(os.path.join(path, '*_0001.jpg'))  # positive image
            print('Positive: %s' % (os.path.basename(positive_img_names[0])))
            #get positive image
            img_positive = io.imread(positive_img_names[0]) / 255.
            img_positive = rescale(img_positive, 112./600., order=5, multichannel=True)
            img_positive = prep(img_positive)#[1,3,112,112]
            fdict = {model_dic['image_input']: img_positive, model_dic['keep_prob']: 1.0, model_dic['is_train']: False}
            pos_embedding_np = sess.run(model_dic['embedding'], feed_dict=fdict)  # [2,512]

            name_list = []
            img_list = []
            img_names = tf.gfile.Glob(os.path.join(path, '*.jpg'))  #tf.gfile.Glob得到的是一个list
            for file in img_names:
                if file != positive_img_names[0]:
                    print('%s' % (os.path.basename(file)))
                    img = prep(rescale(io.imread(file) / 255., 112./600., order=5, multichannel=True))#[1,3,112,112]
                    name_list.append(os.path.basename(file)[:-4])
                    img_list.append(img)

            feed_img = np.squeeze(np.array(img_list), axis=1)  #[?,3,112,112]
            fdict1 = {model_dic['image_input']: feed_img,
                     model_dic['keep_prob']: 1.0,
                     model_dic['is_train']: False,
                     model_dic['positive_embedding']:pos_embedding_np[0]}
            last_conv_np, grad_conv_np, grad_input_np, loss_np = sess.run([model_dic['stage4_unit3_conv2'],
                                                                                          model_dic['grad_conv'],
                                                                                          model_dic['grad_input'],
                                                                                          model_dic['loss']],
                                                                                         feed_dict=fdict1)

            #add neg and pos embedings
            csv_file_name = os.path.join(output_path, 'distance.csv')
            write_csv(name_list,loss_np * (-1),csv_file_name)

            #cam, norm_grad_input_0_1:[112,112]
            norm_grad_input = cacu_norm_grad(grad_input_np)
            cam, norm_grads_conv = calcu_cam(last_conv_np, grad_conv_np)

            cam_dir =  os.path.join(output_path, 'cam')
            grad_dir = os.path.join(output_path, 'grad')

            # Filter with threshold
            find_adv_image(args, norm_grad_input, grad_dir, feed_img, name_list, sess ,pos_embedding_np[0], model_dic)
            #find_adv_image(args, cam, cam_dir, feed_img, name_list, sess ,pos_embedding_np[0], model_dic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='data/test',
                        type=str, help='Path to the image.' )  # p6;先用[250，250]的维度来进行操作
    parser.add_argument('--output_path', default='data/test_dbscan_output',
                        type=str, help='Path to the image.')
    parser.add_argument('--eps', type=int, default=2, help="eps for dbscan of the area")
    parser.add_argument('--min_samples', type=int, default=6, help="min samples for dbscan")
    parser.add_argument('--model', type=str, default='models/r100.pb',
                        help="model name to look, arcface.pb_weizhi or insightface.pb")
    parser.add_argument('--max_percent', type=float, default=0.5,
                        help="maximum percentage of points")
    parser.add_argument('--min_percent', type=float, default=0.01,
                        help="minimum percentage of points")
    parser.add_argument('--gradient_step', type=int, default=51,
                        help="gradient steps")
    parser.add_argument('--max_cluster', type=int, default=50,
                        help="maximum clusters allowed")
    parser.add_argument('--cluster_method', default='dbscan',
                        help="kmeans or dbscan")
    parser.add_argument('--size', type=int, default=1,
                        help="the size of patches would be 2*size+1")


    args = parser.parse_args()

    main(args)

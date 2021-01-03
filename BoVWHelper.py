import numpy as np
import cv2
import os

from scipy.spatial import distance
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AffinityPropagation
from random import randint
from matplotlib import pyplot as plt
from tqdm import tqdm


def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if i == 0:
            count = distance.euclidean(image, center[i])
            # count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            # dist = L1_dist(image, center[i])
            if dist < count:
                ind = i
                count = dist
    return ind


def load_images_from_folder(folder):
    images = {}
    for filename in tqdm(os.listdir(folder)):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat, 0)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category

    print("Images Loaded...")
    return images


def random_color_generator(number_of_color):
    list_color = []
    for val in range(number_of_color):
        color = [randint(0, 255), randint(0, 255), randint(0, 255)]
        list_color.append(color)
    return list_color


# ---------------------------------------------------------------


def sift_features(images):
    sift_vectors = {}
    descriptor_list = []

    sift = cv2.xfeatures2d.SIFT_create()
    for key, value in tqdm(images.items()):
        features = []
        for img in value:
            kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                descriptor_list.extend(des)
                features.append(des)
        sift_vectors[key] = features
    print("Features extracted...")
    return [descriptor_list, sift_vectors, kp]


def surf_features(images):
    surf_features = {}
    descriptor_list = []

    surf = cv2.xfeatures2d.SURF_create()
    for key, value in images.items():
        features = []
        for img in value:
            kp, des = surf.detectAndCompute(img, None)
            if des is not None:
                descriptor_list.extend(des)
                features.append(des)
        surf_features[key] = features

    return [descriptor_list, surf_features, kp]


def orb_features(images):
    orb_features = {}
    descriptor_list = []

    orb = cv2.ORB_create()
    for key, value in images.items():
        features = []
        for img in value:
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                descriptor_list.extend(des)
                features.append(des)
        orb_features[key] = features

    return [descriptor_list, orb_features, kp]


def classification_of_kp(key_point, descriptor):
    # Get position of keypoints
    pos_kp = []
    for pos in key_point:
        pos_kp.append(pos.pt)

    # Scale keypoints for better precision
    pos_kp_scale = StandardScaler().fit_transform(pos_kp)

    # Apply DBSCAN algorithm for pre classifying of cells

    # DBSCAN algorithm gives label for every group of feature
    db_scan = DBSCAN(eps=0.1, min_samples=6).fit(pos_kp_scale)
    labels = db_scan.labels_
    print(labels)
    # Appending location and label of every cell or noise to the dictionary structure
    cluster_label = {}
    cluster_descriptor = {}
    random_cell =[]

    _index = 0
    for x in labels:
        if x != -1:
            if x not in cluster_label:
                cluster_label[x] = []
                cluster_label[x].append(pos_kp[_index])
                cluster_descriptor[x] = []
                cluster_descriptor[x].append(descriptor[_index])
            else:
                cluster_label[x].append(pos_kp[_index])
                cluster_descriptor[x] = np.vstack((cluster_descriptor[x], descriptor[_index]))

        _index += 1

    for x in cluster_descriptor.values():
        random_cell.append(x)

    cluster_descriptor.clear()
    cluster_descriptor['random'] = random_cell

    # Number of clustered feature, this is just for checking label count.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Features clustered to ", n_clusters_, " labels")

    return cluster_label, cluster_descriptor, n_clusters_


def k_means_descriptor(k, descriptor_list):
    print("K Means starting...")
    k_means = KMeans(n_clusters=k, n_init=10)
    k_means.fit(descriptor_list)
    visual_words = k_means.cluster_centers_
    print("Visual words are ready.")
    return visual_words


def mean_shift_descriptor(descriptor_list):
    # bandwidth = estimate_bandwidth(descriptor_list, quantile=0.2)
    print("Mean Shift starting...")
    mean_shift = MeanShift(bandwidth=2)
    mean_shift.fit(descriptor_list)
    visual_words = mean_shift.cluster_centers_
    print("Visual words are ready.")
    return visual_words


def affinity_descriptor(descriptor_list):
    print("Affinity Propagation starting...")
    af = AffinityPropagation()
    af.fit(descriptor_list)
    visual_words = af.cluster_centers_
    print("Visual words are ready.")
    return visual_words


def image_class(all_bovw, centers):
    print("---------- Histogram Stage ----------")
    dict_feature = {}
    for key, value in tqdm(all_bovw.items()):
        category = []
        for img in tqdm(value):
            histogram = np.zeros(len(centers))
            for each_feature in img:
                if each_feature is not None:
                    ind = find_index(each_feature, centers)
                    histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category

    print("====================")
    return dict_feature


def knn(images, tests):
    print("---------- Classification Of Test Data ----------")
    num_test = 0
    img_key = []
    correct_predict = 0
    class_based = {}

    for test_key, test_val in tests.items():
        class_based[test_key] = [0, 0]  # [correct, all]
        for tst in test_val:
            predict_start = 0
            minimum = 0
            key = "a"  # predicted
            for train_key, train_val in images.items():
                for train in train_val:
                    if predict_start == 0:
                        minimum = distance.euclidean(tst, train)
                        # minimum = L1_dist(tst,train)
                        key = train_key
                        # img_key.append(key)
                        predict_start += 1
                    else:
                        dist = distance.euclidean(tst, train)
                        # dist = L1_dist(tst,train)
                        if dist < minimum:
                            minimum = dist
                            key = train_key
                            # img_key.append(key)

            print("Value: ", key, " Distance: ", minimum, "True Value: ", test_key)
            img_key.append(key)
            if test_key == key:
                correct_predict += 1
                class_based[test_key][0] += 1
            num_test += 1
            class_based[test_key][1] += 1

    return [img_key, (num_test, correct_predict, class_based)]


def accuracy(results):
    avg_accuracy = (results[1] / results[0]) * 100
    print("Average accuracy: %" + str(avg_accuracy))
    print("\nClass based accuracies: \n")
    for key, value in results[2].items():
        if value[1] != 0:
            acc = (value[0] / value[1]) * 100
            print(key + " : %" + str(acc))


def mark_cells(img, cluster_label, label_size, keys):
    color_list = random_color_generator(label_size + 1)

    cell_images = []
    cell_images_all_cluster = img.copy()
    cell_images_all_marks = img.copy()

    for _ in range(label_size + 1):
        cell_images.append(img.copy())

    print(len(cell_images))

    for idx, x in enumerate(cluster_label):
        pos_x_list = []
        pos_y_list = []

        for y in cluster_label[x]:
            position = (int(y[0]), int(y[1]))
            print("Pos: ", position)
            pos_x_list.append(int(y[0]))
            pos_y_list.append(int(y[1]))

            # cv2.circle(cell_images[idx], position, 3, color_list[idx], -1)
            cv2.circle(cell_images_all_cluster, position, 3, color_list[idx], -1)

        print("Cell Type: ", keys[idx])
        pos_x_min = min(pos_x_list) - 5
        pos_x_max = max(pos_x_list) + 5
        pos_y_min = min(pos_y_list) - 5
        pos_y_max = max(pos_y_list) + 5

        print(pos_x_max, pos_x_min, pos_y_max, pos_y_min)
        cv2.rectangle(cell_images[idx], (pos_x_min, pos_y_min), (pos_x_max, pos_y_max), (0, 255, 0), 2)
        cv2.putText(cell_images[idx], keys[idx], (pos_x_min, pos_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(cell_images[idx], keys[idx], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("", cell_images[idx])
        # plt.imsave(str(idx) + ".png", cell_images[idx])

        cv2.rectangle(cell_images_all_marks, (pos_x_min, pos_y_min), (pos_x_max, pos_y_max), (0, 255, 0), 2)
        cv2.putText(cell_images_all_marks, keys[idx], (pos_x_min, pos_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        cv2.waitKey(0)

    cv2.imshow("All Clusters", cell_images_all_cluster)
    cv2.waitKey(0)
    cv2.imshow("All Clusters", cell_images_all_marks)
    cv2.waitKey(0)

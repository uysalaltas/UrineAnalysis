import cv2
import BoVWHelper as bovw
import numpy as np
import pickle


def bag_of_visual_words_single_image():
    images = bovw.load_images_from_folder('train')
    test = bovw.load_images_from_folder('single test')

    img = cv2.imread('single test/random/nh03100.jpg')

    sifts = bovw.sift_features(images)
    descriptor_list = sifts[0]
    all_bovw_feature = sifts[1]

    test_sifts = bovw.sift_features(test)

    cluster_cell_pos, cluster_cell_des, n_label = bovw.classification_of_kp(test_sifts[2], test_sifts[0])

    # Takes the central points which is visual words
    visual_words = bovw.k_means_descriptor(150, descriptor_list)

    # Creates histograms for train data
    bovw_train = bovw.image_class(all_bovw_feature, visual_words)
    # Creates histograms for test data
    bovw_test = bovw.image_class(cluster_cell_des, visual_words)

    cell_keys, results_bowl = bovw.knn(bovw_train, bovw_test)
    print(results_bowl)

    bovw.mark_cells(img, cluster_cell_pos, n_label, cell_keys)


def bag_of_visual_words_cropped():
    images = bovw.load_images_from_folder('train')
    test = bovw.load_images_from_folder('test')

    sifts = bovw.sift_features(images)
    descriptor_list = sifts[0]
    all_bovw_feature = sifts[1]

    test_sifts = bovw.sift_features(test)
    test_bovw_feature = test_sifts[1]

    # Takes the central points which is visual words
    visual_words = bovw.k_means_descriptor(150, descriptor_list)

    # Creates histograms for train data
    bovw_train = bovw.image_class(all_bovw_feature, visual_words)
    # Creates histograms for test data
    bovw_test = bovw.image_class(test_bovw_feature, visual_words)

    cell_keys, results_bowl = bovw.knn(bovw_train, bovw_test)
    print(results_bowl)

    bovw.accuracy(results_bowl)


# -------------------------------------------------


def bovw_train_and_save():
    images = bovw.load_images_from_folder('train')

    sifts = bovw.sift_features(images)
    descriptor_list = sifts[0]
    all_bovw_feature = sifts[1]

    visual_words = bovw.k_means_descriptor(150, descriptor_list)
    bovw_train = bovw.image_class(all_bovw_feature, visual_words)
    #
    # output = open('train.pkl', 'wb')
    # pickle.dump(bovw_train, output)
    # output.close()
    #
    # np.savetxt('visual_words.csv', visual_words, delimiter=',')

    return bovw_train


def bovw_read_train_data():
    pkl_file = open('train.pkl', 'rb')
    train = pickle.load(pkl_file)
    pkl_file.close()

    visual_words = np.loadtxt('visual_words.csv', delimiter=',')

    return visual_words, train


def bovw_train_test(visual_words, train_bovw):
    test = bovw.load_images_from_folder('test')
    test_sifts = bovw.sift_features(test)
    test_bovw_feature = test_sifts[1]

    test_bovw = bovw.image_class(test_bovw_feature, visual_words)
    cell_keys, results_bowl = bovw.knn(train_bovw, test_bovw)
    print(results_bowl)

    bovw.accuracy(results_bowl)


def bovw_single_test(visual_words, train_bovw):
    test = bovw.load_images_from_folder('single test')
    img = cv2.imread('single test/random/nh00006.jpg')

    test_sifts = bovw.sift_features(test)
    cluster_cell_pos, cluster_cell_des, n_label = bovw.classification_of_kp(test_sifts[2], test_sifts[0])

    # for key, val in cluster_cell_des.items():
    #     for x in val:
    #         print(x)

    test_bovw = bovw.image_class(cluster_cell_des, visual_words)

    cell_keys, results_bowl = bovw.knn(train_bovw, test_bovw)
    # print(results_bowl)

    bovw.mark_cells(img, cluster_cell_pos, n_label, cell_keys)


def bowv_single_cell_test(visual_words, train_bovw):
    test = bovw.load_images_from_folder('single test')
    test_sifts = bovw.sift_features(test)
    test_bovw_feature = test_sifts[1]

    test_bovw = bovw.image_class(test_bovw_feature, visual_words)

    cell_keys, results_bowl = bovw.knn(train_bovw, test_bovw)


if __name__ == '__main__':
    # bovw_train_and_save()
    visual_words, train_bovw = bovw_read_train_data()
    # bovw_single_test(visual_words, train_bovw)
    bovw_train_test(visual_words, train_bovw)
    # bowv_single_cell_test(visual_words, train_bovw)

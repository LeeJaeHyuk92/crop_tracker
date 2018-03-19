# train file

import time
import tensorflow as tf
import os
import numpy as np
import goturn_net
import setproctitle
import cv2
from logger.logger import setup_logger
from loader.loader_imagenet import loader_imagenet
from loader.loader_alov import loader_alov
from helper.config import DEBUG, POLICY
from example_generator import example_generator


setproctitle.setproctitle('TRAIN_TRACKER_IMAGENET_ALOV')
logger = setup_logger(logfile=None)

NUM_EPOCHS = POLICY['NUM_EPOCHS']
BATCH_SIZE = POLICY['BATCH_SIZE']
WIDTH = POLICY['WIDTH']
HEIGHT = POLICY['HEIGHT']
pretraind_model = POLICY['pretrained_model']
kGeneratedExamplesPerImage = POLICY['kGeneratedExamplesPerImage']
logfile = POLICY['logfile']
train_txt = "test_set.txt"


run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True


def train_image(image_loader, images):
    """TODO: Docstring for train_image.
    """
    curr_image = np.random.randint(0, len(images))
    list_annotations = images[curr_image]
    curr_ann = np.random.randint(0, len(list_annotations))

    image, bbox = image_loader.load_annotation(curr_image, curr_ann)

    return image, bbox
    # tracker_trainer.train(image, image, bbox, bbox)


def train_video(videos):
    """TODO: Docstring for train_video.
    """
    video_num = np.random.randint(0, len(videos))
    video = videos[video_num]

    annotations = video.annotations

    if len(annotations) < 2:
        logger.info('Error - video {} has only {} annotations', video.video_path, len(annotations))

    ann_index = np.random.randint(0, len(annotations) - 1)
    frame_num_prev, image_prev, bbox_prev = video.load_annotation(ann_index)

    frame_num_curr, image_curr, bbox_curr = video.load_annotation(ann_index + 1)

    return image_prev, image_curr, bbox_prev, bbox_curr
    # tracker_trainer.train(image_prev, image_curr, bbox_prev, bbox_curr)

def load_training_set(train_file):
    '''
    return train_set
    '''

    ftrain = open(train_file, "r")
    trainlines = ftrain.read().splitlines()
    train_target = []
    train_search = []
    train_box = []
    for line in trainlines:
        line = line.split(",")
        train_target.append(line[0])
        train_search.append(line[1])
        box = [10 * float(line[2]), 10 * float(line[3]), 10 * float(line[4]), 10 * float(line[5])]
        train_box.append(box)
    ftrain.close()

    # total img path, search, trainbox
    return [train_target, train_search, train_box]


def data_reader(objLoaderImgNet, train_imagenet_images, train_alov_videos):
    '''
    this function only read the one pair of images and from the queue
    '''

    # # thanks for https://github.com/nrupatunga/PY-GOTURN
    # logger.info('Loading training data')
    # # TODO, Load imagenet training images and annotations
    # imagenet_folder = os.path.join(POLICY['imagenet'], 'images')
    # imagenet_annotations_folder = os.path.join(POLICY['imagenet'], 'gt')
    # objLoaderImgNet = loader_imagenet(imagenet_folder, imagenet_annotations_folder, logger)
    # train_imagenet_images = objLoaderImgNet.loaderImageNetDet()
    #
    # # Load alov training images and annotations
    # alov_folder = os.path.join(POLICY['alov'], 'images')
    # alov_annotations_folder = os.path.join(POLICY['alov'], 'gt')
    # objLoaderAlov = loader_alov(alov_folder, alov_annotations_folder, logger)
    # objLoaderAlov.loaderAlov()
    # train_alov_videos = objLoaderAlov.get_videos()

    # create example generator and setup the network
    for idx in xrange(BATCH_SIZE):
        objExampleGen = example_generator(float(POLICY['lamda_shift']), float(POLICY['lamda_scale']),
                                          float(POLICY['min_scale']), float(POLICY['max_scale']), logger)

        images = []
        targets = []
        bbox_gt_scaleds = []

        random_bool = np.random.randint(0, 2)
        if random_bool == 0:
            image, bbox = train_image(objLoaderImgNet, train_imagenet_images)
            objExampleGen.reset(bbox, bbox, image, image)
            # Generate more number of examples
            images, targets, bbox_gt_scaleds = objExampleGen.make_training_examples(
                kGeneratedExamplesPerImage, images, targets, bbox_gt_scaleds)
        else:
            img_prev, img_curr, bbox_prev, bbox_curr = train_video(train_alov_videos)
            objExampleGen.reset(bbox_prev, bbox_curr, img_prev, img_curr)
            image, target, bbox_gt_scaled = objExampleGen.make_true_example()
            images.append(image)
            targets.append(target)
            bbox_gt_scaleds.append(bbox_gt_scaled)

        del(objExampleGen)

        # data generator
        # example_generator.reset(bbox_prev, bbox_curr, img_prev, img_curr)

        search_tensor = images[0].astype(np.float32)
        search_tensor = cv2.resize(search_tensor, (HEIGHT, WIDTH), interpolation=cv2.INTER_LINEAR)
        target_tensor = targets[0].astype(np.float32)
        target_tensor = cv2.resize(target_tensor, (HEIGHT, WIDTH), interpolation=cv2.INTER_LINEAR)
        # box_tensor = input_queue[2]
        box_tensor = np.array([bbox_gt_scaleds[0].x1, bbox_gt_scaleds[0].y1, bbox_gt_scaleds[0].x2, bbox_gt_scaleds[0].y2], dtype=np.float32)

        if not idx == 0:
            ep_search_tensor = np.concatenate([ep_search_tensor, np.expand_dims(search_tensor, axis=0)], axis=0)
            ep_target_tensor = np.concatenate([ep_target_tensor, np.expand_dims(target_tensor, axis=0)], axis=0)
            ep_box_tensor = np.concatenate([ep_box_tensor, np.expand_dims(box_tensor, axis=0)], axis=0)
        else:
            ep_search_tensor = np.expand_dims(search_tensor, axis=0)
            ep_target_tensor = np.expand_dims(target_tensor, axis=0)
            ep_box_tensor = np.expand_dims(box_tensor, axis=0)

    if DEBUG:
        idx = 0
        for idx in xrange(len(images)):
            H, W, C = images[idx].shape
            cv2.imwrite(str(idx) + '_' + 'target.jpg', targets[idx])
            x1 = int(bbox_gt_scaleds[idx].x1 / 10 * W)
            y1 = int(bbox_gt_scaleds[idx].y1 / 10 * H)
            x2 = int(bbox_gt_scaleds[idx].x2 / 10 * W)
            y2 = int(bbox_gt_scaleds[idx].y2 / 10 * H)
            images_gt = cv2.rectangle(images[idx], (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.imwrite(str(idx) + '_' + 'image_gt.jpg', images_gt)
            # logger.info(str(idx) + '_'+ 'image.jpg s bbox_gt_scaled is:')
            # logger.info('x1 : ' + str(bbox_gt_scaleds[idx].x1))
            # logger.info('y1 : ' + str(bbox_gt_scaleds[idx].y1))
            # logger.info('x2 : ' + str(bbox_gt_scaleds[idx].x2))
            # logger.info('y2 : ' + str(bbox_gt_scaleds[idx].y2))


    return [ep_search_tensor, ep_target_tensor, ep_box_tensor]


def next_batch(objLoaderImgNet, train_imagenet_images, train_alov_videos):
    min_queue_examples = 128
    num_threads = 8
    [search_tensor, target_tensor, box_tensor] = data_reader(objLoaderImgNet, train_imagenet_images, train_alov_videos)
    [search_batch, target_batch, box_batch] = tf.train.shuffle_batch(
        [search_tensor, target_tensor, box_tensor],
        batch_size=1,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads + 2) * BATCH_SIZE,
        seed=88,
        min_after_dequeue=min_queue_examples)
    return [search_batch, target_batch, box_batch]


if __name__ == "__main__":
    if (os.path.isfile(logfile)):
        os.remove(logfile)

    # [train_target, train_search, train_box] = load_training_set(train_txt)
    # target_tensors = tf.convert_to_tensor(train_target, dtype=tf.string)
    # search_tensors = tf.convert_to_tensor(train_search, dtype=tf.string)
    # box_tensors = tf.convert_to_tensor(train_box, dtype=tf.float64)
    # input_queue = tf.train.slice_input_producer([search_tensors, target_tensors, box_tensors], shuffle=True)

    # thanks for https://github.com/nrupatunga/PY-GOTURN
    logger.info('Loading training data')
    # TODO, Load imagenet training images and annotations
    imagenet_folder = os.path.join(POLICY['imagenet'], 'images')
    imagenet_annotations_folder = os.path.join(POLICY['imagenet'], 'gt')
    objLoaderImgNet = loader_imagenet(imagenet_folder, imagenet_annotations_folder, logger)
    train_imagenet_images = objLoaderImgNet.loaderImageNetDet()

    # Load alov training images and annotations
    alov_folder = os.path.join(POLICY['alov'], 'images')
    alov_annotations_folder = os.path.join(POLICY['alov'], 'gt')
    objLoaderAlov = loader_alov(alov_folder, alov_annotations_folder, logger)
    objLoaderAlov.loaderAlov()
    train_alov_videos = objLoaderAlov.get_videos()

    # batch_queue = next_batch(objLoaderImgNet, train_imagenet_images, train_alov_videos)
    alov_images = 0
    for vid_idx in xrange(len(train_alov_videos)):
        video = train_alov_videos[vid_idx]
        annos = video.annotations
        alov_images += len(annos)
    total_image_size = len(train_imagenet_images) + alov_images
    logger.info('total training image size is: IMAGENET: ' + str(len(train_imagenet_images)) + ' and ALOV: ' + str(alov_images))
    tracknet = goturn_net.TRACKNET(BATCH_SIZE)
    tracknet.build()

    global_step = tf.Variable(0, trainable=False, name="global_step")

    train_step = tf.train.AdamOptimizer(0.0001, 0.9).minimize( \
        tracknet.loss_wdecay, global_step=global_step)
    merged_summary = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter('./train_summary', sess.graph)
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)

    coord = tf.train.Coordinator()
    # start the threads
    tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt_dir = "./checkpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    start = 0

    if ckpt and ckpt.model_checkpoint_path:
        start = int(ckpt.model_checkpoint_path.split("-")[1])
        logger.info("start by iteration: %d" % (start))
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
        logger.info("model is restored using " + str(ckpt))
    elif pretraind_model:
        restore = {}
        from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables
        slim = tf.contrib.slim
        for scope in list_variables(pretraind_model):
            if not 'fc4' in scope[0]:
                variables_to_restore = slim.get_variables(scope=scope[0])
                if variables_to_restore:
                    restore[scope[0]] = variables_to_restore[0]                                # variables_to_restore is list : [op]
        saver = tf.train.Saver(restore)
        saver.restore(sess, pretraind_model)
        logger.info("model is restored using " + str(pretraind_model))

    assign_op = global_step.assign(start)
    sess.run(assign_op)
    model_saver = tf.train.Saver(max_to_keep=10)
    try:
        for i in range(start, int((total_image_size) / BATCH_SIZE * NUM_EPOCHS)):
            if i % int((total_image_size) / BATCH_SIZE) == 0:
                logger.info("start epoch[%d]" % (int(float(i) / (total_image_size) * BATCH_SIZE)))
                if i > start:
                    save_ckpt = "checkpoint.ckpt"
                    last_save_itr = i
                    model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i + 1)

            start_time = time.time()
            cur_batch = data_reader(objLoaderImgNet, train_imagenet_images, train_alov_videos)
            logger.debug('data_reader: time elapsed: %.3f' % (time.time() - start_time))

            start_time = time.time()
            if DEBUG:
                idx = 0
                for idx in xrange(len(cur_batch[0])):
                    H, W, C = cur_batch[0][idx].shape
                    cv2.imwrite(str(idx) + '_' + 'image.jpg', cur_batch[0][idx])
                idx = 0
                for idx in xrange(len(cur_batch[1])):
                    H, W, C = cur_batch[1][idx].shape
                    cv2.imwrite(str(idx) + '_' + 'target.jpg', cur_batch[1][idx])


            feed_val, error_box_index = tracknet._batch(cur_batch[2], POLICY)
            cur_batch[0] = np.delete(cur_batch[0], error_box_index, 0)
            cur_batch[1] = np.delete(cur_batch[1], error_box_index, 0)
            cur_batch[2] = np.delete(cur_batch[2], error_box_index, 0)

            [_, loss] = sess.run([train_step, tracknet.loss], feed_dict={tracknet.image: cur_batch[0],
                                                                         tracknet.target: cur_batch[1],
                                                                         tracknet.bbox: cur_batch[2],
                                                                         tracknet.confs: feed_val['confs'],
                                                                         tracknet.coord: feed_val['coord'],
                                                                         tracknet.upleft: feed_val['upleft'],
                                                                         tracknet.botright: feed_val['botright'],
                                                                         tracknet.areas: feed_val['areas']})
            logger.debug('Train: time elapsed: %.3fs, average_loss: %f' % (time.time() - start_time, loss))

            if i % 10 == 0 and i > start:
                summary = sess.run(merged_summary, feed_dict={tracknet.image: cur_batch[0],
                                                              tracknet.target: cur_batch[1],
                                                              tracknet.bbox: cur_batch[2],
                                                              tracknet.confs: feed_val['confs'],
                                                              tracknet.coord: feed_val['coord'],
                                                              tracknet.upleft: feed_val['upleft'],
                                                              tracknet.botright: feed_val['botright'],
                                                              tracknet.areas: feed_val['areas']})
                train_writer.add_summary(summary, i)
    except KeyboardInterrupt:
        print("get keyboard interrupt")
        if (i - start > 1000):
            model_saver = tf.train.Saver()
            save_ckpt = "checkpoint.ckpt"
            model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i + 1)

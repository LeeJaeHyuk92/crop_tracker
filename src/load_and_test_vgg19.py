# train file

import time
import tensorflow as tf
import os
import goturn_vgg19
import numpy as np
import cv2
from helper.config import POLICY
from helper.BoundingBox import calculate_box, non_max_suppression_fast
from loader.loader_vot import loader_vot
from logger.logger import setup_logger
from tracker_vgg19 import bbox_estimator

opencv_version = cv2.__version__.split('.')[0]
NUM_EPOCHS = 500
BATCH_SIZE = 1
WIDTH = 224
HEIGHT = 224
ckpt_dir = POLICY['checkpoint']

logger = setup_logger(logfile=None)
logfile = "test.log"
test_txt = "test_set.txt"


def all_checkpoint(ckpt_dir):
    ckpt_path = []
    for ckpt in os.listdir(ckpt_dir):
        if ckpt == 'checkpoint':
            continue

        if not ckpt.split('.')[0] + '.' + ckpt.split('.')[1] in ckpt_path:
            ckpt_path.append(os.path.join(ckpt_dir, ckpt.split('.')[0] + '.' + ckpt.split('.')[1]))

    return ckpt_path


def load_train_test_set(train_file):
    '''
    return train_set or test_set
    example line in the file:
    <target_image_path>,<search_image_path>,<x1>,<y1>,<x2>,<y2>
    (<x1>,<y1>,<x2>,<y2> all relative to search image)
    '''
    ftrain = open(train_file, "r")
    trainlines = ftrain.read().splitlines()
    train_target = []
    train_search = []
    train_box = []
    for line in trainlines:
        # print(line)
        line = line.split(",")
        # remove too extreme cases
        # if (float(line[2]) < -0.3 or float(line[3]) < -0.3 or float(line[4]) > 1.2 or float(line[5]) > 1.2):
        #     continue
        train_target.append(line[0])
        train_search.append(line[1])
        box = [10 * float(line[2]), 10 * float(line[3]), 10 * float(line[4]), 10 * float(line[5])]
        train_box.append(box)
    ftrain.close()
    print("len:%d" % (len(train_target)))

    return [train_target, train_search, train_box]


def data_reader(input_queue):
    '''
    this function only reads the image from the queue
    '''
    search_img = tf.read_file(input_queue[0])
    target_img = tf.read_file(input_queue[1])

    search_tensor = tf.to_float(tf.image.decode_jpeg(search_img, channels=3))
    search_tensor = tf.image.resize_images(search_tensor, [HEIGHT, WIDTH],
                                           method=tf.image.ResizeMethod.CUBIC)
    target_tensor = tf.to_float(tf.image.decode_jpeg(target_img, channels=3))
    target_tensor = tf.image.resize_images(target_tensor, [HEIGHT, WIDTH],
                                           method=tf.image.ResizeMethod.CUBIC)
    box_tensor = input_queue[2]
    return [search_tensor, target_tensor, box_tensor]


def next_batch(input_queue):
    min_queue_examples = 128
    num_threads = 8
    [search_tensor, target_tensor, box_tensor] = data_reader(input_queue)
    [search_batch, target_batch, box_batch] = tf.train.batch(
        [search_tensor, target_tensor, box_tensor],
        batch_size=BATCH_SIZE,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads + 2) * BATCH_SIZE)
    return [search_batch, target_batch, box_batch]


def videos_prediction(videos, tracknet, sess, ckpt):
    video_keys = videos.keys()
    for i in range(0, len(videos)):
        video_frames = videos[video_keys[i]][0]
        annot_frames = videos[video_keys[i]][1]

        num_frames = min(len(video_frames), len(annot_frames))

        # Get the first frame of this video with the intial ground-truth bounding box
        frame_0 = video_frames[0]
        bbox_0 = annot_frames[0]
        sMatImage = cv2.imread(frame_0)
        bbox_estim.init(sMatImage, bbox_0)
        start_time = time.time()

        for i in xrange(1, num_frames):
            frame = video_frames[i]
            sMatImage = cv2.imread(frame)
            sMatImageDraw = sMatImage.copy()
            bbox = annot_frames[i]

            if opencv_version == '2':
                cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)),
                              (255, 255, 255), 2)
            else:
                sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)),
                                              (255, 255, 255), 2)

                bbox = bbox_estim.track(sMatImage, tracknet, sess)

            if not bbox == False:
                if opencv_version == '2':
                    cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)),
                                  (255, 0, 0), 2)
                else:
                    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)),
                                                  (int(bbox.x2), int(bbox.y2)),
                                                  (255, 0, 0), 2)

                cv2.imwrite(
                    './result/' + str(ckpt.split('-')[-1]) + "_" + frame.split('/')[-2] + "_" + frame.split('/')[-1],
                    sMatImageDraw)
                # cv2.imshow('Results', sMatImageDraw)
                # cv2.waitKey(10)
        logger.debug(frame_0)
        logger.debug('test: time elapsed: %.3f fps.' % (1 / ((time.time() - start_time) / num_frames)))


if __name__ == "__main__":
    if (os.path.isfile(logfile)):
        os.remove(logfile)
    # logger.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
    #     level=logger.DEBUG,filename=logfile)

    # [train_target, train_search, train_box] = load_train_test_set(test_txt)
    # target_tensors = tf.convert_to_tensor(train_target, dtype=tf.string)
    # search_tensors = tf.convert_to_tensor(train_search, dtype=tf.string)
    # box_tensors = tf.convert_to_tensor(train_box, dtype=tf.float64)
    # input_queue = tf.train.slice_input_producer([search_tensors, target_tensors, box_tensors],shuffle=False)
    # batch_queue = next_batch(input_queue)
    bbox_estim = bbox_estimator(False, logger)
    tracknet = goturn_vgg19.TRACKNET(BATCH_SIZE, train=False)
    tracknet.build()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)

    coord = tf.train.Coordinator()
    # start the threads
    tf.train.start_queue_runners(sess=sess, coord=coord)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ###
    # ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver = tf.train.Saver()
    #     saver.restore(sess, ckpt.model_checkpoint_path)
    #     ckpt = ckpt.model_checkpoint_path
    #     logger.info(str(ckpt) + " is restored")

    all_ckpt = all_checkpoint(ckpt_dir)
    for ckpt in all_ckpt:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        logger.info(str(ckpt) + " is restored")

        try:
            # for i in range(0, int(len(train_box)/BATCH_SIZE)):
            # cur_batch = sess.run(batch_queue)

            # import cv2
            # cv2.imwrite(str(i) + '_test' + 'image.jpg', cur_batch[1][0,...].astype(np.uint8))

            objLoaderVot = loader_vot(POLICY['vot'], logger)
            videos = objLoaderVot.get_videos()
            videos_prediction(videos, tracknet, sess, ckpt)

            # feed_val, _ = tracknet._batch(cur_batch[2], POLICY)
            # start_time = time.time()
            # [batch_loss, fc4] = sess.run([tracknet.loss, tracknet.fc4], feed_dict={tracknet.image: cur_batch[0],
            #                                                                        tracknet.target: cur_batch[1],
            #                                                                        tracknet.bbox: cur_batch[2],
            #                                                                        tracknet.confs: feed_val['confs'],
            #                                                                        tracknet.coord: feed_val['coord'],
            #                                                                        tracknet.upleft: feed_val['upleft'],
            #                                                                        tracknet.botright: feed_val['botright'],
            #                                                                        tracknet.areas: feed_val['areas']})
            #
            # temp = sess.run(tracknet.fc2, feed_dict={tracknet.image: cur_batch[0],
            #                                          tracknet.target: cur_batch[1],
            #                                          tracknet.bbox: cur_batch[2],
            #                                          tracknet.confs: feed_val['confs'],
            #                                          tracknet.coord: feed_val['coord'],
            #                                          tracknet.upleft: feed_val['upleft'],
            #                                          tracknet.botright: feed_val['botright'],
            #                                          tracknet.areas: feed_val['areas']})
            # predict_boxes = calculate_box(fc4)
            # predict_boxes = non_max_suppression_fast(predict_boxes, POLICY['thresh_IOU'])
            # image = cur_batch[0][0, ...].astype(np.uint8)
            # for pbox in predict_boxes:
            #     image = cv2.rectangle(image, (int(227 * pbox[0]/10), int(227 * pbox[1]/10)),
            #                           (int(227 * pbox[2]/10), int(227 * pbox[3]/10)), (0, 255, 0), 2)
            #     cv2.putText(image, str(pbox[4]), (int(227 * pbox[0]/10), int(227 * pbox[1]/10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
            #                 2)
            #     cv2.imwrite(str(i) + '_pred_' + 'image.jpg', image)
            #
            # logger.info('temp: %s' % (temp))
            # logger.info('batch box: %s' %(predict_boxes))
            # logger.info('gt batch box: %s' %(cur_batch[2]))
            # logger.info('batch loss = %f'%(batch_loss))
            # logger.debug('test: time elapsed: %.3fs.'%(time.time()-start_time))
        except KeyboardInterrupt:
            print("get keyboard interrupt")



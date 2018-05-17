import tensorflow as tf
import numpy as np
from helper.config import POLICY
from logger.logger import setup_logger

logger = setup_logger(logfile=None)
VGG_MEAN = [103.939, 116.779, 123.68] ## RGB
slim = tf.contrib.slim

def expit_tensor(x):
    return 1. / (1. + tf.exp(-x))


# https://github.com/tensorflow/tensorflow/issues/4079
def LeakyReLU(x, alpha=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + alpha)
        f2 = 0.5 * (1.0 - alpha)
        return f1 * x + f2 * abs(x)


class TRACKNET:
    def __init__(self, batch_size, vgg19_npy_path=None, train=True):

        self.batch_size = batch_size
        self.target = tf.placeholder(tf.float32, [None, POLICY['WIDTH'], POLICY['HEIGHT'], 3])
        self.image = tf.placeholder(tf.float32, [None, POLICY['WIDTH'], POLICY['HEIGHT'], 3])
        self.parameters = {}
        self.outdim = POLICY['side'] * POLICY['side'] * POLICY['num'] * 5
        self.train = train
        self.wd = 0.0005

        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}

        if train:
            self.bbox = tf.placeholder(tf.float32, [None, 4])
            self.confs = tf.placeholder(tf.float32, [None, POLICY['side'] * POLICY['side'], POLICY['num']])
            self.coord = tf.placeholder(tf.float32, [None, POLICY['side'] * POLICY['side'], POLICY['num'], 4])
            self.upleft = tf.placeholder(tf.float32, [None, POLICY['side'] * POLICY['side'], POLICY['num'], 2])
            self.botright = tf.placeholder(tf.float32, [None, POLICY['side'] * POLICY['side'], POLICY['num'], 2])
            self.areas = tf.placeholder(tf.float32, [None, POLICY['side'] * POLICY['side'], POLICY['num']])

    def build(self):
        ########### for target ###########
        # [filter_height, filter_width, in_channels, out_channels]
        tf.summary.image("image", self.image, max_outputs=2)
        tf.summary.image("target", self.target, max_outputs=2)

        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=self.image)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        self.image_m = tf.concat(axis=3, values=[
            blue - VGG_MEAN[2],
            green - VGG_MEAN[1],
            red - VGG_MEAN[1],
        ])
        assert self.image_m.get_shape().as_list()[1:] == [224, 224, 3]

        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=self.target)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        self.target_m = tf.concat(axis=3, values=[
            blue - VGG_MEAN[2],
            green - VGG_MEAN[1],
            red - VGG_MEAN[1],
        ])
        assert self.target_m.get_shape().as_list()[1:] == [224, 224, 3]



        self.target_conv1_1 = self.conv_layer(self.target_m, 3, 64, "target_conv1_1", "conv1_1")
        self.target_conv1_2 = self.conv_layer(self.target_conv1_1, 64, 64, "target_conv1_2", "conv1_2")
        self.target_pool1 = self.max_pool(self.target_conv1_2, 'target_pool1')

        self.target_conv2_1 = self.conv_layer(self.target_pool1, 64, 128, "target_conv2_1", "conv2_1")
        self.target_conv2_2 = self.conv_layer(self.target_conv2_1, 128, 128, "target_conv2_2", "conv2_2")
        self.target_pool2 = self.max_pool(self.target_conv2_2, 'target_pool2')

        self.target_conv3_1 = self.conv_layer(self.target_pool2, 128, 256, "target_conv3_1", "conv3_1")
        self.target_conv3_2 = self.conv_layer(self.target_conv3_1, 256, 256, "target_conv3_2", "conv3_2")
        self.target_conv3_3 = self.conv_layer(self.target_conv3_2, 256, 256, "target_conv3_3", "conv3_3")
        self.target_conv3_4 = self.conv_layer(self.target_conv3_3, 256, 256, "target_conv3_4", "conv3_4")
        self.target_pool3 = self.max_pool(self.target_conv3_4, 'target_pool3')

        self.target_conv4_1 = self.conv_layer(self.target_pool3, 256, 512, "target_conv4_1", "conv4_1")
        self.target_conv4_2 = self.conv_layer(self.target_conv4_1, 512, 512, "target_conv4_2", "conv4_2")
        self.target_conv4_3 = self.conv_layer(self.target_conv4_2, 512, 512, "target_conv4_3", "conv4_3")
        self.target_conv4_4 = self.conv_layer(self.target_conv4_3, 512, 512, "target_conv4_4", "conv4_4")
        self.target_pool4 = self.max_pool(self.target_conv4_4, 'target_pool4')

        self.target_conv5_1 = self.conv_layer(self.target_pool4, 512, 512, "target_conv5_1", "conv5_1")
        self.target_conv5_2 = self.conv_layer(self.target_conv5_1, 512, 512, "target_conv5_2", "conv5_2")
        self.target_conv5_3 = self.conv_layer(self.target_conv5_2, 512, 512, "target_conv5_3", "conv5_3")
        self.target_conv5_4 = self.conv_layer(self.target_conv5_3, 512, 512, "target_conv5_4", "conv5_4")
        self.target_pool5 = self.max_pool(self.target_conv5_4, 'target_pool5')

        ########### for image ###########
        # [filter_height, filter_width, in_channels, out_channels]
        self.image_conv1_1 = self.conv_layer(self.image_m, 3, 64, "image_conv1_1", "conv1_1")
        self.image_conv1_2 = self.conv_layer(self.image_conv1_1, 64, 64, "image_conv1_2", "conv1_2")
        self.image_pool1 = self.max_pool(self.image_conv1_2, 'image_pool1')

        self.image_conv2_1 = self.conv_layer(self.image_pool1, 64, 128, "image_conv2_1", "conv2_1")
        self.image_conv2_2 = self.conv_layer(self.image_conv2_1, 128, 128, "image_conv2_2", "conv2_2")
        self.image_pool2 = self.max_pool(self.image_conv2_2, 'image_pool2')

        self.image_conv3_1 = self.conv_layer(self.image_pool2, 128, 256, "image_conv3_1", "conv3_1")
        self.image_conv3_2 = self.conv_layer(self.image_conv3_1, 256, 256, "image_conv3_2", "conv3_2")
        self.image_conv3_3 = self.conv_layer(self.image_conv3_2, 256, 256, "image_conv3_3", "conv3_3")
        self.image_conv3_4 = self.conv_layer(self.image_conv3_3, 256, 256, "image_conv3_4", "conv3_4")
        self.image_pool3 = self.max_pool(self.image_conv3_4, 'image_pool3')

        self.image_conv4_1 = self.conv_layer(self.image_pool3, 256, 512, "image_conv4_1", "conv4_1")
        self.image_conv4_2 = self.conv_layer(self.image_conv4_1, 512, 512, "image_conv4_2", "conv4_2")
        self.image_conv4_3 = self.conv_layer(self.image_conv4_2, 512, 512, "image_conv4_3", "conv4_3")
        self.image_conv4_4 = self.conv_layer(self.image_conv4_3, 512, 512, "image_conv4_4", "conv4_4")
        self.image_pool4 = self.max_pool(self.image_conv4_4, 'image_pool4')

        self.image_conv5_1 = self.conv_layer(self.image_pool4, 512, 512, "image_conv5_1", "conv5_1")
        self.image_conv5_2 = self.conv_layer(self.image_conv5_1, 512, 512, "image_conv5_2", "conv5_2")
        self.image_conv5_3 = self.conv_layer(self.image_conv5_2, 512, 512, "image_conv5_3", "conv5_3")
        self.image_conv5_4 = self.conv_layer(self.image_conv5_3, 512, 512, "image_conv5_4", "conv5_4")
        self.image_pool5 = self.max_pool(self.image_conv5_4, 'image_pool5')

        ########### Concatnate two layers ###########
        self.concat = tf.concat([self.target_pool5, self.image_pool5], axis=3)
        # self.concat_1dconv = slim.conv2d(self.concat, 256, [1, 1], stride=1,
        #                                  padding='SAME',
        #                                  activation_fn=tf.nn.relu,
        #                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
        #                                  weights_regularizer=slim.l2_regularizer(0.0005))

        self.concat_1dconv = self.conv_1d_layer(self.concat, 1024, 256, "concat_1dconv", "None")

        ########### fully connencted layers ###########
        self.fc6 = self.fc_layer(self.concat_1dconv, 12544, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if (self.train):
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if (self.train):
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.fc8 = self.fc_layer(self.relu7, 4096, 5, "fc8")

        # tf.summary.image("objectness", self.net_grid[:, :, :, 4:], max_outputs=2)

        # self.print_shapes()
        self.data_dict = None


        self.net_grid = tf.reshape(self.fc8, shape=[-1, POLICY['side'], POLICY['side'], (4 + 1)])

        if (self.train):
            self.loss = self.loss_grid(self.fc8, POLICY, name="loss")
            # self.loss = self._loss_layer(self.fc4, self.bbox ,name = "loss")

            l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_weight_loss')
            self.loss_wdecay = self.loss + l2_loss

    def _loss_layer(self, bottom, label, name=None):
        diff = tf.subtract(self.fc8, self.bbox)
        diff_flat = tf.abs(tf.reshape(diff, [-1]))
        loss = tf.reduce_sum(diff_flat, name=name)
        return loss

    def loss_grid(self, net_out, training_schedule, name=None):
        """
        from YOLOv2, link: https://github.com/thtrieu/darkflow
        """
        # meta
        m = training_schedule
        sconf = float(m['object_scale'])
        snoob = float(m['noobject_scale'])
        scoor = float(m['coord_scale'])
        H, W = m['side'], m['side']
        B = m['num']
        HW = H * W  # number of grid cells
        anchors = m['anchors']

        # Extract the coordinate prediction from net.out
        net_out_reshape = tf.reshape(net_out, [-1, H, W, B, (4 + 1)])
        coords = net_out_reshape[:, :, :, :, :4]
        coords = tf.reshape(coords, [-1, H * W, B, 4])
        adjusted_coords_xy = expit_tensor(coords[:, :, :, 0:2])
        adjusted_coords_wh = tf.sqrt(
            tf.exp(coords[:, :, :, 2:4]) * tf.reshape(anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]) + 1e-8)

        coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

        adjusted_c = expit_tensor(net_out_reshape[:, :, :, :, 4])
        adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

        adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c], 3)

        wh = tf.pow(coords[:, :, :, 2:4], 2) * np.reshape([W, H], [1, 1, 1, 2])
        area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
        centers = coords[:, :, :, 0:2]
        floor = centers - (wh * .5)
        ceil = centers + (wh * .5)

        # calculate the intersection areas
        intersect_upleft = tf.maximum(floor, self.upleft)
        intersect_botright = tf.minimum(ceil, self.botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, self.areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
        best_box = tf.to_float(best_box)
        confs = tf.multiply(best_box, self.confs)

        # take care of the weight terms
        conid = snoob * (1. - confs) + sconf * confs
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
        cooid = scoor * weight_coo

        # self.fetch += [confs, conid, cooid]
        true = tf.concat([self.coord, tf.expand_dims(confs, 3)], 3)
        wght = tf.concat([cooid, tf.expand_dims(conid, 3)], 3)

        loss = tf.pow(adjusted_net_out - true, 2)
        loss = tf.multiply(loss, wght)

        tf.summary.scalar('loss_cx', tf.reduce_sum(loss, [0, 1, 2])[0])
        tf.summary.scalar('loss_cy', tf.reduce_sum(loss, [0, 1, 2])[1])
        tf.summary.scalar('loss_w_root', tf.reduce_sum(loss, [0, 1, 2])[2])
        tf.summary.scalar('loss_h_root', tf.reduce_sum(loss, [0, 1, 2])[3])
        tf.summary.scalar('loss_obj', tf.reduce_sum(loss, [0, 1, 2])[4])

        loss = tf.reshape(loss, [-1, H * W * B * (4 + 1)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss, name=name)
        tf.summary.scalar('total loss', loss)

        return loss

    def _batch(self, bboxes, training_schedule):
        """
        Takes a chunk of parsed annotations
        returns value for placeholders of net's
        input & loss layer correspond to this chunk
        :param box: box.x1, box.y1, box.x2, box.y2
        """
        error_box_index = []

        meta = training_schedule
        S, B = meta['side'], meta['num']
        w, h = 10, 10  # 10 is self.bbox's width, height
        # Calculate regression target
        cellx = 1. * w / S
        celly = 1. * h / S

        count = 0
        for idx, bbox in enumerate(bboxes):
            obj = [0, 0, 0, 0, 0]

            centerx = .5 * (bbox[0] + bbox[2])  # xmin, xmax
            centery = .5 * (bbox[1] + bbox[3])  # ymin, ymax
            cx = centerx / cellx
            cy = centery / celly
            # if cx >= S or cy >= S:
            #     raise ('center point error')
            #     return None, None
            obj[3] = (bbox[2] - bbox[0]) / w
            obj[4] = (bbox[3] - bbox[1]) / h
            obj[3] = np.sqrt(obj[3])
            obj[4] = np.sqrt(obj[4])
            obj[1] = cx - np.floor(cx)  # centerx
            obj[2] = cy - np.floor(cy)  # centery
            obj += [int(np.floor(cy) * S + np.floor(cx))]

            # show(im, allobj, S, w, h, cellx, celly) # unit test

            # Calculate placeholders' values
            confs = np.zeros([S * S, B])
            coord = np.zeros([S * S, B, 4])
            prear = np.zeros([S * S, 4])

            try:
                coord[obj[5], :, :] = [obj[1:5]] * B
                prear[obj[5], 0] = obj[1] - obj[3] ** 2 * .5 * S  # xleft
                prear[obj[5], 1] = obj[2] - obj[4] ** 2 * .5 * S  # yup
                prear[obj[5], 2] = obj[1] + obj[3] ** 2 * .5 * S  # xright
                prear[obj[5], 3] = obj[2] + obj[4] ** 2 * .5 * S  # ybot
                confs[obj[5], :] = [1.] * B

                # Finalise the placeholders' values
                upleft = np.expand_dims(prear[:, 0:2], 1)
                botright = np.expand_dims(prear[:, 2:4], 1)
                wh = botright - upleft
                area = wh[:, :, 0] * wh[:, :, 1]

                upleft = np.concatenate([upleft] * B, 1)
                botright = np.concatenate([botright] * B, 1)
                areas = np.concatenate([area] * B, 1)

                confs = np.expand_dims(confs, 0)
                coord = np.expand_dims(coord, 0)
                upleft = np.expand_dims(upleft, 0)
                botright = np.expand_dims(botright, 0)
                areas = np.expand_dims(areas, 0)
                if not count == 0:
                    batch_confs = np.concatenate([batch_confs, confs], axis=0)
                    batch_coord = np.concatenate([batch_coord, coord], axis=0)
                    batch_upleft = np.concatenate([batch_upleft, upleft], axis=0)
                    batch_botright = np.concatenate([batch_botright, botright], axis=0)
                    batch_areas = np.concatenate([batch_areas, areas], axis=0)
                else:
                    batch_confs = confs
                    batch_coord = coord
                    batch_upleft = upleft
                    batch_botright = botright
                    batch_areas = areas
                count += 1

            except IndexError:
                logger.error(str(idx) + ' is not boundary')
                logger.error('cx is ' + str(np.floor(cx)))
                logger.error('cy is ' + str(np.floor(cy)))
                error_box_index.append(idx)

        feed_val = {
            'confs': batch_confs, 'coord': batch_coord,
            'areas': batch_areas, 'upleft': batch_upleft,
            'botright': batch_botright
        }
        return feed_val, error_box_index

    def print_shapes(self):
        print("%s:" % (self.image_conv1), self.image_conv1.get_shape().as_list())
        print("%s:" % (self.image_pool1), self.image_pool1.get_shape().as_list())
        print("%s:" % (self.image_lrn1), self.image_lrn1.get_shape().as_list())
        print("%s:" % (self.image_conv2), self.image_conv2.get_shape().as_list())
        print("%s:" % (self.image_pool2), self.image_pool2.get_shape().as_list())
        print("%s:" % (self.image_lrn2), self.image_lrn2.get_shape().as_list())
        print("%s:" % (self.image_conv3), self.image_conv3.get_shape().as_list())
        print("%s:" % (self.image_conv4), self.image_conv4.get_shape().as_list())
        print("%s:" % (self.image_conv5), self.image_conv5.get_shape().as_list())
        print("%s:" % (self.image_pool5), self.image_pool5.get_shape().as_list())
        print("%s:" % (self.concat), self.concat.get_shape().as_list())
        print("%s:" % (self.fc1), self.fc1.get_shape().as_list())
        print("%s:" % (self.fc2), self.fc2.get_shape().as_list())
        print("%s:" % (self.fc3), self.fc3.get_shape().as_list())
        print("%s:" % (self.fc4), self.fc4.get_shape().as_list())
        print("kernel_sizes:")
        for key in self.parameters:
            print("%s:" % (key), self.parameters[key][0].get_shape().as_list())

    ### VGG 19
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_1d_layer(self, bottom, in_channels, out_channels, name, preweight, trainable=True):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(1, in_channels, out_channels, preweight, trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            weight_decay = tf.multiply(tf.nn.l2_loss(filt), 0.0005, name='fc_weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)

            # _activation_summary(relu)
            # relu = tf.Print(relu, [tf.shape(relu)], message='Shape of %s' % name, first_n=1, summarize=4)
            return relu

    def conv_layer(self, bottom, in_channels, out_channels, name, preweight, trainable=False):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, preweight, trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            # _activation_summary(relu)
            # relu = tf.Print(relu, [tf.shape(relu)], message='Shape of %s' % name, first_n=1, summarize=4)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name, trainable=True):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, trainable)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            weight_decay = tf.multiply(tf.nn.l2_loss(weights), 0.0005, name='fc_weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)

            # _activation_summary(fc)
            # fc = tf.Print(fc, [tf.shape(fc)], message='Shape of %s' % name, first_n=1, summarize=4)
            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name, trainable=False):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters", trainable)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable)

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, trainable=True):

        weights = tf.Variable(tf.truncated_normal([in_size, out_size], dtype=tf.float32, stddev=0.001), name='_weights')

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = tf.Variable(initial_value, name='_biases')

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name, trainable=True):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        # if trainable:
        #     var = tf.Variable(value, trainable=trainable, name=var_name)
        # else:
        #     var = tf.constant(value, dtype=tf.float32, name=var_name)

        var = tf.Variable(value, trainable=trainable, name=var_name)
        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.debug("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)

if __name__ == "__main__":
    tracknet = TRACKNET(10)
    tracknet.build()
    sess = tf.Session()
    a = np.full((tracknet.batch_size, 227, 227, 3), 1)
    b = np.full((tracknet.batch_size, 227, 227, 3), 2)
    sess.run(tf.global_variables_initializer())
    sess.run([tracknet.image_pool5], feed_dict={tracknet.image: a, tracknet.target: b})




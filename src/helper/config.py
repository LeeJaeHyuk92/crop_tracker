import tensorflow as tf

DEBUG = False

POLICY = {
    'logfile': 'train.log',
    'imagenet': '/home/jaehyuk/dataset/ILSVRC2014',
    'alov': '/home/jaehyuk/dataset/ALOV_vot2014',
    'vot': '/home/jaehyuk/dataset/vot2014_sample',
    'checkpoint': '/home/jaehyuk/code/experiment/crop_tracker/checkpoints',
    'vgg': './src/vgg19.npy',
    'thresh': .5,
    'thresh_IOU': .5,
    'num': 1,
    # 'anchors': [3.5, 3.5], # 3.0, 4.0, 4.0, 3.0, 3, 3, 4, 4],
    'anchors': [0.5, 0.5],
    # 'pretrained_model': '/datahdd/workdir/jaehyuk/dataset/pretrained/GOTURN/checkpoints/checkpoint.ckpt-1',
    'pretrained_model': None,

    # train
    'lamda_shift': 5.,
    'lamda_scale': 15.,
    'min_scale': -0.4,
    'max_scale': 0.4,

    'step_values': [200000, 300000, 400000, 500000],
    'learning_rates': [0.00001, 0.000005, 0.0000025, 0.00000125, 0.000000625],
    'momentum': 0.9,
    'momentum2': 0.999,
    'decay': 0.0005,
    'max_iter': 600000,

    'NUM_EPOCHS': 120,
    'BATCH_SIZE': 24,
    'WIDTH': 224,
    'HEIGHT': 224,
    'kGeneratedExamplesPerImage': 1,
    'side': 1,
    'channels': 3,
    'optimizer': dict({
        'rmsprop': tf.train.RMSPropOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adagradDA': tf.train.AdagradDAOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adam': tf.train.AdamOptimizer,
        'ftrl': tf.train.FtrlOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,}),

    'object_scale': 1,
    'noobject_scale': 1,
    'class_scale': 1,
    'coord_scale': 1,
    'PATHS': {
        'train': './data/tfrecords/train_1_adj.tfrecords',
        'validate': './data/tfrecords/tfc_val.tfrecords',
        'sample': './data/tfrecords/train_1_adj_sample.tfrecords',
    },
}
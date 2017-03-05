import tensorflow as tf
import numpy as np
from scipy.misc import imsave,toimage
from skimage.transform import resize
from copy import deepcopy
import os

import constants as c
if c.WGAN:
    from loss_functions_WGAN import g_loss
else:
    from loss_functions import combined_loss, adv_loss
from utils import psnr_error, sharp_diff_error, display_result
from tfutils import w, b, video_downsample, video_upsample, batch_norm, beta_, gamma_, pop_mean_, pop_var_

# noinspection PyShadowingNames
class GeneratorModel:
    def __init__(self, session, summary_writer, height_train, width_train, height_test,
                 width_test, scale_layer_fms, scale_kernel_sizes):
        """
        Initializes a GeneratorModel.

        @param session: The TensorFlow Session.
        @param summary_writer: The writer object to record TensorBoard summaries
        @param height_train: The height of the input images for training.
        @param width_train: The width of the input images for training.
        @param height_test: The height of the input images for testing.
        @param width_test: The width of the input images for testing.
        @param scale_layer_fms: The number of feature maps in each layer of each scale network.
        @param scale_kernel_sizes: The size of the kernel for each layer of each scale network.

        @type session: tf.Session
        @type summary_writer: tf.train.SummaryWriter
        @type height_train: int
        @type width_train: int
        @type height_test: int
        @type width_test: int
        @type scale_layer_fms: list<list<int>>
        @type scale_kernel_sizes: list<list<int>>
        """
        self.sess = session
        self.summary_writer = summary_writer
        self.height_train = height_train
        self.width_train = width_train
        self.height_test = height_test
        self.width_test = width_test
        self.scale_layer_fms = scale_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.num_scale_nets = len(scale_layer_fms)

        self.define_graph()

    # noinspection PyAttributeOutsideInit
    def define_graph(self):
        """
        Sets up the model graph in TensorFlow.
        """
        with tf.variable_scope('generator'):
            ##
            # Data
            ##

            with tf.name_scope('data'):
                self.input_frames_train = tf.placeholder(
                    tf.float32, shape=[None, self.height_train, self.width_train, c.NUM_INPUT_CHANNEL * c.HIST_LEN])
                self.gt_frames_train = tf.placeholder(
                    tf.float32, shape=[None, self.height_train, self.width_train, c.NUM_INPUT_CHANNEL*c.PRED_LEN])

                self.input_frames_test = tf.placeholder(
                    tf.float32, shape=[None, self.height_test, self.width_test, c.NUM_INPUT_CHANNEL * c.HIST_LEN])
                self.gt_frames_test = tf.placeholder(
                    tf.float32, shape=[None, self.height_test, self.width_test, c.NUM_INPUT_CHANNEL*c.PRED_LEN])
                self.convKeepProb = tf.placeholder(tf.float32)
                self.teacher_forcing = tf.placeholder(tf.int32, shape=[])
                self.bn_mode = tf.placeholder(tf.bool) # true: training , false:testing

                # # resize input to psedu_size
                # self.input_frames_train = tf.image.resize_images(self.input_frames_train,[c.PSEUDO_HEIGHT,c.PSEUDO_WIDTH])
                # self.gt_frames_train = tf.image.resize_images(self.gt_frames_train,[c.PSEUDO_HEIGHT,c.PSEUDO_WIDTH])
                # self.input_frames_test = tf.image.resize_images(self.input_frames_test,[c.PSEUDO_HEIGHT,c.PSEUDO_WIDTH])
                # self.gt_frames_test = tf.image.resize_images(self.gt_frames_test,[c.PSEUDO_HEIGHT,c.PSEUDO_WIDTH])

                # use variable batch_size for more flexibility
                self.batch_size_train = tf.shape(self.input_frames_train)[0]
                self.batch_size_test = tf.shape(self.input_frames_test)[0]

            ##
            # Scale network setup and calculation
            ##

            self.summaries_train = []
            self.scale_preds_train = []  # the generated images at each scale
            self.scale_inject_feature_train = [] # the injected features at each scale
            self.scale_gts_train = []  # the ground truth images at each scale
            self.fake_scale_logit = []  # the predictions from the discriminator model

            self.summaries_test = []
            self.scale_preds_test = []  # the generated images at each scale
            self.scale_inject_feature_test = [] # the injected features at each scale
            self.scale_gts_test = []  # the ground truth images at each scale

            self.optimizer_scale = []
            self.step_T = []
            self.train_op_scale = []

            for scale_num in xrange(self.num_scale_nets):
                with tf.variable_scope('scale_' + str(scale_num)):
                    with tf.name_scope('setup'):
                        ws = []
                        bs = []
                        betas = []
                        gammas = []
                        pop_means =[]
                        pop_vars=[]

                        # create weights for kernels
                        for i in xrange(len(self.scale_kernel_sizes[scale_num])):
                            ws_feature_input_num = self.scale_layer_fms[scale_num][i]
                            if c.RES_FEATURE is not True:
                                if scale_num>0:
                                    if i == c.INJECT_LAYER_TO[scale_num-1]:
                                        ws_feature_input_num = self.scale_layer_fms[scale_num][i]+self.scale_layer_fms[scale_num-1][c.INJECT_LAYER_FROM[scale_num-1]]
                                
                            ws.append(w([self.scale_kernel_sizes[scale_num][i],
                                         self.scale_kernel_sizes[scale_num][i],
                                         ws_feature_input_num,#self.scale_layer_fms[scale_num][i],
                                         self.scale_layer_fms[scale_num][i + 1]],
                                         scope = 'convlayer_' + str(i)))
                            bs.append(b([self.scale_layer_fms[scale_num][i + 1]],
                                         scope = 'convlayer_' + str(i)))
                            betas.append(beta_([self.scale_layer_fms[scale_num][i + 1]],
                                         scope = 'convlayer_' + str(i)))
                            gammas.append(gamma_([self.scale_layer_fms[scale_num][i + 1]],
                                         scope = 'convlayer_' + str(i)))
                            pop_means.append(pop_mean_([self.scale_layer_fms[scale_num][i + 1]],
                                         scope = 'convlayer_' + str(i)))
                            pop_vars.append(pop_var_([self.scale_layer_fms[scale_num][i + 1]],
                                         scope = 'convlayer_' + str(i)))

                    with tf.name_scope('calculation'):
                        def calculate(height, width, inputs, gts, last_gen_frames):
                            # scale inputs and gts
                            scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                            scale_height = int(height * scale_factor)
                            scale_width = int(width * scale_factor)

                            inputs = tf.image.resize_images(inputs, [scale_height, scale_width])
                            inputs = video_downsample(inputs,1/scale_factor)
                            scale_gts = tf.image.resize_images(gts, [scale_height, scale_width])
                            scale_gts = video_downsample(scale_gts,1/scale_factor)

                            # for all scales but the first, add the frame generated by the last
                            # scale to the input
                            if scale_num > 0:

                                last_gen_frames = tf.cond(self.teacher_forcing < 1, lambda: last_gen_frames, lambda: video_downsample(tf.image.resize_images(scale_gts, [scale_height/2, scale_width/2]),2))
                                last_gen_frames = tf.image.resize_images(last_gen_frames,
                                                                         [scale_height,
                                                                         scale_width])
                                # inputs = tf.concat(3, [inputs, last_gen_frames])

                            # generated frame predictions
                            preds = inputs

                            # perform convolutions
                            with tf.name_scope('convolutions'):
                                for i in xrange(len(self.scale_kernel_sizes[scale_num])):
                                    # Convolve layer
                                    if scale_num > 0:
                                        if i == c.INJECT_LAYER_TO[scale_num-1]:
                                            if c.RES_FEATURE:
                                                preds = preds + last_gen_frames
                                            else:
                                                preds = tf.concat(3, [preds, last_gen_frames])

                                    preds = tf.nn.conv2d(
                                        preds, ws[i], [1, 1, 1, 1], padding=c.PADDING_G)

                                    if c.BATCH_NORM:
                                        preds = batch_norm(preds,betas[i], gammas[i], pop_means[i], pop_vars[i], 
                                                            self.bn_mode )

                                    # Activate with ReLU (or Tanh for last layer)
                                    if i == len(self.scale_kernel_sizes[scale_num]) - 1:
                                        preds = tf.nn.tanh(preds + bs[i])
                                    else:
                                        preds = tf.nn.relu(preds + bs[i])
                                        preds = tf.nn.dropout(preds,self.convKeepProb)
                                    if i == (len(self.scale_kernel_sizes[scale_num]) + c.INJECT_LAYER_FROM[scale_num]):
                                        inject = preds

                            # perform Laplacian
                            if c.DOLAPLACIAN==1 and scale_num > 0:
                                with tf.name_scope('Laplacian'):
                                    last_gen_temporal_up = video_upsample(last_gen_frames)
                                    preds = preds + last_gen_temporal_up

                            return preds, scale_gts, inject


                        ##
                        # Perform train calculation
                        ##

                        # for all scales but the first, add the frame generated by the last
                        # scale to the input
                        if scale_num > 0:
                            last_scale_pred_train = self.scale_preds_train[scale_num - 1]
                            last_scale_feature_train = self.scale_inject_feature_train[scale_num - 1]
                        else:
                            last_scale_pred_train = None
                            last_scale_feature_train = None

                        # calculate
                        train_preds, train_gts, inject_feature_train = calculate(self.height_train,
                                                                           self.width_train,
                                                                           self.input_frames_train,
                                                                           self.gt_frames_train,
                                                                           last_scale_feature_train)
                        self.scale_preds_train.append(train_preds)
                        self.scale_inject_feature_train.append(inject_feature_train)
                        self.scale_gts_train.append(train_gts)

                        # We need to run the network first to get generated frames, run the
                        # discriminator on those frames to get fake_scale_logit, then run this
                        # again for the loss optimization.
                        if c.ADVERSARIAL:
                            self.fake_scale_logit.append(tf.placeholder(tf.float32, [None, 1]))

                        ##
                        # Perform test calculation
                        ##

                        # for all scales but the first, add the frame generated by the last
                        # scale to the input
                        if scale_num > 0:
                            last_scale_pred_test = self.scale_preds_test[scale_num - 1]
                            last_scale_feature_test = self.scale_inject_feature_test[scale_num - 1]
                        else:
                            last_scale_pred_test = None
                            last_scale_feature_test = None

                        # calculate
                        test_preds, test_gts, inject_feature_test = calculate(self.height_train,
                                                                         self.width_train,
                                                                         self.input_frames_test,
                                                                         self.gt_frames_test,
                                                                         last_scale_feature_test)
                        self.scale_preds_test.append(test_preds)
                        self.scale_inject_feature_test.append(inject_feature_test)
                        self.scale_gts_test.append(test_gts)

            ##
            # Training
            ##

            with tf.name_scope('train'):
                # loss for Non-teacher-forcing
                ## global loss is the combined loss from every scale network
                batch_size = tf.shape(self.scale_preds_train[0])[0]

                self.g_loss = g_loss(self.scale_preds_train,
                                                             self.scale_gts_train,
                                                             self.fake_scale_logit)[0]

                #self.adversarial_loss = adv_loss(self.fake_scale_logit,tf.ones([batch_size, 1]))[0]
                
                #self.optimizer = tf.train.AdamOptimizer(learning_rate=c.LRATE_G, name='optimizer')
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=c.LRATE_G, name='optimizer')

                self.global_step = tf.Variable(0, trainable=False)
                self.train_op = self.optimizer.minimize(self.g_loss,
                                                            global_step=self.global_step,
                                                            name='train_op')

                # scale0_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator/scale_0/setup")
                # scale3_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator/scale_3/setup")                    

                # self.scale0_grad = tf.gradients(self.g_loss, scale0_vars)
                # self.scale3_grad = tf.gradients(self.g_loss, scale3_vars)



                g_loss_summary = tf.scalar_summary('train_loss_G', self.g_loss)
                self.summaries_train.append(g_loss_summary)
                # adv_loss_summary = tf.scalar_summary('train_loss_G_adv', self.adversarial_loss)
                # self.summaries_train.append(adv_loss_summary)

                # loss for teacher forcing
                # self.g_loss_scale = combined_loss(self.scale_preds_train,
                #                                              self.scale_gts_train,
                #                                              self.fake_scale_logit)[1]
                # self.g_loss_scale = tf.unpack(self.g_loss_scale)

                # self.adversarial_loss_scale = adv_loss(self.fake_scale_logit,tf.ones([batch_size, 1]))[1]
                # self.adversarial_loss_scale = tf.pack(self.adversarial_loss_scale)

                # loss_num = len(self.g_loss_scale)

                # for i in xrange(loss_num):
                #     self.optimizer_scale.append(tf.train.AdamOptimizer(learning_rate=c.LRATE_G, name='optimizer'))
                #     self.step_T.append( tf.Variable(0, trainable=False))

                # for i in xrange(loss_num):
                #     self.train_op_scale.append( self.optimizer_scale[i].minimize(self.g_loss_scale[i],
                #                                                         global_step=self.step_T[i],
                #                                                         name='train_op'))

                #     # train loss summary
                #     g_loss_summary = tf.scalar_summary('train_loss_G'+str(i), self.g_loss_scale[i])
                #     self.summaries_train.append(g_loss_summary)
                #     adv_loss_summary = tf.scalar_summary('train_loss_G_adv'+str(i), self.adversarial_loss_scale[i])
                #     self.summaries_train.append(adv_loss_summary)
                # self.global_step = self.step_T[0]+self.step_NonT

            ##
            # Error
            ##

            with tf.name_scope('error'):
                # error computation
                # get error at largest scale
                self.psnr_error_train, _ = psnr_error(self.scale_preds_train[-1],
                                                   self.gt_frames_train)
                self.sharpdiff_error_train = sharp_diff_error(self.scale_preds_train[-1],
                                                              self.gt_frames_train)
                self.psnr_error_test, _ = psnr_error(self.scale_preds_test[-1],
                                                  self.gt_frames_test)
                self.sharpdiff_error_test = sharp_diff_error(self.scale_preds_test[-1],
                                                             self.gt_frames_test)
                # train error summaries
                summary_psnr_train = tf.scalar_summary('train_PSNR',
                                                       self.psnr_error_train)
                summary_sharpdiff_train = tf.scalar_summary('train_SharpDiff',
                                                            self.sharpdiff_error_train)
                self.summaries_train += [summary_psnr_train, summary_sharpdiff_train]

                # test error
                summary_psnr_test = tf.scalar_summary('test_PSNR',
                                                      self.psnr_error_test)
                summary_sharpdiff_test = tf.scalar_summary('test_SharpDiff',
                                                           self.sharpdiff_error_test)
                self.summaries_test += [summary_psnr_test, summary_sharpdiff_test]

            # add summaries to visualize in TensorBoard
            self.summaries_train = tf.merge_summary(self.summaries_train)
            self.summaries_test = tf.merge_summary(self.summaries_test)

    def train_step(self, batch, discriminator=None, teach = 0):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [c.BATCH_SIZE x self.height x self.width x (c.NUM_INPUT_CHANNEL * (c.HIST_LEN + 1))].
                      The input and output frames, concatenated along the channel axis (index 3).
        @param discriminator: The discriminator model. Default = None, if not adversarial.

        @return: The global step.
        """
        ##
        # Split into inputs and outputs
        ##

        input_frames = batch[:, :, :, :-c.NUM_INPUT_CHANNEL*c.PRED_LEN]
        gt_frames = batch[:, :, :, -c.NUM_INPUT_CHANNEL*c.PRED_LEN:]

        ##
        # Train
        ##

        feed_dict = {self.input_frames_train: input_frames, 
                     self.gt_frames_train: gt_frames,
                     self.convKeepProb: c.CONV_KEEPPROB,
                     self.teacher_forcing: teach,
                     self.bn_mode : True}

        if c.ADVERSARIAL:
            # Run the generator first to get generated frames
            scale_preds = self.sess.run(self.scale_preds_train, feed_dict=feed_dict)


            # Run the discriminator nets on those frames to get predictions
            d_feed_dict = {}
            if c.CONSIDER_PAST_FRAMES==0:
                for scale_num, gen_frames in enumerate(scale_preds):
                    d_feed_dict[discriminator.scale_nets[scale_num].input_frames] = gen_frames
                    d_feed_dict[discriminator.scale_nets[scale_num].fcKeepProb] = 1.0#c.FC_KEEPPROB#1.0
                    d_feed_dict[discriminator.scale_nets[scale_num].convKeepProb] = 1.0#c.CONV_KEEPPROB#1.0
                    d_feed_dict[discriminator.scale_nets[scale_num].bn_mode] = False # batch norm on test mode

            if c.CONSIDER_PAST_FRAMES ==1:
                batch_size = np.shape(input_frames)[0]
                for scale_num, gen_frames in enumerate(scale_preds):
                    scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                    scale_net = discriminator.scale_nets[scale_num]
                    scaled_hist_frames = np.empty([batch_size, scale_net.height, scale_net.width, c.NUM_INPUT_CHANNEL*c.HIST_LEN])
                    for i, img in enumerate(input_frames):
                        # for skimage.transform.resize, images need to be in range [0, 1], so normalize to
                        # [0, 1] before resize and back to [-1, 1] after
                        sknorm_img = (img / 2) + 0.5
                        resized_frame = resize(sknorm_img, [scale_net.height, scale_net.width, c.NUM_INPUT_CHANNEL*c.HIST_LEN])
                        scaled_hist_frames[i] = (resized_frame - 0.5) * 2
                    scaled_hist_frames = video_downsample(scaled_hist_frames,1/scale_factor)
                    scaled_all_frames_g = np.concatenate([scaled_hist_frames, gen_frames],axis=3)
                    d_feed_dict[discriminator.scale_nets[scale_num].input_frames] = scaled_all_frames_g
                    d_feed_dict[discriminator.scale_nets[scale_num].fcKeepProb] = 1.0#c.FC_KEEPPROB#1.0
                    d_feed_dict[discriminator.scale_nets[scale_num].convKeepProb] = 1.0#c.CONV_KEEPPROB#1.0
                    d_feed_dict[discriminator.scale_nets[scale_num].bn_mode] = False # batch norm on test mode

            fake_scale_logit = self.sess.run(discriminator.scale_preds, feed_dict=d_feed_dict)

            # Add discriminator predictions to the
            for i, logit in enumerate(fake_scale_logit):
                feed_dict[self.fake_scale_logit[i]] = logit

        # if c.TEACTHER_FORCE == 0:
        _, g_loss, global_psnr_error, global_sharpdiff_error, global_step, summaries = \
            self.sess.run([self.train_op,
                           self.g_loss,
                           self.psnr_error_train,
                           self.sharpdiff_error_train,
                           self.global_step,
                           self.summaries_train],
                          feed_dict=feed_dict)
        #scale0_g , scale3_g = self.sess.run([self.scale0_grad[0],self.scale3_grad[0]],feed_dict=feed_dict)


        # else:
        #     _, g_loss_scale, global_psnr_error, global_sharpdiff_error, global_step, summaries = \
        #         self.sess.run([self.train_op_scale,
        #                        self.g_loss_scale,
        #                        self.psnr_error_train,
        #                        self.sharpdiff_error_train,
        #                        self.global_step,
        #                        self.summaries_train],
        #                       feed_dict=feed_dict)

        ##
        # User output
        ##
        if global_step % c.STATS_FREQ == 0:
            print 'GeneratorModel : Step ', global_step
            print '                 Global Loss    : ', g_loss
            print '                 PSNR Error     : ', global_psnr_error
            print '                 Sharpdiff Error: ', global_sharpdiff_error
        if global_step % c.SUMMARY_FREQ == 0:
            self.summary_writer.add_summary(summaries, global_step)
            print 'GeneratorModel: saved summaries'
        if global_step % c.IMG_SAVE_FREQ == 0:
            print '-' * 30
            print 'Saving images...'

            # if not adversarial, we didn't get the preds for each scale net before for the
            # discriminator prediction, so do it now
            if not c.ADVERSARIAL:
                scale_preds = self.sess.run(self.scale_preds_train, feed_dict=feed_dict)

            # re-generate scale gt_frames to avoid having to run through TensorFlow.
            scale_gts = []
            scale_input = []
            for scale_num in xrange(self.num_scale_nets):
                scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                scale_height = int(self.height_train * scale_factor)
                scale_width = int(self.width_train * scale_factor)

                # resize gt_output_frames for scale and append to scale_gts_train
                scaled_gt_frames = np.empty([c.BATCH_SIZE, scale_height, scale_width, c.NUM_INPUT_CHANNEL*c.PRED_LEN])
                for i, img in enumerate(gt_frames):
                    # for skimage.transform.resize, images need to be in range [0, 1], so normalize
                    # to [0, 1] before resize and back to [-1, 1] after
                    sknorm_img = (img / 2) + 0.5
                    resized_frame = resize(sknorm_img, [scale_height, scale_width, c.NUM_INPUT_CHANNEL*c.PRED_LEN])                    
                    scaled_gt_frames[i] = (resized_frame - 0.5) * 2
                scaled_gt_frames = video_downsample(scaled_gt_frames,1/scale_factor)
                scale_gts.append(scaled_gt_frames)

                scaled_input_frames = np.empty([c.BATCH_SIZE, scale_height, scale_width, c.NUM_INPUT_CHANNEL*c.HIST_LEN])
                for i, img in enumerate(input_frames):
                    # for skimage.transform.resize, images need to be in range [0, 1], so normalize
                    # to [0, 1] before resize and back to [-1, 1] after
                    sknorm_img = (img / 2) + 0.5
                    resized_frame = resize(sknorm_img, [scale_height, scale_width, c.NUM_INPUT_CHANNEL*c.HIST_LEN])                    
                    scaled_input_frames[i] = (resized_frame - 0.5) * 2
                scaled_input_frames = video_downsample(scaled_input_frames,1/scale_factor)
                scale_input.append(scaled_input_frames)

            # for every clip in the batch, save the inputs, scale preds and scale gts
            pred_dir = c.get_dir(os.path.join(c.IMG_SAVE_DIR, 'Step_' + str(global_step)))
            for pred_num in xrange(len(input_frames)):

                # save preds and gts at each scale
                # noinspection PyUnboundLocalVariable
                for scale_num, scale_pred in enumerate(scale_preds):
                    gen_img = scale_pred[pred_num]
                    gen_len = gen_img.shape[2]
                    path = os.path.join(pred_dir, 'scale' + str(scale_num))
                    gt_img = scale_gts[scale_num][pred_num]
                    input_img = scale_input[scale_num][pred_num]

                    file = os.path.join(pred_dir, str(pred_num)+'_scale_'+str(scale_num))+'.pdf'
                    display_result(input_img,gen_img,gt_img,file)

            print 'Saved images!'
            print '-' * 30

        return global_step

    def test_batch(self, batch, global_step, num_rec_out=1, save_imgs=True, IndOffset = 0, discriminator=None):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [batch_size x self.height x self.width x (3 * (c.HIST_LEN+ num_rec_out))].
                      A batch of the input and output frames, concatenated along the channel axis
                      (index 3).
        @param global_step: The global step.
        @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                            using previously-generated frames as input. Default = 1.
        @param save_imgs: Whether or not to save the input/output images to file. Default = True.

        @return: A tuple of (psnr error, sharpdiff error) for the batch.
        """
        if num_rec_out < 1:
            raise ValueError('num_rec_out must be >= 1')

        print '-' * 30
        print 'Testing:'

        ##
        # Split into inputs and outputs
        ##

        input_frames = batch[:, :, :, :c.NUM_INPUT_CHANNEL * c.HIST_LEN]
        gt_frames = batch[:, :, :, c.NUM_INPUT_CHANNEL * c.HIST_LEN:]

        ##
        # Generate num_rec_out recursive predictions
        ##

        working_input_frames = deepcopy(input_frames)  # input frames that will shift w/ recursion
        
        rec_summaries = []
        total_preds=tf.placeholder(
                    tf.float32, shape=[None, self.height_train, self.width_train, c.NUM_INPUT_CHANNEL * num_rec_out])
        psnr_total,psnr_total_each = psnr_error(total_preds,gt_frames)
        for rec_num in xrange(0,num_rec_out,c.PRED_LEN):
            working_gt_frames = gt_frames[:, :, :, c.NUM_INPUT_CHANNEL * rec_num: c.NUM_INPUT_CHANNEL * (rec_num + c.PRED_LEN)]

            feed_dict = {self.input_frames_test: working_input_frames,
                         self.gt_frames_test: working_gt_frames,
                         self.convKeepProb: 1.0,
                         self.teacher_forcing: 0,
                         self.bn_mode: False}
            preds, psnr, sharpdiff, summaries = self.sess.run([self.scale_preds_test,
                                                               self.psnr_error_test,
                                                               self.sharpdiff_error_test,
                                                               self.summaries_test],
                                                              feed_dict=feed_dict)

            # from utils_debug import get_discriminate_pred
            # d_pred_scale_gen, d_pred_scale_gt = get_discriminate_pred(self.sess,discriminator,working_input_frames,working_gt_frames,preds)
            # import pdb; pdb.set_trace()  # breakpoint 3a72a72f //
            
            # remove first input and add new pred as last input
            working_input_frames = np.concatenate(
                [working_input_frames[:, :, :, c.NUM_INPUT_CHANNEL*c.PRED_LEN:], preds[-1]], axis=3)

            # add predictions and summaries
            if rec_num == 0:
                rec_preds = preds
            else:
                for scale_num, scale_pred in enumerate(preds):
                    rec_preds[scale_num] = np.concatenate([rec_preds[scale_num],scale_pred], axis=3)
            rec_summaries.append(summaries)


            # print 'Recursion ', rec_num
            # print 'PSNR Error     : ', psnr
            # print 'Sharpdiff Error: ', sharpdiff

        psnr_trials,psnr_totals_each = self.sess.run([psnr_total,psnr_total_each],feed_dict = {total_preds:rec_preds[-1]})

        print 'PSNR Error     : ', psnr_trials
        # write summaries
        # TODO: Think of a good way to write rec output summaries - rn, just using first output.
        self.summary_writer.add_summary(rec_summaries[0], global_step)

        ##
        # Save images
        ##

        if save_imgs:
            scale_gts = []
            scale_input = []
            for scale_num in xrange(self.num_scale_nets):
                scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                scale_height = int(self.height_train * scale_factor)
                scale_width = int(self.width_train * scale_factor)
                scale_len = int(num_rec_out * scale_factor)

                # resize gt_output_frames for scale and append to scale_gts_train
                scaled_gt_frames = np.empty([c.BATCH_SIZE, scale_height, scale_width, c.NUM_INPUT_CHANNEL*num_rec_out])
                for i, img in enumerate(gt_frames):
                    # for skimage.transform.resize, images need to be in range [0, 1], so normalize
                    # to [0, 1] before resize and back to [-1, 1] after
                    sknorm_img = (img / 2) + 0.5
                    resized_frame = resize(sknorm_img, [scale_height, scale_width, c.NUM_INPUT_CHANNEL*num_rec_out])                    
                    scaled_gt_frames[i] = (resized_frame - 0.5) * 2
                scaled_gt_frames = video_downsample(scaled_gt_frames,1/scale_factor)
                scale_gts.append(scaled_gt_frames)

                scaled_input_frames = np.empty([c.BATCH_SIZE, scale_height, scale_width, c.NUM_INPUT_CHANNEL*c.HIST_LEN])
                for i, img in enumerate(input_frames):
                    # for skimage.transform.resize, images need to be in range [0, 1], so normalize
                    # to [0, 1] before resize and back to [-1, 1] after
                    sknorm_img = (img / 2) + 0.5
                    resized_frame = resize(sknorm_img, [scale_height, scale_width, c.NUM_INPUT_CHANNEL*c.HIST_LEN])                    
                    scaled_input_frames[i] = (resized_frame - 0.5) * 2
                scaled_input_frames = video_downsample(scaled_input_frames,1/scale_factor)
                scale_input.append(scaled_input_frames)

            pred_dir = c.get_dir(os.path.join(c.IMG_SAVE_DIR, 'Tests/Step_' + str(global_step)))
            for pred_num in xrange(len(input_frames)):
                for scale_num, scale_pred in enumerate(rec_preds):

                    file = os.path.join(pred_dir, str(pred_num+IndOffset)+'_scale_'+str(scale_num)+'_psnr_'+str(psnr_totals_each[pred_num]))+'.pdf'
                    gen_img = scale_pred[pred_num]
                    gen_len = gen_img.shape[2]
                    path = os.path.join(pred_dir, 'scale' + str(scale_num))
                    gt_img = scale_gts[scale_num][pred_num]
                    input_img = scale_input[scale_num][pred_num]

                    display_result(input_img,gen_img,gt_img,file)
        print '-' * 30

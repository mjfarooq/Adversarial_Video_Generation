import tensorflow as tf
import numpy as np
from tfutils import video_downsample
from skimage.transform import resize
import constants as c


def get_discriminate_pred(sess,discriminator,input_frames,working_gt_frames,g_frames):
    num_scale_nets = len(g_frames)
    d_feed_dict_gen = {}
    d_feed_dict_gt = {}

    if c.CONSIDER_PAST_FRAMES ==1:
        batch_size = np.shape(input_frames)[0]
        for scale_num, gen_frames in enumerate(g_frames):
            scale_factor = 1. / 2 ** ((num_scale_nets - 1) - scale_num)
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
            d_feed_dict_gen[discriminator.scale_nets[scale_num].input_frames] = scaled_all_frames_g
            d_feed_dict_gen[discriminator.scale_nets[scale_num].fcKeepProb] = 1.0#c.FC_KEEPPROB#1.0
            d_feed_dict_gen[discriminator.scale_nets[scale_num].convKeepProb] = 1.0#c.CONV_KEEPPROB#1.0

            scaled_gt_frames = np.empty([batch_size, scale_net.height, scale_net.width, c.NUM_INPUT_CHANNEL*c.PRED_LEN])
            for i, img in enumerate(working_gt_frames):
                # for skimage.transform.resize, images need to be in range [0, 1], so normalize to
                # [0, 1] before resize and back to [-1, 1] after
                sknorm_img = (img / 2) + 0.5
                resized_frame = resize(sknorm_img, [scale_net.height, scale_net.width, c.NUM_INPUT_CHANNEL*c.PRED_LEN])
                scaled_gt_frames[i] = (resized_frame - 0.5) * 2
            scaled_gt_frames = video_downsample(scaled_gt_frames,1/scale_factor)
            
            scaled_all_frames_gt = np.concatenate([scaled_hist_frames, scaled_gt_frames],axis=3)
            d_feed_dict_gt[discriminator.scale_nets[scale_num].input_frames] = scaled_all_frames_gt
            d_feed_dict_gt[discriminator.scale_nets[scale_num].fcKeepProb] = 1.0#c.FC_KEEPPROB#1.0
            d_feed_dict_gt[discriminator.scale_nets[scale_num].convKeepProb] = 1.0#c.CONV_KEEPPROB#1.0
    d_pred_scales_gen = sess.run(discriminator.scale_preds, feed_dict=d_feed_dict_gen)
    d_pred_scales_gt = sess.run(discriminator.scale_preds, feed_dict=d_feed_dict_gt)
    return d_pred_scales_gen, d_pred_scales_gt
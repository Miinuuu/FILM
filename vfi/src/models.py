import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import numpy as np
from tensorflow_addons import optimizers as tfa_optimizers

#from feature_extractor import * 
#from fusion import * 
#from options import * 
#from pyramid_flow_estimator import *  
#from util import * 

from vfi.src import feature_extractor
from vfi.src import fusion
from vfi.src import options
from vfi.src import pyramid_flow_estimator
from vfi.src import util

def _mse_psnr(original, reconstruction,training):
  """Calculates mse and PSNR.

  If training is False, we quantize the pixel values before calculating the
  metrics.

  Args:
    original: Image, in [0, 1].
    reconstruction: Reconstruction, in [0, 1].
    training: Whether we are in training mode.

  Returns:
    Tuple mse, psnr.
  """
  # The images/reconstructions are in [0...1] range, but we scale them to
  # [0...255] before computing the MSE.
  mse_per_batch = tf.reduce_mean(
      tf.math.squared_difference(
          (original * 255.0),
          (reconstruction * 255.0)),
      axis=(1, 2, 3))
  mse = tf.reduce_mean(mse_per_batch)
  psnr_factor = -10. / tf.math.log(10.)
  psnr = tf.reduce_mean(psnr_factor * tf.math.log(mse_per_batch / (255.**2)))
  return mse, psnr


class Model(tf.Module):
    def __init__(self,config= options.Options) :
        super(Model, self).__init__()
   
        self.config=config
        self.extract = feature_extractor.FeatureExtractor('feat_net', self.config)
        self.predict_flow = pyramid_flow_estimator.PyramidFlowEstimator(
            'predict_flow', config)
        self.fuse = fusion.Fusion('fusion', self.config)
        self._all_trainable_variables = None
        #self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=.0001, decay_steps=750000, decay_rate=0.464158)
        #self.optimizer = tfa_optimizers.AdamW(learning_rate= self.learning_rate_schedule,weight_decay=0.0001)
        #self.learning_rate_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=0.0001, decay_steps=750000, alpha=0.464158)


        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.0001,
        decay_steps = 750000,
        decay_rate = 0.464158,
        staircase = True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate= self.learning_rate_schedule)


    def _iter_trainable_variables(self):

        def ensure_nonempty(seq):
            if not seq:
                raise ValueError("No trainable variables!")
            return seq

        yield from ensure_nonempty(self.extract.trainable_variables)
        yield from ensure_nonempty(self.predict_flow.trainable_variables)
        yield from ensure_nonempty(self.fuse.trainable_variables)

    @property
    def all_trainable_variables(self):
        if self._all_trainable_variables is None:
            self._all_trainable_variables = list(self._iter_trainable_variables())
            assert self._all_trainable_variables
            assert len(self._all_trainable_variables) == len(self.trainable_variables)
        return self._all_trainable_variables
            
    def train_step(self, img0,gt,img1):

        with tf.GradientTape() as tape:
            pred,metric = self.interpolator(img0,gt,img1)
            loss= metric['loss']
        var_list = self.all_trainable_variables
        gradients = tape.gradient ( loss, var_list)
        self.optimizer.apply_gradients(zip(gradients,var_list))

        return metric
    def write_ckpt(self, path, step):
        """Creates a checkpoint at `path` for `step`."""
        print('Creates a checkpoint at'+ str(path) + 'for' + str(step))
        ckpt = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
        manager.save(checkpoint_number=step)
        return tf.train.latest_checkpoint(path)
        
    def load_ckpt(self, path):
        """load a checkpoint at `path` for `step`."""
        ckpt = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=3)
        print('load a checkpoint at'+ str(path) + 'for' + manager.latest_checkpoint)
        return ckpt.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    @property
    def global_step(self):
        """Returns the global step variable."""
        return self._optimizer.iterations
    # Actually consist of 'calculate_flow' and 'coraseWarp_and_Refine'
    def interpolator(self,img0,gt,img1, timestep=0.5):
        #print("1. flow estimation")
        B = img0.shape[0] # batch size 
        image_pyramids = [
        util.build_image_pyramid(img0, self.config),
        util.build_image_pyramid(img1, self.config) ]

        
        feature_pyramids = [self.extract(image_pyramids[0]), self.extract(image_pyramids[1])]
 

        # Predict forward flow.
        forward_residual_flow_pyramid = self.predict_flow(feature_pyramids[0],
                                                    feature_pyramids[1])
        # Predict backward flow.
        backward_residual_flow_pyramid = self.predict_flow(feature_pyramids[1],
                                                        feature_pyramids[0])


        fusion_pyramid_levels = self.config.fusion_pyramid_levels

        forward_flow_pyramid = util.flow_pyramid_synthesis(
            forward_residual_flow_pyramid)[:fusion_pyramid_levels]
        backward_flow_pyramid =  util.flow_pyramid_synthesis(
            backward_residual_flow_pyramid)[:fusion_pyramid_levels]

        # We multiply the flows with t and 1-t to warp to the desired fractional time.
        #
        # Note: In film_net we fix time to be 0.5, and recursively invoke the interpo-
        # lator for multi-frame interpolation. Below, we create a constant tensor of
        # shape [B]. We use the `time` tensor to infer the batch size.
        mid_time = tf.keras.layers.Lambda(lambda x: tf.ones_like(x) * 0.5)(timestep)
        backward_flow =  util.multiply_pyramid(backward_flow_pyramid, mid_time)
        forward_flow =  util.multiply_pyramid(forward_flow_pyramid, 1 - mid_time)

        pyramids_to_warp = [
             util.concatenate_pyramids(image_pyramids[0][:fusion_pyramid_levels],
                                        feature_pyramids[0][:fusion_pyramid_levels]),
             util.concatenate_pyramids(image_pyramids[1][:fusion_pyramid_levels],
                                        feature_pyramids[1][:fusion_pyramid_levels])
        ]

        forward_warped_pyramid =  util.pyramid_warp(pyramids_to_warp[0], backward_flow)
        backward_warped_pyramid =  util.pyramid_warp(pyramids_to_warp[1], forward_flow)

        aligned_pyramid =  util.concatenate_pyramids(forward_warped_pyramid,
                                                    backward_warped_pyramid)
        aligned_pyramid =  util.concatenate_pyramids(aligned_pyramid, backward_flow)
        aligned_pyramid =  util.concatenate_pyramids(aligned_pyramid, forward_flow)

        
        prediction = self.fuse(aligned_pyramid)

        output_color = prediction[..., :3]
        outputs = {'image': output_color}

        loss_l1 = tf.reduce_mean(tf.abs(output_color-gt))
        mse,psnr = _mse_psnr(gt,output_color,False)
        metric={'loss' : loss_l1 , 'psnr' : psnr , 'mse':mse }     
        #return flow_list, mask_list, merged, pred
    
        return output_color, metric 
    

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Various utilities used in the film_net frame interpolator model."""
from typing import List

from options import Options 
import tensorflow as tf
import tensorflow_addons.image as tfa_image


def build_image_pyramid(image: tf.Tensor,
                        options: Options) -> List[tf.Tensor]:
  """Builds an image pyramid from a given image.

  The original image is included in the pyramid and the rest are generated by
  successively halving the resolution.

  Args:
    image: the input image.
    options: film_net options object

  Returns:
    A list of images starting from the finest with options.pyramid_levels items
  """
  levels = options.pyramid_levels
  pyramid = []
  pool = tf.keras.layers.AveragePooling2D(
      pool_size=2, strides=2, padding='valid')
  for i in range(0, levels):
    pyramid.append(image)
    if i < levels-1:
      image = pool(image)
  return pyramid

@tf.function
def warp(image: tf.Tensor, flow: tf.Tensor) -> tf.Tensor:
  """Backward warps the image using the given flow.

  Specifically, the output pixel in batch b, at position x, y will be computed
  as follows:
    (flowed_y, flowed_x) = (y+flow[b, y, x, 1], x+flow[b, y, x, 0])
    output[b, y, x] = bilinear_lookup(image, b, flowed_y, flowed_x)

  Note that the flow vectors are expected as [x, y], e.g. x in position 0 and
  y in position 1.

  Args:
    image: An image with shape BxHxWxC.
    flow: A flow with shape BxHxWx2, with the two channels denoting the relative
      offset in order: (dx, dy).
  Returns:
    A warped image.
  """
  # tfa_image.dense_image_warp expects unconventional negated optical flow, so
  # negate the flow here. Also revert x and y for compatibility with older saved
  # models trained with custom warp op that stored (x, y) instead of (y, x) flow
  # vectors.
  flow = -flow[..., ::-1]

  # Note: we have to wrap tfa_image.dense_image_warp into a Keras Lambda,
  # because it is not compatible with Keras symbolic tensors and we want to use
  # this code as part of a Keras model.  Wrapping it into a lambda has the
  # consequence that tfa_image.dense_image_warp is only called once the tensors
  # are concrete, e.g. actually contain data. The inner lambda is a workaround
  # for passing two parameters, e.g you would really want to write:
  # tf.keras.layers.Lambda(tfa_image.dense_image_warp)(image, flow), but this is
  # not supported by the Keras Lambda.
  warped = tf.keras.layers.Lambda(
      lambda x: tfa_image.dense_image_warp(*x))((image, flow))
  return tf.reshape(warped, shape=tf.shape(image))


def multiply_pyramid(pyramid: List[tf.Tensor],
                     scalar: tf.Tensor) -> List[tf.Tensor]:
  """Multiplies all image batches in the pyramid by a batch of scalars.

  Args:
    pyramid: Pyramid of image batches.
    scalar: Batch of scalars.

  Returns:
    An image pyramid with all images multiplied by the scalar.
  """
  # To multiply each image with its corresponding scalar, we first transpose
  # the batch of images from BxHxWxC-format to CxHxWxB. This can then be
  # multiplied with a batch of scalars, then we transpose back to the standard
  # BxHxWxC form.
  return [
      tf.transpose(tf.transpose(image, [3, 1, 2, 0]) * scalar, [3, 1, 2, 0])
      for image in pyramid
  ]


def flow_pyramid_synthesis(
    residual_pyramid: List[tf.Tensor]) -> List[tf.Tensor]:
  """Converts a residual flow pyramid into a flow pyramid."""
  flow = residual_pyramid[-1]
  flow_pyramid = [flow]
  for residual_flow in reversed(residual_pyramid[:-1]):
    level_size = tf.shape(residual_flow)[1:3]
    flow = tf.image.resize(images=2*flow, size=level_size)
    flow = residual_flow + flow
    flow_pyramid.append(flow)
  # Use reversed() to return in the 'standard' finest-first-order:
  return list(reversed(flow_pyramid))


def pyramid_warp(feature_pyramid: List[tf.Tensor],
                 flow_pyramid: List[tf.Tensor]) -> List[tf.Tensor]:
  """Warps the feature pyramid using the flow pyramid.

  Args:
    feature_pyramid: feature pyramid starting from the finest level.
    flow_pyramid: flow fields, starting from the finest level.

  Returns:
    Reverse warped feature pyramid.
  """
  warped_feature_pyramid = []
  for features, flow in zip(feature_pyramid, flow_pyramid):
    warped_feature_pyramid.append(warp(features, flow))
  return warped_feature_pyramid


def concatenate_pyramids(pyramid1: List[tf.Tensor],
                         pyramid2: List[tf.Tensor]) -> List[tf.Tensor]:
  """Concatenates each pyramid level together in the channel dimension."""
  result = []
  for features1, features2 in zip(pyramid1, pyramid2):
    result.append(tf.concat([features1, features2], axis=-1))
  return result


''' torch version
def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warps the image using the given flow.

    Specifically, the output pixel in batch b, at position x, y will be computed
    as follows:
      (flowed_y, flowed_x) = (y+flow[b, y, x, 1], x+flow[b, y, x, 0])
      output[b, y, x] = bilinear_lookup(image, b, flowed_y, flowed_x)

    Note that the flow vectors are expected as [x, y], e.g. x in position 0 and
    y in position 1.

    Args:
      image: An image with shape BxHxWxC.
      flow: A flow with shape BxHxWx2, with the two channels denoting the relative
        offset in order: (dx, dy).
    Returns:
      A warped image.
    """
    flow = -flow.flip(1)

    dtype = flow.dtype
    device = flow.device

    # warped = tfa_image.dense_image_warp(image, flow)
    # Same as above but with pytorch
    ls1 = 1 - 1 / flow.shape[3]
    ls2 = 1 - 1 / flow.shape[2]

    normalized_flow2 = flow.permute(0, 2, 3, 1) / torch.tensor(
        [flow.shape[2] * .5, flow.shape[3] * .5], dtype=dtype, device=device)[None, None, None]
    normalized_flow2 = torch.stack([
        torch.linspace(-ls1, ls1, flow.shape[3], dtype=dtype, device=device)[None, None, :] - normalized_flow2[..., 1],
        torch.linspace(-ls2, ls2, flow.shape[2], dtype=dtype, device=device)[None, :, None] - normalized_flow2[..., 0],
    ], dim=3)

    warped = F.grid_sample(image, normalized_flow2,
                           mode='bilinear', padding_mode='border', align_corners=False)
    return warped.reshape(image.shape)


'''
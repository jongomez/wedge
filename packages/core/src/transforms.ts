import * as tfOriginal from "@tensorflow/tfjs";
import { getElementCount } from "./backends/webgl/buffersAndTextures";
import { maxTextureDim } from "./constants";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tf = tfOriginal as any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TFTensor = any;

export function padChannels(tensor: TFTensor, nodeName: string): TFTensor {
  // Determine the shape based on whether a shape override is provided
  const originalShape = tensor.shape;

  // Handle different tensor dimensions
  let HWCShape: number[];
  let is1D: boolean = originalShape.length === 1;
  let is2D: boolean = originalShape.length === 2;
  let is3D: boolean = originalShape.length === 3;
  let is4D: boolean = originalShape.length === 4;

  if (is1D) {
    const originalLength = originalShape[0];
    const paddedLength = Math.ceil(originalLength / 4) * 4;
    const numElementsToAdd = paddedLength - originalLength;

    const zeros = tf.zeros([numElementsToAdd]);
    const paddedTensor = tf.concat([tensor, zeros]);

    return paddedTensor;
  } else if (is2D) {
    const originalLength = (originalShape as [number, number])[1];
    const paddedLength = Math.ceil(originalLength / 4) * 4;
    const numElementsToAdd = paddedLength - originalLength;

    const zeros = tf.zeros([originalShape[0], numElementsToAdd]);
    const paddedTensor = tf.concat([tensor, zeros], 1);

    return paddedTensor;
  } else if (is3D) {
    HWCShape = originalShape;
  } else if (is4D) {
    HWCShape = originalShape.slice(1); // Extract height, width, channels
  } else {
    throw new Error("Unsupported tensor dimension. Shape length: " + originalShape.length + " Node name: " + nodeName);
  }

  const channelPaddedShape = getChannelPaddedShape(HWCShape);
  const originalChannels = getNumChannel(HWCShape);
  const paddedChannels = getNumChannel(channelPaddedShape);
  const numChannelsToAdd = paddedChannels - originalChannels;

  // Create a zero tensor for the channels to add
  let zeros: TFTensor;
  if (is3D) {
    zeros = tf.zeros([HWCShape[0], HWCShape[1], numChannelsToAdd]);
  } else {
    zeros = tf.zeros([originalShape[0], HWCShape[0], HWCShape[1], numChannelsToAdd]);
  }

  const paddedTensor = tf.concat([tensor, zeros], tensor.shape.length - 1);
  return paddedTensor;
}


function getNumChannel(shape: number[]): number {
  // Assuming channels last - HWC
  return shape[shape.length - 1];
}

export function getChannelPaddedShape(originalShape: number[]): number[] {
  const currentChannels = getNumChannel(originalShape);
  const channelsToAdd = currentChannels % 4 === 0 ? 0 : 4 - (currentChannels % 4);
  const newChannels = currentChannels + channelsToAdd;
  const channelPaddedShape = [originalShape[0], originalShape[1], newChannels];

  return channelPaddedShape;
}

export function removePadChannels(
  dataWithPaddedChannels: Float32Array,
  originalElementCount: number,
  originalShape: number[]): Float32Array {

  const originalChannels = getNumChannel(originalShape);
  const finalChannels = Math.ceil(originalChannels / 4) * 4;
  const numChannelsToRemove = finalChannels - originalChannels;

  if (numChannelsToRemove === 0) {
    // Still need to handle texture spatial padding
    return dataWithPaddedChannels.slice(0, originalElementCount);
  }

  const dataWithoutPaddedChannels = new Float32Array(originalElementCount);

  let outputIndex = 0; // Index for the output array
  // Iterate through the original data
  for (let index = 0; index < dataWithPaddedChannels.length; index++) {
    // Calculate current channel position in the stride
    const currentChannelPosition = index % finalChannels;

    // Only copy data if the channel position is less than the number of original channels
    if (currentChannelPosition < originalChannels) {
      dataWithoutPaddedChannels[outputIndex] = dataWithPaddedChannels[index];
      outputIndex++;
    }
  }

  return dataWithoutPaddedChannels;
}

export const conv2dWeightsTransform = (
  weights: TFTensor
): TFTensor => {
  weights = convertHWCNToNHWC(weights);

  // const zShapeOriginal = weights.shape[3];

  // // @ts-ignore
  // weights = weights.reshape([weights.shape[2], weights.shape[1], weights.shape[0] * weights.shape[3]]);

  const zShape = weights.shape[3];
  const xShape = weights.shape[2];
  const yShape = weights.shape[1];
  const numFilters = weights.shape[0];

  if (!xShape || !yShape || !zShape) {
    throw new Error("The shape of the weights tensor is invalid. Got: " + weights.shape);
  }

  if (zShape % 4 !== 0) {
    throw new Error(`The number of filters in the Conv2D layer must be divisible by 4. Found ${zShape} filters.`);
  }

  // Stack the weights along the x axis. 
  // Also, convert the weights format to PSEUDO channels first: CHWN
  // - PSEUDO because the last N dimension should have 4 channel elements for each element.
  const weightsRaw = weights.dataSync();
  const newWeights = new Float32Array(weights.size);
  let newIndex = 0;

  // Helper function to calculate the linear index from the multi-dimensional index
  const getIndex = (n: number, y: number, x: number, z: number, numFilters: number, yShape: number, xShape: number, zShape: number) => {
    return (((n * yShape + y) * xShape + x) * zShape + z);
  };

  // XXX: WARNING: The order of the loops is VERY important here.
  for (let z = 0; z < zShape; z += 4) { // Processing 4 z-values at a time
    for (let y = 0; y < yShape; y++) {
      for (let x = 0; x < xShape; x++) {
        for (let n = 0; n < numFilters; n++) {
          for (let i = 0; i < 4; i++) {
            const zOffset = z + i;

            if (zOffset >= zShape) {
              // This should never happen - as the z axis should be padded to be a multiple of 4.
              throw new Error(`zOffset is out of bounds. Got: ${zOffset}, ${zShape}`);
            }

            const index = getIndex(n, y, x, zOffset, numFilters, yShape, xShape, zShape);
            newWeights[newIndex++] = weightsRaw[index];
          }
        }
      }
    }
  }

  let newXShape = xShape * numFilters;
  let newYShape = yShape;

  //
  //// TODO: The newYShape logic is not implemented yet on the shader side.
  if (newXShape > maxTextureDim) {
    const newNumFilters = Math.ceil(newXShape / maxTextureDim);
    newXShape = newXShape / newNumFilters;
    newYShape = yShape * newNumFilters;
  }

  if (newXShape > maxTextureDim || newYShape > maxTextureDim || zShape > maxTextureDim || numFilters > maxTextureDim) {
    throw new Error(`The shape of the weights tensor is too large. Got: ${newXShape}, ${yShape}, ${zShape}, ${numFilters}`);
  }

  const newWeightsTensor = tf.tensor(newWeights, [newYShape, newXShape, zShape], "float32");

  return newWeightsTensor;
}

// XXX: WARNING: The following function MODIFIES the input tensor.
export const convertHWCNToNHWC = (weights: TFTensor): TFTensor => {
  // Puts the number of filters dimension first (instead of last) i.e. goes from HWCN to NHWC.
  weights = weights.transpose([3, 0, 1, 2]);
  weights = padChannels(weights, "conv2dWeightsTransform_dont_know_the_node_name");

  // I don't think this return is necessary, but keeping it for now.
  return weights;
}

export const biasWeightsTransform = (weights: TFTensor): TFTensor => {
  // Just pad with zeros so the shape is divisible by 4.
  const elementCount = getElementCount(weights.shape);
  const numZerosToAdd = 4 - (elementCount % 4);

  // FIXME:
  // Hmmm I don't think this works for bias w shapes that are not 1D?!???
  // weights = tf.concat([weights, tf.zeros([numZerosToAdd])]);

  return weights;
}

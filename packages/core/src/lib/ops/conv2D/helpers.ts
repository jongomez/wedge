
import type { Conv2DParams, WebGLDataTextureArray } from "../../types";


// Padding references:
// https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
// https://stackoverflow.com/questions/68035443/what-does-padding-same-exactly-mean-in-tensorflow-conv2d-is-it-minimum-paddin
// https://stackoverflow.com/questions/48491728/what-is-the-behavior-of-same-padding-when-stride-is-greater-than-1
// https://github.com/tensorflow/tensorflow/issues/46980
// https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/conv_util.ts#L447-L458
export const getConvPadding = (
  output: WebGLDataTextureArray,
  input: WebGLDataTextureArray,
  kernelX: number,
  kernelY: number,
  strideX: number,
  strideY: number,
  pad: Conv2DParams["pad"]): [number, number] => {
  let padX = 0;
  let padY = 0;

  if (pad === "same") {
    // Hmmm why doesn't this one work? And the bottom one (taken from the tfjs repo) does?
    // if (input.shape[0] % strideX === 0) {
    //   padX = Math.max(kernelX - strideX, 0);
    // } else {
    //   padX = Math.max(kernelX - (input.shape[0] % strideX), 0);
    // }

    // if (input.shape[1] % strideY === 0) {
    //   padY = Math.max(kernelY - strideY, 0);
    // } else {
    //   padY = Math.max(kernelY - (input.shape[1] % strideY), 0);
    // }

    const outputWidth = output.originalShape[0];
    const outputHeight = output.originalShape[1];
    const inputWidth = input.originalShape[0];
    const inputHeight = input.originalShape[1];

    const padAlongX = Math.max(0, (outputWidth - 1) * strideX + kernelX - inputWidth);
    const padAlongY = Math.max(0, (outputHeight - 1) * strideY + kernelY - inputHeight);

    padX = Math.floor(padAlongX / 2);
    padY = Math.floor(padAlongY / 2);
  } else if (pad !== "valid") {
    throw new Error("Invalid padding type: " + pad);
  }

  return [padX, padY];
}
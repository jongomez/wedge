import * as tfOriginal from '@tensorflow/tfjs';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const tf = tfOriginal as any;

export function compareTensors(tensorA: any, tensorB: any, tolerance: number): boolean {
  // Ensure the shapes of the tensors are identical
  if (!tf.util.arraysEqual(tensorA.shape, tensorB.shape)) {
    console.log("Tensors have different shapes:", tensorA.shape, "vs", tensorB.shape);
    return false;
  }

  // Calculate the absolute difference between the tensors
  const difference = tf.abs(tensorA.sub(tensorB));

  // Calculate the maximum difference to check against the tolerance
  const maxDifference = difference.max().dataSync()[0];

  console.log("TensorA:", tensorA.dataSync());
  console.log("TensorB:", tensorB.dataSync());

  if (isNaN(maxDifference)) {
    console.error("Tensors have NaN values.");
    return false;
  } else if (maxDifference > tolerance) {
    console.error(`Tensors differ by a maximum of ${maxDifference}, which is greater than the tolerance of ${tolerance}.`);
    // console.log("TensorA:", tensorA.dataSync());
    // console.log("TensorB:", tensorB.dataSync());

    // Optional: Print the differences if needed for debugging
    // difference.print();
    return false;
  } else {
    console.log("Tensors are similar within the specified tolerance.");
  }

  const meanDifference = difference.max().dataSync()[0];
  console.log(`Mean absolute difference between tensors: ${meanDifference}`);

  return true;
}


export function createSequentialTensor(shape: number[], start = 1): any {
  // Calculate the total number of elements in the final tensor.
  const totalElements = shape.reduce((a, b) => a * b, 1);

  // Create a tensor with sequential values
  const sequentialValues = tf.range(start, totalElements + start).toFloat();

  // Reshape the tensor to match the shape of the convolutional layer's weights
  const sequentialWeights = sequentialValues.reshape(shape);

  return sequentialWeights;
}

export function createWebGLContext(): WebGL2RenderingContext {
  // Use offscreen canvas.
  const offscreenCanvas = new OffscreenCanvas(1, 1);
  const gl = offscreenCanvas.getContext("webgl2");

  if (!gl) {
    throw new Error("Unable to create WebGL context");
  }

  return gl;
}

import * as tf from '@tensorflow/tfjs';
import CryptoJS from 'crypto-js';
import { TestMap, TestState, TestType } from './types';
import { windowUpdateTestMap } from './window';

export function compareTensors(tensorA: tf.Tensor, tensorB: tf.Tensor, tolerance: number): boolean {
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


export function createSequentialTensor(shape: number[], start = 1): tf.Tensor {
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

function generateHash(input: string): string {
  const hashOutput = CryptoJS.SHA256(input).toString(CryptoJS.enc.Hex);
  return hashOutput;
}

export function toValidDOMId(id: string): string {
  // Remove invalid characters
  let finalId = id.replace(/[^a-zA-Z0-9\-_:.]/g, '-');
  finalId = finalId.replace(/--+/g, '-'); // Remove multiple consecutive hyphens
  finalId = finalId.replace(/^-|-$/g, ''); // Remove leading and trailing hyphens
  finalId = finalId.toLowerCase();

  // Ensure ID does not start with a digit, hyphen, or period
  if (/^[0-9\-:.]/.test(finalId)) {
    finalId = '_' + finalId;
  }

  // Hash the original id, to make sure the replacements we made don't generate duplicate IDs.
  const hash = generateHash(id);
  finalId = `${finalId}-${hash.substring(0, 6)}`;

  return finalId;
}

export const updateTestMapWithTest = (
  testId: string,
  test: TestType,
  setTestMap: React.Dispatch<React.SetStateAction<TestMap>>) => {
  setTestMap((prevTestMap) => {
    const newTestMap = new Map(prevTestMap);
    newTestMap.set(testId, test);
    windowUpdateTestMap(newTestMap);

    return newTestMap;
  });
}

export async function executeTest(
  test: TestType,
  testId: string,
  setCurrentIndex: React.Dispatch<React.SetStateAction<number>>,
  setTestMap: React.Dispatch<React.SetStateAction<TestMap>>,
) {
  let resultInfo = '';
  let finalState: TestState = 'Pending';

  // Execute the test and handle both success and failure
  try {
    let iframeContentWindow: Window = window;
    let iframeContentDocument: Document = document;

    if (test.hasChildren) {
      [iframeContentWindow, iframeContentDocument] = await getIframeWindowAndDocument(testId);
    }

    // XXX: test.fn may not be async and not return a promise. "await" will still work though.
    await test.fn(iframeContentWindow, iframeContentDocument);
    console.log(`Test "${test.title}" passed ✓`);
    finalState = "Success";
  } catch (error) {
    let errorMessage = `Test "${test.title}" failed: `;
    errorMessage += (error instanceof Error) ? error.message : JSON.stringify(error);
    console.error(errorMessage);
    resultInfo = errorMessage;
    finalState = "Fail";
  }

  updateTestMapWithTest(testId, { ...test, state: finalState, resultInfo }, setTestMap);
  setCurrentIndex(curr => curr + 1);
};


export const waitFor = async (condition: () => boolean, maxWaitMilliseconds = 5000): Promise<void> => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now(); // Record the start time

    const interval = setInterval(() => {
      if (condition()) {
        clearInterval(interval);
        resolve();
      } else if (Date.now() - startTime > maxWaitMilliseconds) { // Check if the max wait time has passed
        clearInterval(interval);
        reject(new Error('waitFor - Timed out waiting for condition to be true'));
      }
    }, 100);
  });
}

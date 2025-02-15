import * as tf from '@tensorflow/tfjs';

// Define a custom layer to add a scalar to the input tensor
/*
class AddScalarLayer extends tf.layers.Layer {
  scalarToAdd: number;

  constructor(scalar: number, name: string) {
    const config: LayerArgs = {
      inputShape: [256, 256, 3],
      name,
    }

    super(config);

    this.scalarToAdd = scalar;
  }

  call(input: tf.Tensor) {
    return tf.tidy(() => {
      return input.add(this.scalarToAdd);
    });
  }

  // Define the output shape of the layer (same as the input shape in this case)
  computeOutputShape(inputShape: tf.Shape) {
    return inputShape;
  }

  // Serialization for saving and loading the model
  getConfig(): tf.serialization.ConfigDict {
    const config: tf.serialization.ConfigDict = super.getConfig();
    Object.assign(config, { scalar: this.scalarToAdd });
    return config;
  }

  static get className(): string {
    return 'MyAddScalarLayer';
  }
}
*/

// Function to create the model
export function createAllConvModel(numConvLayers: number): tf.LayersModel {
  // Register the custom layer for serialization purposes
  // tf.serialization.registerClass(AddScalarLayer);

  const input: tf.SymbolicTensor = tf.input({ shape: [256, 256, 3] });
  /*
    // Add the first scalar (e.g., 1) using the custom layer
    let result = new AddScalarLayer(1, "firstAdd").apply(input) as tf.SymbolicTensor;
  
    // Add the second scalar (e.g., another scalar) using another custom layer
    result = new AddScalarLayer(2, "secondAdd").apply(result) as tf.SymbolicTensor;
  */

  const convLayerSettings = {
    filters: 1,
    kernelSize: 4,
    strides: 4,
    padding: "same" as const,
  };

  // Add 2d conv layer
  let result = tf.layers.conv2d({ ...convLayerSettings, filters: 40 }).apply(input) as tf.SymbolicTensor;

  for (let layerNum = 0; layerNum < numConvLayers - 1; layerNum++) {
    // Add another 2d conv layer
    result = tf.layers.conv2d(convLayerSettings).apply(result) as tf.SymbolicTensor;
  }

  // Create the model
  const model: tf.LayersModel = tf.model({ inputs: input, outputs: result });

  return model;
}

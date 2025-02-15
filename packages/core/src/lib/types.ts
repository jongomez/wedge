import { Node } from "@tensorflow/tfjs-converter/dist/operations/types";

// Include a type for single
export type DataFormat = "NHWC" | "HWC" | "VEC";

export type InitWebGLReturn = {
  canvas: OffscreenCanvas;
  gl: WebGL2RenderingContext;
  maxColorAttachments: number;
}

export type ProgramInfo = {
  program: WebGLProgram;
  attribLocations: {
    vertexPosition: number;
  };
}

// FBO - Frame Buffer Object.
export type TexturesAndOutputFBO = {
  inputTextures: WebGLTexture[],
  outputTextures: WebGLTexture[],
  weightTextures: WebGLTexture[],
  frameBuffer: WebGLFramebuffer,
}

export type OutputShape = number[];
export type InputShape = number[];

export type ModelType = "GraphModel" | "LayersModel" | "onnx" | "tflite";

export type WedgeOptions = {
  // XXX: NOTE: these width and height values may not be necessary? As we're drawing to a texture, not the canvas.
  canvasWidth: number;
  canvasHeight: number;
  viewportMaxSize: number;

  hasBatchDimension: boolean;

  transformations: {
    padChannels: boolean;
  }
  renderTargetBreakpoints: {
    outputTextureElementCount: number;
    numberOfRenderTargets: number;
  }[];
}

export type WebGLDataBase = {
  nodeName: string;
  uniformName: string;
};

export type WebGLDataNonTexture = WebGLDataBase & {
  webGLType: "float" | "vec2" | "vec3" | "vec4";
  data: number[];
};

// Note: For now, webGL textures do not have a data field. This is because textures can be too large.
export type WebGLDataTextureBase = WebGLDataBase & {
  texture: WebGLTexture;

  // Original dimensions:
  originalShape: number[];
  originalElementCount: number;

  // Texture dimensions:
  RGBATextureShape: number[];
  RGBATextureElementCount: number;
}

export type WebGLDataTexture = WebGLDataTextureBase & {
  webGLType: "sampler2D";
};

export type WebGLDataTextureArray = WebGLDataTextureBase & {
  webGLType: "sampler2DArray";
};

export type WebGLData = WebGLDataNonTexture | WebGLDataTexture | WebGLDataTextureArray;

export type ArithmeticOpName = "AddV2" | "Mul"

export type SingleInputBasicOpName = "Relu"

// Here I'm creating 2 types for op names, as I feel some of the graph model ops really shouldn't be called ops.
// For example, Placeholder and Const. These are more like "data" nodes.
export type OpName = ArithmeticOpName
  | SingleInputBasicOpName
  | "Conv2D"
  | "_FusedConv2D"
  | "DepthwiseConv2D"
  | "DepthwiseConv2dNative"
  | "FusedDepthwiseConv2dNative"
  | "ResizeBilinear"
  | "NotSupported"

export type GraphModelOpNames = OpName | "Placeholder" | "Const"

export type LayersModelLayerClass = "Conv2D"
  | "DenseLayer"
  | "InputLayer"
  | "MaxPooling2DLayer"
  | "ReLU"
  | "DepthwiseConv2D"
  | "Add"

export type Conv2DParams = {
  strides: number[];
  pad: "same" | "valid";
  kernelX: number;
  kernelY: number;
  kernelDepth: number;
  numFilters: number;
  activation: 'relu' | null;
  hasBias: boolean;
}

export type ResizeBilinearParams = {
  alignCorners: boolean;
  halfPixelCenters: boolean;
  dtype: "float32" | "int32";
  size: number;
}

export type OpParams = Conv2DParams | ResizeBilinearParams;

// Operation nodes have a corresponding WebGL program - with vertex and fragment shaders.
export type WebGLOpNode = {
  node: Node;
  // Operations have input(s) and output(s) - and possibly weight(s).
  inputs: (WebGLData | null)[];
  output: WebGLDataTextureArray | null;
  weights: WebGLDataTextureArray[];
  opParams: OpParams | null;
  type: OpName;
  fsSource: string;
};

export type WebGLOpNodeWithProgram = {
  opNode: WebGLOpNode;
  programInfo: ProgramInfo;
}

export type WebGLOpNodeWithProgramMap = Map<string, WebGLOpNodeWithProgram>;

// export type WebGLData = {
//   isOperation: boolean;
//   shape: number[];
//   texture: WebGLTexture | null;
//   scalar: number | null;
// }

export type WebGLOpNodeMap = Map<string, WebGLOpNode>;
export type NodeWebGLDataMap = Map<string, WebGLData>;

export type CustomShapeUpdate = (
  shapeWithPaddedChannels: number[],
  currentWidth: number,
  currentHeight: number,
  numRenderTargets: number,
  nodeName: string) => [number, number, number, number]

export type NamedArrayBufferViewMap = {
  [key: string]: ArrayBufferView;
};

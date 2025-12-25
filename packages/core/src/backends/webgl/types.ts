import { Graph } from "../../graph/types";
import { OpParams, Ops } from "../../ops/types";
import { TensorWebGL } from "../../tensor/TensorWebGL";
import { DataArray } from "../../tensor/types";

export type Vector3String = {
  x: string;
  y: string;
  z: string;
};

export type ModelConfig = {
  backend: "webgl";
  runtime: "tfjs" | "mediapipe" | "tfjs-tflite" | "wedge";
  url: string;
}

export type GlobalState = {
  targetFPS: number;
  cameraWidth: number;
  cameraHeight: number;
  isVideoStreamLoaded: boolean;
  isCameraCanvasLoaded: boolean;
  modelConfig: ModelConfig;
}

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

export type WebGLDataNodeBase = {
  nodeName: string;
  uniformName: string;
};

export type WebGLDataNodeNonTexture = WebGLDataNodeBase & {
  webGLType: "float" | "vec2" | "vec3" | "vec4";
  data: number[];
};

// Note: For now, webGL textures do not have a data field. This is because textures can be too large.
export type WebGLDataNodeTextureBase = WebGLDataNodeBase & {
  texture: WebGLTexture;

  // Original dimensions:
  originalShape: number[];
  originalElementCount: number;

  // Texture dimensions:
  RGBATextureShape: number[];
  RGBATextureElementCount: number;
}

export type WebGLDataNodeTexture = WebGLDataNodeTextureBase & {
  webGLType: "sampler2D";
};

export type WebGLDataNodeTextureArray = WebGLDataNodeTextureBase & {
  webGLType: "sampler2DArray";
};

// Union type for WebGL data nodes (defined early for use in WebGLOpNode)
export type WebGLData = WebGLDataNodeNonTexture | WebGLDataNodeTexture | WebGLDataNodeTextureArray;
export type WebGLDataNonTexture = WebGLDataNodeNonTexture;
export type WebGLDataTexture = WebGLDataNodeTexture;
export type WebGLDataTextureArray = WebGLDataNodeTextureArray;

export type ArithmeticOpName = "AddV2" | "Mul"

export type SingleInputBasicOpName = "Relu" | "Relu6" | "Sigmoid"

export type PadOpName = "Pad" | "PadV2" | "MirrorPad"

// Here I'm creating 2 types for op names, as I feel some of the graph model ops really shouldn't be called ops.
// For example, Placeholder and Const. These are more like "data" nodes.
export type OpName = ArithmeticOpName
  | SingleInputBasicOpName
  | PadOpName
  | "Conv2D"
  | "_FusedConv2D"
  | "DepthwiseConv2D"
  | "DepthwiseConv2dNative"
  | "FusedDepthwiseConv2dNative"
  | "ResizeBilinear"
  | "Reshape"
  | "MaxPool"
  | "NotSupported"

export type GraphModelOpNames = OpName | "Placeholder" | "Const"

export type LayersModelLayerClass = "Conv2D"
  | "DenseLayer"
  | "InputLayer"
  | "MaxPooling2D"
  | "ReLU"
  | "DepthwiseConv2D"
  | "Add"
  | "ZeroPadding2D"
  | "Reshape"
  | "Activation"


// Operation nodes have a corresponding WebGL program - with vertex and fragment shaders.
export type WebGLOpNode = {
  node: any; // Node type from @tensorflow/tfjs-converter
  // Operations have input(s) and output(s) - and possibly weight(s).
  inputs: (WebGLData | null)[];
  output: WebGLDataNodeTextureArray | null;
  weights: WebGLDataNodeTextureArray[];
  opParams: OpParams | PadParams | ReshapeParams | MaxPoolParams | null;
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

export type WebGLOpInput = TensorWebGL | DataArray;
export type WebGLOpOutput = TensorWebGL;

export type WebGLOps = Ops<WebGLOpInput, WebGLOpOutput>
export type GraphWebGL = Graph;

// Operation parameter types
export type Conv2DParams = {
  strides: number[];
  pad: "same" | "valid";
  kernelX: number;
  kernelY: number;
  kernelDepth: number;
  numFilters: number;
  activation: 'relu' | 'linear' | null;
  hasBias: boolean;
}

export type DepthwiseConv2DParams = Conv2DParams;

export type ResizeBilinearParams = {
  alignCorners: boolean;
  halfPixelCenters: boolean;
  dtype: "float32" | "int32";
  size: number;
}

export type PadParams = {
  paddings: number[][];  // [[before, after], ...] for each dimension
  constantValue: number;
}

export type ReshapeParams = {
  newShape: number[];
}

export type MaxPoolParams = {
  poolSize: number[];
  strides: number[];
  pad: "same" | "valid";
}

export type NodeWebGLDataMap = Map<string, WebGLData>;
export type WebGLOpNodeMap = Map<string, WebGLOpNode>;

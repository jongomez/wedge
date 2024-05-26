import { GraphModel } from "@tensorflow/tfjs-converter";
import { NNShaders } from "./nnshaders/NNShaders";

export type Vector3String = {
  x: string;
  y: string;
  z: string;
};

export type ModelConfig = {
  backend: "webgl";
  runtime: "tfjs" | "mediapipe" | "tfjs-tflite" | "nnshaders";
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


export type Model = NNShaders | GraphModel;
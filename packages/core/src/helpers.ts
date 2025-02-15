import { ModelType } from "./lib/types";
import { ModelConfig } from "./types";

export function getModelType(modelConfig: ModelConfig): ModelType {
  const { url } = modelConfig;

  if (url.endsWith(".tflite")) {
    return "tflite";
  } else if (url.endsWith(".json")) {
    return "GraphModel";
  } else {
    throw new Error(`Unknown model type for url: ${url}`);
  }

}


// import { Color3 } from "babylonjs";
// import { SupportedModels } from "./pose_estimation/params";
import { Vector3String } from "./types";

// Ultimate red blue green colors:
// https://www.reddit.com/r/Design/comments/3jrc0b/comment/curpe8o/?utm_source=share&utm_medium=web3x
// With a slightly different red though.

// export const originalRed = new Color3(0.94, 0.29, 0.14);
// #EC331A
// export const red = new Color3(0.93, 0.2, 0.1);
// // #58B129
// export const green = new Color3(0.35, 0.69, 0.16);
// // #2193F2
// export const blue = new Color3(0.13, 0.58, 0.95);

export const DEBUG = true;

export const defaultVector3String: Vector3String = {
  x: "",
  y: "",
  z: "",
};

export const webGLCanvasId = "webGLCanvas";
export const cameraCanvasId = "cameraCanvas";
export const videoElementId = "videoElement";

export const outputShapeX = 128;
export const outputShapeY = 128;
export const outputShapeZ = 256;

// export const currentModel: SupportedModels = "MyModel";

export const maxTextureDim = 2048;
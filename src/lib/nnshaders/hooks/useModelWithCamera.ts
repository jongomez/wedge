// Register WebGL backend.
import { useEffect } from "react";
import { NNShaders } from "../NNShaders";

import { GlobalState, ModelConfig } from "@/lib/types";
import { Render2DPoints } from "../canvas/Render2DPoints";
import { Render3DPoints } from "../canvas/Render3DPoints";
import { handleTests } from "./tests/tests";
import { useLoadModel } from "./useLoadModel";
import { updateFPS } from "./utils";

let isEstimatingPose = false;
let isCameraRolling = false;

const checkIfCameraIsRolling = (video: HTMLVideoElement): boolean => {
  isCameraRolling = video.currentTime > 0 && !video.paused && !video.ended && video.readyState > 2;
  return isCameraRolling;
}

export const useModelWithCamera = (
  cameraCanvas: HTMLCanvasElement | null,
  video: HTMLVideoElement | null,
  isVideoElementLoaded: boolean,
  isCameraCanvasLoaded: boolean,
  isVideoStreamLoaded: boolean,
  modelConfig: ModelConfig,
  isTest = false
  scene?: Scene | null,
) => {
  // Step 1 - load the model.
  const model = useLoadModel(modelConfig);

  //
  //// Step 2 - Use the model.
  useEffect(() => {
    // console.log("isVideoElementLoaded", isVideoElementLoaded);
    // console.log("detector", detector);
    // console.log("scene", scene);
    // console.log("isCameraCanvasLoaded", isCameraCanvasLoaded);
    // console.log("isVideoStreamLoaded", isVideoStreamLoaded);

    if (!isVideoElementLoaded || !isCameraCanvasLoaded || !isVideoStreamLoaded || !scene) {
      return;
    }

    // if(!detector) {
    if (!myPoseEstimator) {
      return;
    }

    if (!cameraCanvas) {
      console.error("cameraCanvas is null, but isCameraCanvasLoaded is true - returning early.");
      return;
    }

    if (!video) {
      console.error("video is null, but isVideoElementLoaded is true - returning early.");
      return;
    }

    if (modelResult) {
      console.log("modelResult already exists - returning early.");
      return;
    }

    const cameraCanvas2DContext = cameraCanvas.getContext("2d", { willReadFrequently: true });

    if (!cameraCanvas2DContext) {
      throw Error("Could not get 2D context from cameraCanvas - returning early.");
    }

    const render2DPoints = new Render2DPoints(cameraCanvas2DContext);
    const render3DPoints = new Render3DPoints(scene);
    const offscreenCanvas = new OffscreenCanvas(256, 256);

    if (isTest) {
      handleTests(
        cameraCanvas,
        offscreenCanvas,
        render2DPoints,
        render3DPoints,
        modelResult
      );
    } else {
      if (!checkIfCameraIsRolling(video)) {
        console.log("Camera is not rolling yet. Waiting for it to start...");
      }

      renderResultLoop(
        // detector,
        myPoseEstimator,
        cameraCanvas2DContext,
        offscreenCanvas,
        video,
        render2DPoints,
        render3DPoints,
        modelResult
      );
    }
  }, [myPoseEstimator,
    cameraCanvas,
    isVideoStreamLoaded,
    modelResult,
    scene,
    isVideoElementLoaded,
    isCameraCanvasLoaded,
    video,
    isTest]);
};

let predictionTime = 0;

// Render loop inspired by:
// https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/demos/live_video/src/index.js#L193-L201
async function renderResultLoop(
  // detector: PoseDetector,
  myPoseEstimator: PoseEstimator | ImageClassifier | NNShaders,
  cameraCanvas2DContext: CanvasRenderingContext2D,
  offscreenCanvas: OffscreenCanvas,
  video: HTMLVideoElement,
  render2DPoints: Render2DPoints,
  render3DPoints: Render3DPoints,
  modelResult: GlobalState["modelResult"]
) {
  const startTime = performance.now()
  updateFPS(startTime);
  // updateLatency(startTime);

  if (!isCameraRolling) {
    if (checkIfCameraIsRolling(video)) {
      isCameraRolling = true;
      console.log("Camera is rolling!")
    }
  }

  if (isEstimatingPose || !isCameraRolling) {
    requestAnimationFrame(() =>
      renderResultLoop(
        // detector,
        myPoseEstimator,
        cameraCanvas2DContext,
        offscreenCanvas,
        video,
        render2DPoints,
        render3DPoints,
        modelResult
      )
    );

    return;
  }

  isEstimatingPose = true;

  const estimationConfig = { flipHorizontal: true };
  const timestamp = performance.now();

  // Draw stuff on top of the camera canvas aka 2D canvas.
  render2DPoints.draw(video, modelResult || [], false);

  try {
    // If you want to use WebGPU API - estimatePosesGPU is not available here? I don't get it.
    // On their demo they use it (although they don't use typescript):
    // https://github.com/tensorflow/tfjs-models/blob/master/pose-detection/demos/live_video/src/index.js#L167-L173
    // They then use another renderer - the RendererWebGPU
    // const poses = await detector.estimatePoses(cameraCanvas, estimationConfig, timestamp);

    let keypoints: Keypoint[] = [];
    if (typeof (myPoseEstimator as PoseEstimator).estimatePoses === "function") {
      // tfjs runtime:
      keypoints = await (myPoseEstimator as PoseEstimator).estimatePoses(video, timestamp);
    } else if (typeof (myPoseEstimator as NNShaders).predict === "function") {
      // NNShader runtime:
      // const dummyTensor = ones([1, 256, 256, 4]);
      // const dummyTensorData = dummyTensor.dataSync();
      const videoDataUint8ClampedArray = cameraCanvas2DContext.getImageData(0, 0, 256, 256).data;

      if (!videoDataUint8ClampedArray) {
        throw new Error("Could not get video data from canvas.");
      }

      const videoDataFloat32Array = new Float32Array(videoDataUint8ClampedArray);

      (myPoseEstimator as NNShaders).predict([videoDataFloat32Array]);

      // console.log("NNShaders runtime bruh")
    } else {
      // mediapipe runtime:
      // FIXME: TODO: The timestamp should be the frame's timestamp, not the current time.
      const timestamp = performance.now();
      // const img = await cropAndScaleImage(cameraCanvas, offscreenCanvas);
      // NOTE: classifyForVideo and regular classify are very similar:
      // https://github.com/google/mediapipe/blob/dc808842a73939d123ae02e497c04073fd58bc32/mediapipe/tasks/web/vision/core/vision_task_runner.ts#L113-L135
      // const regionOfInterest = {
      //   right: 255,
      //   left: 0,
      //   top: 0,
      //   bottom: 255,
      // }

      // Extract image from video:


      const classifierResult = await (myPoseEstimator as ImageClassifier).classifyForVideo(video, timestamp);
      // keypoints = categoriesToKeypoints(classifierResult.classifications[0].categories);
    }

    modelResult = keypoints;

    // Draw stuff on top of the 3D canvas AKA the actual game canvas.
    render3DPoints.draw(modelResult);
  } catch (error) {
    console.error("Error in pose estimation:", error);
  } finally {
    isEstimatingPose = false;
  }

  predictionTime = performance.now() - startTime;

  if (predictionTime === 0) {
    // console.log("Time to FIRST EVER PREDICTION:", predictionTime);
  } else {
    // console.log("Time to prediction:", predictionTime);
  }

  let rafId = requestAnimationFrame(() =>
    renderResultLoop(
      // detector,
      myPoseEstimator,
      cameraCanvas2DContext,
      offscreenCanvas,
      video,
      render2DPoints,
      render3DPoints,
      modelResult
    )
  );
}

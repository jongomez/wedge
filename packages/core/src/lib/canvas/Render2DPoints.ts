import * as poseDetectionUtils from "./utils";

import { CameraVideoHeight, CameraVideoWidth } from "@/components/Main";
import assert from "assert";
import { currentModel, outputShapeX } from "../constants";
import { Keypoint } from "../types";
import * as params from "./params";

// #ffffff - White
// #800000 - Maroon
// #469990 - Malachite
// #e6194b - Crimson
// #42d4f4 - Picton Blue
// #fabed4 - Cupid
// #aaffc3 - Mint Green
// #9a6324 - Kumera
// #000075 - Navy Blue 
// #f58231 - Jaffa
// #4363d8 - Royal Blue
// #ffd8b1 - Caramel
// #dcbeff - Mauve
// #808000 - Olive
// #ffe119 - Candlelight
// #911eb4 - Seance
// #bfef45 - Inchworm
// #f032e6 - Razzle Dazzle Rose
// #3cb44b - Chateau Green
// #a9a9a9 - Silver Chalice
const COLOR_PALETTE: string[] = [
  "#ffffff",
  "#800000",
  "#469990",
  "#e6194b",
  "#42d4f4",
  "#fabed4",
  "#aaffc3",
  "#9a6324",
  "#000075",
  "#f58231",
  "#4363d8",
  "#ffd8b1",
  "#dcbeff",
  "#808000",
  "#ffe119",
  "#911eb4",
  "#bfef45",
  "#f032e6",
  "#3cb44b",
  "#a9a9a9",
];



export class Render2DPoints {
  private context2D: CanvasRenderingContext2D;
  private videoWidth: number;
  private videoHeight: number;

  constructor(canvas2DContext: CanvasRenderingContext2D) {
    this.context2D = canvas2DContext;
    this.videoWidth = CameraVideoWidth;
    this.videoHeight = CameraVideoHeight;
    // this.flip(this.videoWidth, this.videoHeight);
  }

  private flip(videoWidth: number, videoHeight: number): void {
    this.context2D.translate(videoWidth, 0);
    this.context2D.scale(-1, 1);
  }

  public draw(
    video: CanvasImageSource,
    keypoints: Keypoint[],
    isModelChanged: boolean
  ): void {
    this.drawcontext2D(video);

    if (!keypoints || !keypoints.length || isModelChanged) {
      return
    }

    const keypointsAdjusted: Keypoint[] = []

    const canvasWidth = this.context2D.canvas.width
    const canvasHeight = this.context2D.canvas.height

    assert(canvasWidth === canvasHeight, "Canvas width and height should be equal")

    const ratio = canvasWidth / outputShapeX

    for (const keypoint of keypoints) {
      keypointsAdjusted.push({
        x: keypoint.x * ratio,
        y: keypoint.y * ratio,
        z: keypoint.z
      })
    }

    this.drawKeypoints2D(keypointsAdjusted);
    this.drawSkeletonLines2D(keypointsAdjusted);
  }

  private drawcontext2D(video: CanvasImageSource): void {
    this.context2D.drawImage(video, 0, 0, this.videoWidth, this.videoHeight);
  }

  public clearcontext2D(): void {
    this.context2D.clearRect(0, 0, this.videoWidth, this.videoHeight);
  }

  private drawKeypoints2D(keypoints: Keypoint[]): void {
    const keypointInd = poseDetectionUtils.getKeypointIndexBySide(
      currentModel
    );
    this.context2D.fillStyle = "Red";
    this.context2D.strokeStyle = "White";
    this.context2D.lineWidth = params.DEFAULT_LINE_WIDTH;

    for (const i of keypointInd.middle) {
      this.drawKeypoint2D(keypoints[i]);
    }

    this.context2D.fillStyle = "Green";
    for (const i of keypointInd.left) {
      this.drawKeypoint2D(keypoints[i]);
    }

    this.context2D.fillStyle = "Orange";
    for (const i of keypointInd.right) {
      this.drawKeypoint2D(keypoints[i]);
    }
  }

  private drawKeypoint2D(keypoint: Keypoint): void {
    const circle = new Path2D();
    circle.arc(keypoint.x, keypoint.y, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
    this.context2D.fill(circle);
    this.context2D.stroke(circle);
  }

  /**
   * Draw the skeleton of a body on the video.
   * @param keypoints A list of keypoints.
   */
  private drawSkeletonLines2D(keypoints: Keypoint[]): void {
    // Each poseId is mapped to a color in the color palette.
    const color = "White";
    this.context2D.fillStyle = color;
    this.context2D.strokeStyle = color;
    this.context2D.lineWidth = 1;

    poseDetectionUtils.getAdjacentPairs(currentModel).forEach(([i, j]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];

      this.context2D.beginPath();
      this.context2D.moveTo(kp1.x, kp1.y);
      this.context2D.lineTo(kp2.x, kp2.y);
      this.context2D.stroke();
    });
  }
}

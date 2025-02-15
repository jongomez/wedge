import * as poseDetectionUtils from "./utils";

import {
  Color3,
  LinesMesh,
  Mesh,
  MeshBuilder,
  Scene,
  StandardMaterial
} from "babylonjs";
import { currentModel } from "../constants";


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
type Keypoint = {
  x: number;
  y: number;
  z: number;
};

export class Render3DPoints {
  private scene: Scene;
  private sphereMap: Map<number, Mesh>; // Store spheres with their index as the key
  private lineMap: Map<string, LinesMesh>; // Store lines with a unique identifier as the key

  constructor(scene: Scene) {
    this.scene = scene;
    this.sphereMap = new Map();
    this.lineMap = new Map();

  }

  public draw(keypoints: Keypoint[]): void {
    // Clear spheres that are not used in the current frame
    // this.clearUnusedSpheres(poses);
    // this.clearUnusedLines(poses);
    this.drawKeypoints3D(keypoints);
  }

  private clearUnusedSpheres(keypoints: Keypoint[]): void {
    const activeIndices = new Set<number>();

    keypoints.forEach((_, index) => activeIndices.add(index));

    this.sphereMap.forEach((sphere, index) => {
      if (!activeIndices.has(index)) {
        sphere.dispose();
        this.sphereMap.delete(index);
      }
    });
  }

  private clearUnusedLines(keypoints: Keypoint[]): void {
    const activeLines = new Set<string>();
    const connections = poseDetectionUtils.getAdjacentPairs(
      currentModel
    );


    for (const pair of connections) {
      if (keypoints[pair[0]] && keypoints[pair[1]]) {
        const lineKey = `line${pair[0]}-${pair[1]}`;
        activeLines.add(lineKey);
      }
    }

    this.lineMap.forEach((line, key) => {
      if (!activeLines.has(key)) {
        line.dispose();
        this.lineMap.delete(key);
      }
    });
  }

  private drawKeypoints3D(keypoints: Keypoint[]): void {
    const keypointInd = poseDetectionUtils.getKeypointIndexBySide(
      currentModel
    );

    // // Add cubes for reference:
    // const rootCube = MeshBuilder.CreateBox("rootCube", { size: 0.5 }, this.scene);
    // const aboveRootCube = MeshBuilder.CreateBox("aboveRootCube", { size: 1 }, this.scene);
    // aboveRootCube.position.y = 5;

    keypoints.forEach((keypoint, index) => {
      let sphere = this.sphereMap.get(index);
      if (!sphere) {
        console.log("Creating sphere...")
        // Create a new sphere if it doesn't exist
        sphere = MeshBuilder.CreateSphere(
          `sphere${index}`,
          { diameter: 0.65 },
          this.scene
        );

        const sphereMaterial = new StandardMaterial("sphereMaterial", this.scene);
        if (index === 0) {
          sphereMaterial.diffuseColor = new Color3(1, 0, 0); // Red
        } else if (keypointInd.left.indexOf(index) > -1) {
          sphereMaterial.diffuseColor = new Color3(0, 1, 0); // Green
        } else if (keypointInd.right.indexOf(index) > -1) {
          sphereMaterial.diffuseColor = new Color3(1, 0.5, 0); // Orange
        }
        sphere.material = sphereMaterial;

        this.sphereMap.set(index, sphere);
      }

      // Update sphere position
      // const position = new Vector3(-keypoint.x, -keypoint.y, keypoint.z ?? 0);
      // sphere.position = position;
      sphere.position.x = keypoint.x;
      sphere.position.y = -keypoint.y;
      sphere.position.z = keypoint.z;

      if (index === 0) {
        // The root's depth indicates distance. That is all.
        // The other position depths are different.
        sphere.position.z = keypoints[1].z;
      }

      const connections = poseDetectionUtils.getAdjacentPairs(
        currentModel
      );

      // FIXME: While we don't have all the spheres in this.sphereMap,
      // it doesn't make sense to call this. Wasted computation.
      connections.forEach((pair) => {
        const lineKey = `line${pair[0]}-${pair[1]}`;
        let line = this.lineMap.get(lineKey);

        const firstSphere = this.sphereMap.get(pair[0]);
        const secondSphere = this.sphereMap.get(pair[1]);
        if (!firstSphere || !secondSphere) return; // Ensure both spheres exist

        if (line) {
          // Update existing line
          line = MeshBuilder.CreateLines(
            lineKey,
            { points: [firstSphere.position, secondSphere.position], updatable: true, instance: line },
            this.scene
          );
        } else {
          // Create new line
          line = MeshBuilder.CreateLines(
            lineKey,
            { points: [firstSphere.position, secondSphere.position], updatable: true },
            this.scene
          );
          this.lineMap.set(lineKey, line);
        }
      });
    });
  }
}

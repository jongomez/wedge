import { cameraCanvasId } from "@/lib/constants";
import { GlobalState } from "@/lib/types";
import React from "react";
import { CameraVideoHeight, CameraVideoWidth } from "./CameraModelMain";

export interface CameraCanvasProps {
  setGlobalState: React.Dispatch<React.SetStateAction<GlobalState>>;
  cameraCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
}

export const CameraCanvas = React.memo(function CameraCanvas({
  setGlobalState,
  cameraCanvasRef
}: CameraCanvasProps) {
  return (
    <canvas
      ref={(cameraCanvasElement) => {
        if (!cameraCanvasElement) {
          return
        }

        cameraCanvasRef.current = cameraCanvasElement
        setGlobalState((prevState) => ({
          ...prevState,
          isCameraCanvasLoaded: true
        }))
      }}
      id={cameraCanvasId}
      width={CameraVideoWidth + "px"}
      height={CameraVideoHeight + "px"}
      style={{
        maxHeight: "50vh",
        // aspectRatio: "640 / 480",
      }}
    />
  );
});

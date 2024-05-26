import { initMakerBabylon } from "@/lib/babylonjs/initMaker";
import { webGLCanvasId } from "@/lib/constants";
import { Scene } from "babylonjs";
import React, { useEffect } from "react";

let didInit = false;

export interface WebGLCanvasProps {
  setErrorMessage: (errorMessage: string) => void;
  setIsLoading: (isLoading: boolean) => void;
  setScene: (scene: Scene) => void;
}

export const WebGLCanvas = React.memo(function WebGLCanvas({
  setErrorMessage,
  setIsLoading,
  setScene
}: WebGLCanvasProps) {
  useEffect(() => {
    if (didInit) {
      // If we're here, then strict mode is on - reactStrictMode is true on the next.config.js
      console.warn(
        "WebGLCanvas Warning - tried to initialize more than once. Skipping this time."
      );
      return;
    }

    //const errorMessage = initBabylon(setIsLoading);
    const errorMessage = initMakerBabylon(setIsLoading, setScene);

    if (errorMessage) {
      console.error(errorMessage);
      setErrorMessage(errorMessage);
    }

    didInit = true;
    // This is a one-time effect - no need for cleanup or dependencies.
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <canvas
      id={webGLCanvasId}
      width="100%"
      height="100%"
      style={{
        width: "100%",
        height: "100%",
      }}
    />
  );
});

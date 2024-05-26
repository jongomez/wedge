import { GlobalState } from "@/lib/types";
import React from "react";
import { CameraVideoHeight, CameraVideoWidth } from "./CameraModelMain";

export type CameraVideoProps = {
  setGlobalState: React.Dispatch<React.SetStateAction<GlobalState>>;
  isVideoElementLoaded: boolean;
  videoRef: React.MutableRefObject<HTMLVideoElement | null>;
}

export const CameraVideo = ({ setGlobalState, videoRef, isVideoElementLoaded }: CameraVideoProps) => {

  return (
    <video
      autoPlay={true}
      muted // needed to avoid: "play() failed because the user didn't interact with the document first".
      style={{ height: CameraVideoHeight + "px", width: CameraVideoWidth + "px", display: "none" }}
      ref={(videoElement) => {
        if (!videoElement) {
          return
        }

        if (isVideoElementLoaded) {
          return
        }

        videoRef.current = videoElement
        setGlobalState((prevState: GlobalState): GlobalState => ({
          ...prevState,
          isVideoElementLoaded: true,
        }));
      }}
    ></video>
  );
}


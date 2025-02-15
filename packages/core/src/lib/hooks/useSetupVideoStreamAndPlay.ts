import { useEffect } from "react";



type UseSetupVideoStreamAndPlay = {
  stream: MediaStream | null;
  videoRef: React.MutableRefObject<HTMLVideoElement | null>;
  isVideoElementLoaded: boolean;
  successCallback: () => void;
};

export const useSetupVideoStreamAndPlay = ({
  stream,
  videoRef,
  isVideoElementLoaded,
  successCallback }: UseSetupVideoStreamAndPlay) => {
  useEffect(() => {
    if (!videoRef) {
      return;
    }

    const videoElement = (videoRef as React.MutableRefObject<HTMLVideoElement | null>).current;

    if (!stream || !videoElement || !isVideoElementLoaded) {
      return;
    }

    console.log("Setting up video stream (should only happen once)");

    videoElement.srcObject = stream;
    videoElement.play();

    successCallback();
  }, [stream, isVideoElementLoaded, videoRef]);

};
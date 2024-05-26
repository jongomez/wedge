import { useEffect, useRef, useState } from "react";
import { GlobalState } from "../../types";


export const useGetMediaStream = (globalState: GlobalState): MediaStream | null => {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const { targetFPS, cameraWidth, cameraHeight } = globalState;
  const isStreamInitialized = useRef(false);

  useEffect(() => {
    if (isStreamInitialized.current) {
      console.warn("Attempted to initialize the stream more than once.");
      return;
    } else {
      isStreamInitialized.current = true;
    }

    // if (!cameraWidth || !cameraHeight) {
    //   return;
    // }

    const cameraTargetFPS = targetFPS;
    const videoConfig: MediaStreamConstraints = {
      audio: false,
      video: {
        facingMode: "user",
        // Only setting the video to a specified size for large screen, on
        // mobile devices accept the default size.
        width: cameraWidth,
        height: cameraHeight,
        frameRate: {
          ideal: cameraTargetFPS,
        },
      },
    };

    navigator.mediaDevices
      .getUserMedia(videoConfig)
      .then((mediaStream: MediaStream) => {
        setStream(mediaStream);
      })
      .catch((err) => {
        console.error("Error getting media stream:", err);
      });

    // Using an empty dependency array here because we want to run this effect only once.
    // But, if necessary, go ahead and add dependencies.
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return stream;
};
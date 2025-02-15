import { CameraVideo } from "@/components/CameraVideo";
import { WebGLCanvas } from "@/components/Canvas3d";
import { CameraCanvas } from "@/components/CanvasCamera";
import { Loading } from "@/components/Loading";
import { WebGLNotSupported } from "@/components/WebGLErrors";
import { useModelWithCamera } from "@/lib/wedge/hooks/useTensorFlowModel";
import { useGetMediaStream } from "@/lib/wedge/hooks/useGetMediaStream";
import { useGlobalState } from "@/lib/wedge/hooks/useGlobalState";
import { useSetupVideoStreamAndPlay } from "@/lib/wedge/hooks/useSetupVideoStreamAndPlay";

import { GlobalState } from "@/lib/types";
import { FC, useRef } from "react";

let didInit = false;

export const CameraVideoHeight = 480;
// export const CameraVideoHeight = "100%";
export const CameraVideoWidth = 480;

type CameraModelMainProps = {
  isTest?: boolean;
};

export const CameraModelMain: FC<CameraModelMainProps> = ({ isTest }) => {
  const [globalState, setGlobalState] = useGlobalState();
  const stream = useGetMediaStream(globalState);
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraCanvasRef = useRef<HTMLCanvasElement>(null);

  useModelWithCamera(
    cameraCanvasRef.current,
    videoRef.current,
    globalState.scene,
    globalState.modelResult,
    globalState.isVideoElementLoaded,
    globalState.isCameraCanvasLoaded,
    globalState.isVideoStreamLoaded,
    globalState.poseEstimatorConfig,
    isTest
  );

  useSetupVideoStreamAndPlay({
    stream,
    videoRef,
    isVideoElementLoaded: globalState.isVideoElementLoaded,
    successCallback: () => {
      setGlobalState((prevState: GlobalState) => ({
        ...prevState,
        isVideoStreamLoaded: true,
      }));
    },
  });

  if (globalState.canvas3DErrorMessage.includes("WebGL not supported")) {
    return <WebGLNotSupported />;
  }

  return (
    <div
      style={{
        // position: "absolute",
        // top: 0,
        // left: 0,
        height: "100%",
        width: "100%",
      }}
    >
      {/* Loading is wrapped in a div because of this: https://stackoverflow.com/questions/54880669/react-domexception-failed-to-execute-removechild-on-node-the-node-to-be-re  */}
      {/* The reason is babylonJS adds a sneaky parent div to the canvas */}
      <Loading isLoading={globalState.isLoading} />

      <div id="speed" style={{ position: "absolute", top: 12, left: 12, color: "white" }}></div>

      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column"
        }}
      >
        {globalState.show3DCanvas && (
          <div
            style={{
              // NOTE: If the canvas is not 100% width / height, picking on the canvas doesn't work as expected.
              // Hence the position: "absoute" - it ensures that the canvas is 100% width / height.
              // This means the explorer is shown on top of the canvas.
              // There's probably a way around this, but this works for now.
              // position: "absolute",
              // top: 0,
              // left: 0,
              // width: "50%",
              height: "50%",
              flexGrow: 1,
            }}
          >
            <WebGLCanvas
              setIsLoading={(isLoading: boolean) => {
                setGlobalState((prevState: GlobalState) => ({
                  ...prevState,
                  isLoading,
                }));
              }}
              setErrorMessage={(canvas3DErrorMessage: string) => {
                setGlobalState((prevState: GlobalState) => ({
                  ...prevState,
                  canvas3DErrorMessage,
                }));
              }}
              setScene={(scene) => {
                setGlobalState((prevState: GlobalState) => ({
                  ...prevState,
                  scene,
                }));
              }}
            />
          </div>
        )}

        {globalState.show2DCanvas && (
          <div
            style={{
              display: "flex",
              // width: "50%",
              height: "100%",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <CameraVideo
              videoRef={videoRef}
              setGlobalState={setGlobalState}
              isVideoElementLoaded={globalState.isVideoElementLoaded}
            />
            <CameraCanvas cameraCanvasRef={cameraCanvasRef} setGlobalState={setGlobalState} />
          </div>
        )}
      </div>
    </div>
  );
}

import { GlobalState } from "@/lib/types";
import { Dispatch, SetStateAction } from "react";

export const useGlobalState = (): [
  GlobalState,
  Dispatch<SetStateAction<GlobalState>>
] => {
  const modelResult: number[] = [];

  const initialGameState: GlobalState = {
    isLoading: true,
    canvas3DErrorMessage: "",
    canvas2DErrorMessage: "",
    // XXX: Setting the following to "false" helps when developing UI stuff.
    // Because the canvas is not rendered, the UI is much faster to develop.
    show3DCanvas: true,
    show2DCanvas: true,
    poseEstimatorConfig: {
      backend: "webgl",
      runtime: "nnshaders",
      // runtime: "mediapipe"
    },
    targetFPS: 120,
    cameraWidth: CameraVideoWidth,
    cameraHeight: CameraVideoHeight,
    flags: {},
    modelResult,
    isVideoElementLoaded: false,
    isCameraCanvasLoaded: false,
    isVideoStreamLoaded: false,
    scene: null,
  };

  // More verbose but less funky than say: return useState<GameState>(...)
  const [gameState, setGameState] = useState<GameState>(initialGameState);

  // TODO: Maybe get the height in a different place? 
  // ... this will likely lead to a nasty re-render / resize.
  useEffect(() => {
    // if (isMobile()) {
    if (true) { // For testing purposes
      const cameraWidth = CameraVideoWidth
      const cameraHeight = CameraVideoHeight

      setGameState((prevGameState) => ({
        ...prevGameState,
        cameraWidth,
        cameraHeight,
      }));
    }
  }, []);


  return [gameState, setGameState];
};

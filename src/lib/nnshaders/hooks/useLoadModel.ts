import { loadModel } from "@/lib/loadModels";
import { Model, ModelConfig } from "@/lib/types";
import { useEffect, useRef, useState } from "react";

export const useLoadModel = (
  modelConfig: ModelConfig,
): Model | null => {
  const [myModel, setMyModel] = useState<Model | null>(null);
  const isModelLoaded = useRef(false);
  const isMounted = useRef(false);

  //
  //// Step 1 - isMounted logic. Used so we don't set state on an unmounted component.
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  //
  //// Step 2 - Load the model and setup the detector.
  useEffect(() => {
    if (isModelLoaded.current) {
      return;
    }

    loadModel(modelConfig).then((model) => {
      if (isMounted.current) {
        setMyModel(model);
      }
    });

    isModelLoaded.current = true;
  }, [modelConfig.runtime, modelConfig.backend, modelConfig.url]);

  return myModel;
}
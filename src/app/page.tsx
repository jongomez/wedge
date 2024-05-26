"use client"

import { ModelVisualize } from "@/components/visualize/ModelVisualize";
import { useLoadModel } from "@/lib/nnshaders/hooks/useLoadModel";
import { ModelConfig } from "@/lib/types";

export default function Home() {
  const modelConfig: ModelConfig = {
    runtime: "nnshaders",
    backend: "webgl",
    url: "/models/pose_three/model.json"
  };

  // const dragAndDropModel = useDragAndDropModel();
  const model = useLoadModel(modelConfig);


  if (model === null) {
    // TODO: Add / create / modify the spinner loading component
    return (
      <main className="main" >
        <div className="card loading-container">
          <h2>
            Loading model...
          </h2>
          <p>
            Please wait while the model is loading.
          </p>
        </div>
      </main >
    );
  }

  return (
    <main className="main" >
      <ModelVisualize model={model} />
    </main >
  );
}

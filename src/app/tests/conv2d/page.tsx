"use client";

import { NNShadersConv2DTests } from "@/components/test/NNShadersConv2DTests";
import { TestContainer } from "@/components/test/TestContainer";


export default function TestPage() {
  const env = process.env.NODE_ENV

  if (env !== "development") {
    return <div>What are you doing here?</div>
  }

  // TODO: Raise active state and url param handling to this level.

  return <TestContainer>
    <NNShadersConv2DTests />
  </TestContainer>
}

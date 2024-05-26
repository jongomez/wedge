"use client";

import { TestContainer } from "@/components/test/TestContainer";

export default function TestPage() {
  const env = process.env.NODE_ENV

  if (env !== "development") {
    return <div>What are you doing here?</div>
  }

  // TODO: Raise active state and url param handling to this level.

  return <TestContainer>
    <p>Click on sidebar links for the tests</p>
  </TestContainer>
}

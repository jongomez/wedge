"use client";

import { sidebarMenu } from "@wedge/core/tests/constants";

export default function TestPage() {
  const pagesWithoutHomes = Object.keys(sidebarMenu).filter((key) => key !== "/tests");

  return <div>TODO: Use the test components here</div>
}

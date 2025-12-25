"use client";

import { SidebarLayout } from "react-browser-tests";
import { sidebarMenu } from "@wedge/core/tests/constants";
import { usePathname } from "next/navigation";

export default function TestsLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const pathname = usePathname();

  if (process.env.NODE_ENV !== "development") {
    return <div>What are you doing here?</div>
  }

  return (
    <SidebarLayout sidebarMenu={sidebarMenu} activeUrl={pathname}>
      {children}
    </SidebarLayout>
  );
}

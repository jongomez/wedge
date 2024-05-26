
import { FC, useEffect, useRef } from "react";
import { useTestContext } from "./TestContext";
import { TestGroupProvider } from "./TestGroupContext";

export type TestGroupProps = React.HTMLAttributes<HTMLDivElement> & {
  title: string;
  children: React.ReactNode;
};

export const TestGroup: FC<TestGroupProps> = ({ title, children, ...props }) => {
  const { addGroupTitle } = useTestContext();
  const didInit = useRef(false);

  useEffect(() => {
    if (didInit.current) {
      return;
    }

    addGroupTitle(title);

    didInit.current = true;
  }, []);

  return <TestGroupProvider groupTitle={title}>
    <div
      className="test-group"
      {...props}
    >
      {children}
    </div>
  </TestGroupProvider>
}

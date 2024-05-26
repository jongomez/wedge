import deepEqual from 'fast-deep-equal/react';
import { Check, CircleSlash, Hourglass, LoaderCircle, X } from "lucide-react";
import { FC, useEffect, useRef } from "react";
import { useTestContext } from "./TestContext";
import { useTestGroupContext } from "./TestGroupContext";
import { toValidDOMId } from "./testHelpers";
import { TestStateIconProps, TestType } from "./types";

const getTestId = (testTitle: string, groupTitle: string): string => {
  return `${groupTitle} - ${testTitle}`;
}

export const TestStateIcon: FC<TestStateIconProps> = ({
  state
}) => {
  if (state === "Pending") {
    return <Hourglass />;
  } else if (state === "Running") {
    return <LoaderCircle className="spin-animation" />;
  } else if (state === "Fail") {
    return <X />
  } else if (state === "Success") {
    return <Check />
  } else if (state === "Skipped") {
    return <CircleSlash />
  }

  return null;
};

export type TestProps = React.HTMLAttributes<HTMLDivElement> & {
  title: string;
  fn: (contentWindow: Window, contentDocument: Document) => void;
  skip?: boolean;
  only?: boolean;
};


const checkIfTestPropsChanged = (prevProps: TestProps | undefined, props: TestProps): boolean => {
  const previousFn = prevProps?.fn;
  const currentFn = props.fn;
  const fnChanged = previousFn?.toString() !== currentFn.toString();
  const skipChanged = prevProps?.skip !== props.skip;
  const onlyChanged = prevProps?.only !== props.only;
  const titleChanged = prevProps?.title !== props.title;
  const childrenChanged = !deepEqual(prevProps?.children, props.children);

  return fnChanged || skipChanged || onlyChanged || titleChanged || childrenChanged;
}

export const Test: FC<TestProps> = (props) => {
  const { title, fn, skip = false, only = false, children, ...propsRest } = props;
  const groupTitle = useTestGroupContext().groupTitle;
  const { testMap, registerTest, reRunTests, reRunCount } = useTestContext();
  const didInit = useRef(false);
  const testId = getTestId(title, groupTitle);
  const currentTest = testMap.get(testId);
  const currentTestResult = currentTest?.state || "Pending";
  const hasChildren = children !== undefined;
  const DOMId = toValidDOMId(testId)

  // Use a ref to keep the previous props
  const prevPropsRef = useRef<TestProps | undefined>(props);
  const lastReRunCount = useRef<number>(-1);
  const propsHaveChanged = checkIfTestPropsChanged(prevPropsRef.current, props)

  useEffect(() => {
    // Skip effect if the test has already been registered AND the props haven't changed.
    if (didInit.current && !propsHaveChanged && lastReRunCount.current === reRunCount) {
      return;
    }

    if (propsHaveChanged) {
      reRunTests();
    }

    const test: TestType = {
      title,
      groupTitle,
      id: testId,
      fn,
      skip,
      only,
      hasChildren,
      state: "Pending",
    };

    registerTest(test);

    prevPropsRef.current = props;
    lastReRunCount.current = reRunCount;
    didInit.current = true;
  }, [propsHaveChanged, reRunCount]);

  return <div data-test="true" id={DOMId} {...propsRest}>
    <div className="test-title-and-result-container">
      <TestStateIcon state={currentTestResult} />
      <div className="test-title">{title}</div>
      <div className="test-result">{currentTestResult}</div>
    </div >

    {/* Only render the iframe after didInit - otherwise NextJS will throw a hydration mismatch error. */}
    {didInit.current && hasChildren && (
      <>TODO (maybe): Add iframe with react in it and stuff. Ideally you'd create a nextJS page with the component?</>
    )}
  </div>
}
export type TestConfig = {
  urls: Record<string, string>;
  testHome: string;
}

export type TestState = "Pending" | "Running" | "Skipped" | "Success" | "Fail";

export type TestType = {
  id: string;
  title: string;
  groupTitle?: string;
  fn: (contentWindow: Window, contentDocument: Document) => void | Promise<void>;
  skip: boolean;
  only: boolean;
  state: TestState;
  hasChildren: boolean;
  resultInfo?: string;
}

export type OnTestsCompleteCallbackType = (testMap: TestMap) => void

export type TotalNumberOfTests = number | null;

export type TestMap = Map<string, TestType>;

export type NextTestPagesData = {
  totalNumberOfTests: number | null | undefined;
  runningTest: string | null | undefined;
  allTestsComplete: boolean | undefined;
  groupTitles: string[] | undefined;
  testMapConvertedToObj: Record<string, TestType> | undefined;
}

export type TestContextType = {
  testMap: TestMap;
  registerTest: (test: TestType) => void;
  groupTitles: string[];
  addGroupTitle: (title: string) => void;
  reRunCount: number;
  reRunTests: () => void;
}


export type TestStateIconProps = {
  state: TestState;
};

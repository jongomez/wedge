import { useEffect, useRef, useState } from "react";
import { executeTest, toValidDOMId, updateTestMapWithTest } from "./testHelpers";
import { OnTestsCompleteCallbackType, TestMap, TotalNumberOfTests } from "./types";
import { windowUpdateRunningTest, windowUpdateTestsComplete, windowUpdateTotalNumberOfTests } from "./window";



export const useGetTotalNumberOfTests = () => {
  const [totalTests, setTotalTests] = useState<TotalNumberOfTests>(null);
  const didInit = useRef(false);

  useEffect(() => {
    if (totalTests !== null) return;
    if (didInit.current) return;

    const tests = document.querySelectorAll('[data-test="true"]');
    setTotalTests(tests.length);
    windowUpdateTotalNumberOfTests(tests.length);
    didInit.current = true;

    console.log('Total number of tests found:', tests.length);
    console.log('Waiting for the tests to be registered...');
  }, []);

  return totalTests;
};

export const useCheckForDuplicateTests = (testMap: TestMap) => {
  useEffect(() => {
    testMap.forEach((_, key) => {
      const DOMId = toValidDOMId(key);

      const duplicateTest = document.querySelectorAll(`#${DOMId}`).length > 1;

      if (duplicateTest) {
        throw new Error(`Duplicate test ID found: ${key}`);
      }
    });
  }, [testMap]);
}

export const useRunTests = (
  totalNumberOfTests: TotalNumberOfTests,
  groupTitles: string[],
  testMap: TestMap,
  currentTestIndex: number,
  reRunCount: number,
  setCurrentTestIndex: React.Dispatch<React.SetStateAction<number>>,
  setTestMap: React.Dispatch<React.SetStateAction<TestMap>>,
  onTestsComplete?: OnTestsCompleteCallbackType) => {
  const testMapHasAllTests = totalNumberOfTests !== null && testMap.size === totalNumberOfTests;
  useCheckForDuplicateTests(testMap);

  useEffect(() => {
    if (!testMapHasAllTests) return;
    if (testMap.size === 0) return;

    const testEntries = Array.from(testMap.entries());

    if (testEntries.length === 0) {
      console.warn("No tests found :(");
    }

    if (currentTestIndex === 0) {
      console.log("All tests have been registered. Running tests 1 by 1...");
    }

    // Check if there are any running tests. If yes, return early.
    const hasRunningTests = testEntries.some(([_, test]) => test.state === "Running");

    if (hasRunningTests) return;

    // The only flag is used to run only one test. If a test has the only flag, only that test will run.
    const hasOnly = testEntries.some(([_, value]) => value.only);

    // Check if all tests are processed.
    if (currentTestIndex >= testEntries.length) {
      console.log("Done ✨");
      windowUpdateTestsComplete(groupTitles, testMap);
      if (onTestsComplete) onTestsComplete(new Map(testMap));
      return;
    }

    // Fetch the current test to run and its key. The key is the test id.
    const [key, test] = testEntries[currentTestIndex];
    const skipped = test.skip || (hasOnly && !test.only);
    let testState = test.state;

    if (skipped) {
      updateTestMapWithTest(key, { ...test, state: "Skipped" }, setTestMap);
      setCurrentTestIndex(curr => curr + 1);
    } else {
      updateTestMapWithTest(key, { ...test, state: "Running" }, setTestMap);
      // Execute the test. NOTE: this is an async function.
      executeTest(test, key, setCurrentTestIndex, setTestMap);
      windowUpdateRunningTest(test.title);
    }
  }, [testMapHasAllTests, currentTestIndex, reRunCount]);
};
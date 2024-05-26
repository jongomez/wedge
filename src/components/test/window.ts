import { NextTestPagesData, TestMap } from "./types";

declare global {
  interface Window {
    nextTestPagesData: NextTestPagesData;
  }
}

const initWindowObject = () => {
  if (!window.nextTestPagesData) {
    window.nextTestPagesData = {
      allTestsComplete: false,
      groupTitles: [],
      testMapConvertedToObj: {},
      runningTest: "",
      totalNumberOfTests: 0,
    };
  }
}

export const windowUpdateTestsComplete = (groupTitles: string[], testMap: TestMap) => {
  initWindowObject();
  window.nextTestPagesData.allTestsComplete = true;
  window.nextTestPagesData.groupTitles = groupTitles;
  window.nextTestPagesData.testMapConvertedToObj = Object.fromEntries(testMap);
}

export const windowUpdateRunningTest = (testTitle: string) => {
  initWindowObject();
  window.nextTestPagesData.runningTest = testTitle;
}

export const windowUpdateTotalNumberOfTests = (totalNumberOfTests: number) => {
  initWindowObject();
  window.nextTestPagesData.totalNumberOfTests = totalNumberOfTests;
}

export const windowUpdateTestMap = (testMap: TestMap) => {
  initWindowObject();
  window.nextTestPagesData.testMapConvertedToObj = Object.fromEntries(testMap);
}
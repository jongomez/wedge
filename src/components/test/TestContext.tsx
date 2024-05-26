import React, { FC, createContext, useContext, useState } from 'react';
import { useGetTotalNumberOfTests, useRunTests } from './hooks';
import './test.css';
import { OnTestsCompleteCallbackType, TestContextType, TestMap, TestType } from './types';
import { windowUpdateTestMap } from './window';

const TestContext = createContext<TestContextType | undefined>(undefined);

export const useTestContext = () => {
  const context = useContext(TestContext);
  if (!context) {
    throw new Error('useTestContext must be used within a TestProvider');
  }
  return context;
};

export type TestProviderProps = {
  onTestsComplete?: OnTestsCompleteCallbackType;
  children: React.ReactNode;
};

export const TestProvider: FC<TestProviderProps> = ({ onTestsComplete, children }) => {
  const [testMap, setTestMap] = useState<TestMap>(new Map());
  const totalNumberOfTests = useGetTotalNumberOfTests();
  const [groupTitles, setGroupTitles] = useState<string[]>([]);
  const [reRunCount, setReRunCount] = useState(0);
  const [currentTestIndex, setCurrentTestIndex] = useState(0);

  useRunTests(
    totalNumberOfTests,
    groupTitles,
    testMap,
    currentTestIndex,
    reRunCount,
    setCurrentTestIndex,
    setTestMap,
    onTestsComplete);


  const reRunTests = () => {
    console.log('\n\nRe-running tests...');
    setCurrentTestIndex(0);
    setTestMap(new Map());
    setReRunCount(reRunCount + 1);
  }

  const registerTest = (test: TestType) => {
    setTestMap((prevTestMap) => {
      const newTestMap = new Map(prevTestMap);
      newTestMap.set(test.id, test);
      windowUpdateTestMap(newTestMap);

      return newTestMap;
    })
  };

  const addGroupTitle = (groupTitle: string) => {
    setGroupTitles((prevGroupTitles) => {
      const groupTitleSet = new Set(prevGroupTitles);
      groupTitleSet.add(groupTitle);

      return Array.from(groupTitleSet);
    });
  }

  const contextProviderValue: TestContextType = {
    testMap,
    registerTest,
    groupTitles,
    addGroupTitle,
    reRunCount,
    reRunTests,
  };

  return (
    <TestContext.Provider value={contextProviderValue}>
      {children}
    </TestContext.Provider>
  );
};
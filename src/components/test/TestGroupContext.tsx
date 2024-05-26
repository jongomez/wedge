import React, { useContext } from "react";

export type TestGroupContextType = {
  groupTitle: string;
}

export type TestGroupProviderProps = {
  groupTitle: string;
  children: React.ReactNode;
}

const TestGroupContext = React.createContext<TestGroupContextType | undefined>(undefined);

export const TestGroupProvider: React.FC<TestGroupProviderProps> = ({ groupTitle, children }) => {
  return (
    <TestGroupContext.Provider value={{ groupTitle }}>
      {children}
    </TestGroupContext.Provider>
  );
};

export const useTestGroupContext = (): TestGroupContextType => {
  const context = useContext(TestGroupContext);

  if (!context) {
    // No context means the test does not have a parent TestGroup. In this case, return a default context value.
    return { groupTitle: "Default Test Group" };
  }

  return context;
};

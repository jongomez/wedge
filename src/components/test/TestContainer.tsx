import { useRouter } from 'next/navigation';
import { FC, useEffect, useState } from 'react';
import { TestProvider } from './TestContext';
import { testConfig } from './config';


export const toUrlParam = (title: string): string => {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
};

interface TestContainerProps {
  showSidebar?: boolean;
  children: React.ReactNode;
}

const Sidebar = () => {
  const [currentPathname, setCurrentPathname] = useState("");
  const [isHomePage, setIsHomePage] = useState(false);
  const router = useRouter();

  useEffect(() => {
    setCurrentPathname(window.location.pathname);
    setIsHomePage(window.location.pathname === testConfig.testHome);
  }, []);

  const handleNavigation = (urlParam: string) => {
    router.push(urlParam);
  };

  return (
    <div style={{ width: '20%', background: '#f0f0f0', padding: '20px' }}>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        <li
          key="home"
          className={`sidebar-li ${isHomePage ? 'sidebar-li-active' : ''}`}
          onClick={() => handleNavigation('/tests')}>
          Home
        </li>
        {Object.entries(testConfig.urls).map(([url, groupTitle], index) => (
          <li
            key={index}
            className={`sidebar-li ${currentPathname === url ? 'sidebar-li-active' : ''}`}
            onClick={() => handleNavigation(`${url}`)}>
            {groupTitle}
          </li>
        ))}
      </ul>
    </div>
  );
};

export const TestContainer: FC<TestContainerProps> = ({
  showSidebar = true,
  children
}) => {
  return (
    <TestProvider>
      <div style={{ display: 'flex', height: '100vh', width: '100vw' }}>
        {showSidebar && <Sidebar />}
        <div style={{ padding: '20px' }}>
          {children}
        </div>
      </div>
    </TestProvider>
  );
}
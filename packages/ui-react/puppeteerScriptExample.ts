import * as puppeteer from "puppeteer";
import { getNumFailedTests } from "react-browser-tests";
import { TestScriptYargsArgv, defaultPuppeteerLaunchOptions, goToUrl, initPage, showTestProgress, testScriptYargsArgv } from "react-browser-tests/scripts";

// Function to launch Puppeteer and visit a URL in headless mode.
async function runTestsOnUrl(url: string, log: boolean) {
  const browser = await puppeteer.launch(defaultPuppeteerLaunchOptions);
  const page = await initPage(browser, log);
  await goToUrl(page, url);

  // Wait until all tests are complete. Show the test results on the console.
  const testContainerStates = await showTestProgress(page);
  const numFailedTest = getNumFailedTests(testContainerStates);

  if (numFailedTest) {
    console.error(`${numFailedTest} test(s) failed.`);
  } else {
    console.log("All tests passed ✨");
  }

  await browser.close();
}

runTestsOnUrl((testScriptYargsArgv as TestScriptYargsArgv).url, (testScriptYargsArgv as TestScriptYargsArgv).log).catch(error => {
  console.error("Error running tests on URL:", error);
  process.exit(1);
});
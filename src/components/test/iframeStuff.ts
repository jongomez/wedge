
async function getIframeWindowAndDocument(testID: string): Promise<[Window, Document]> {
  let iframeId = toValidDOMId(testID) + "-frame";
  let iframe = document.getElementById(iframeId) as HTMLIFrameElement;

  if (!iframe) {
    throw new Error(`Failed to find iframe with ID ${iframeId}`);
  }

  if (!isIframeLoaded(iframe)) {
    await waitForIframeLoad(iframe);
  }

  const iframeWindow = iframe.contentWindow;
  const iframeDocument = iframe.contentDocument;

  if (!iframeWindow) {
    throw new Error(`Failed to get the content window for iframe with ID ${iframe.id}`);
  }

  if (!iframeDocument) {
    throw new Error(`Failed to get the content document for iframe with ID ${iframe.id}`);
  }

  return [iframeWindow, iframeDocument];
}

function isIframeLoaded(iframe: HTMLIFrameElement): boolean {
  try {
    // Check if the iframe's document is accessible and complete
    const doc = iframe.contentDocument || iframe.contentWindow?.document;
    return doc?.readyState === 'complete';
  } catch (error) {
    // Likely a security error if the iframe is from a different origin
    console.error("Unable to access iframe's document:", error);
    return false;
  }
}

function waitForIframeLoad(iframe: HTMLIFrameElement, timeout: number = 5000): Promise<void> {
  return new Promise((resolve, reject) => {
    // Set up a timer to reject the promise if the iframe takes too long to load
    const timeoutId = setTimeout(() => {
      iframe.onload = null;  // Clean up the onload event handler
      throw new Error(`Loading timeout exceeded for iframe with ID ${iframe.id}`);
    }, timeout);

    iframe.onload = () => {
      clearTimeout(timeoutId);
      resolve();
    };

    iframe.onerror = () => {
      clearTimeout(timeoutId);
      throw new Error(`Failed to load the iframe with ID ${iframe.id}`);
    };
  });
}

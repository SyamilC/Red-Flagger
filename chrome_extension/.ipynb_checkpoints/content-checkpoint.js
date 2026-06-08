// content.js
(function auditTargetDOM() {
  console.log("[RedFlagAgent] Scraping target active viewport DOM...");

  // Selectors mapped to expected parent profiles
  const bioTextElements = document.querySelectorAll(".profile-essay-contents, [data-testid='profile-bio']");
  let combinedBioText = "";
  bioTextElements.forEach(el => combinedBioText += " " + el.innerText);

  const profileImageElements = document.querySelectorAll(".profile-photo, img[src*='profile']");
  const extractedImageUrls = Array.from(profileImageElements).map(img => img.src);

  // Parse structured details out of standard key-value profile tables
  const scrapeTabularFeature = (labelKey) => {
    const detailRow = Array.from(document.querySelectorAll('.profile-details-row, .match-profile-details'))
                           .find(el => el.innerText.toLowerCase().includes(labelKey.toLowerCase()));
    return detailRow ? detailRow.innerText.replace(/.*?\n/, '').trim() : "unknown";
  };

  const profilePayload = {
    text: combinedBioText.trim() || "No text description discovered.",
    imageUrls: extractedImageUrls,
    tabularData: {
      age: parseInt(scrapeTabularFeature("age")) || 25,
      status: scrapeTabularFeature("status"),
      drinks: scrapeTabularFeature("drinks"),
      drugs: scrapeTabularFeature("drugs"),
      smokes: scrapeTabularFeature("smokes"),
      orientation: scrapeTabularFeature("orientation")
    }
  };

  // Dispatch extracted structural data to background engine
  chrome.runtime.sendMessage({ action: "EXECUTE_INFERENCE", payload: profilePayload });
})();
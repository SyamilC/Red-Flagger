// Extracts profile data from the DOM. Update selectors to match your target dating site.
function extractProfileData() {
  const bioText = document.querySelector('.profile-bio, .about-me, [data-test="profile-bio"]')?.innerText || '';
  
  // Simplified tabular mapping for demo purposes (expand based on site structure)
  const tabularData = {
    age: parseInt(document.querySelector('.profile-age')?.innerText) || 25,
    height: 175.0, 
    income: -1,
    drinks: 'socially',
    drugs: 'never',
    smokes: 'no',
    status: 'single',
    sex: 'm',
    orientation: 'straight',
    body_type: 'average',
    diet: 'anything',
    education: 'graduated from college/university',
    ethnicity: 'white',
    job: 'other',
    offspring: "doesn't have kids",
    pets: 'likes dogs',
    religion: 'agnosticism',
    sign: 'leo'
  };

  const images = Array.from(document.querySelectorAll('img.profile-photo, img.avatar')).map(img => img.src);
  
  return {
    text: bioText,
    tabular: tabularData,
    images: images.slice(0, 1) // Analyze first image for performance
  };
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'GET_PROFILE_DATA') {
    const data = extractProfileData();
    sendResponse(data);
  }
});
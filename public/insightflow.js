// InsightFlow AI - Custom UI Script

document.addEventListener('DOMContentLoaded', function() {
  // Create and append the custom UI elements once the Chainlit UI has loaded
  setTimeout(createCustomUI, 1000);
});

// Main function to create the custom UI
function createCustomUI() {
  createLeftSidebar();
  createAppTabs();
  createRightSidebar();
  setupEventListeners();
}

// Create the left sidebar with settings
function createLeftSidebar() {
  const sidebar = document.querySelector('.cl-sidebar');
  if (!sidebar) return;
  
  // Clear existing content
  sidebar.innerHTML = '';
  
  // Add app title
  const appTitle = document.createElement('div');
  appTitle.className = 'app-title';
  appTitle.textContent = 'InsightFlow AI';
  sidebar.appendChild(appTitle);
  
  // Create Research Panel Settings section
  const researchSettings = document.createElement('div');
  researchSettings.className = 'settings-section';
  researchSettings.innerHTML = `
    <h3>Research Panel Settings</h3>
    <div class="persona-types">
      <p>Select Persona Types:</p>
      <div class="persona-checkbox">
        <input type="checkbox" id="analytical" checked>
        <label for="analytical">Analytical/Diagnostic</label>
      </div>
      <div class="persona-checkbox">
        <input type="checkbox" id="scientific" checked>
        <label for="scientific">Scientific/STEM Explorer</label>
      </div>
      <div class="persona-checkbox">
        <input type="checkbox" id="metaphorical" checked>
        <label for="metaphorical">Metaphorical/Creative-Analogy</label>
      </div>
      <div class="persona-checkbox">
        <input type="checkbox" id="philosophical" checked>
        <label for="philosophical">Spiritual/Philosophical</label>
      </div>
      <div class="persona-checkbox">
        <input type="checkbox" id="factual">
        <label for="factual">Practical/Factual</label>
      </div>
      <div class="persona-checkbox">
        <input type="checkbox" id="historical">
        <label for="historical">Historical/Synthesis</label>
      </div>
      <div class="persona-checkbox">
        <input type="checkbox" id="futuristic">
        <label for="futuristic">Futuristic/Speculative</label>
      </div>
    </div>
  `;
  sidebar.appendChild(researchSettings);
  
  // Create Model Settings section
  const modelSettings = document.createElement('div');
  modelSettings.className = 'settings-section';
  modelSettings.innerHTML = `
    <h3>Model Settings</h3>
    <div class="model-selection">
      <span class="model-label">Model</span>
      <div class="persona-checkbox">
        <input type="radio" id="default" name="model" checked>
        <label for="default">Default</label>
      </div>
      <div class="persona-checkbox">
        <input type="radio" id="gpt4" name="model">
        <label for="gpt4">GPT-4</label>
      </div>
      <div class="persona-checkbox">
        <input type="radio" id="claude" name="model">
        <label for="claude">Claude</label>
      </div>
    </div>
    
    <div class="temperature-control">
      <span class="model-label">Temperature: 0.8</span>
      <input type="range" min="0" max="1" step="0.1" value="0.8" class="temperature-slider">
    </div>
  `;
  sidebar.appendChild(modelSettings);
  
  // Create Session controls
  const sessionSection = document.createElement('div');
  sessionSection.className = 'settings-section';
  sessionSection.innerHTML = `
    <h3>Session</h3>
    <button id="clearChat" class="persona-checkbox">Clear Chat History</button>
  `;
  sidebar.appendChild(sessionSection);
}

// Create application tabs (Research Assistant/Multi-Persona Discussion)
function createAppTabs() {
  const chatContainer = document.querySelector('.cl-chat-container');
  if (!chatContainer) return;
  
  // Find the chat area to insert tabs before it
  const chatArea = document.querySelector('.cl-chat-container .cl-message-list');
  if (!chatArea) return;
  
  // Create tabs container
  const tabsContainer = document.createElement('div');
  tabsContainer.className = 'app-tabs';
  
  // Create Research Assistant tab (active by default)
  const researchTab = document.createElement('div');
  researchTab.className = 'app-tab active';
  researchTab.textContent = 'Research Assistant';
  researchTab.dataset.tab = 'research';
  
  // Create Multi-Persona Discussion tab
  const multiPersonaTab = document.createElement('div');
  multiPersonaTab.className = 'app-tab';
  multiPersonaTab.textContent = 'Multi-Persona Discussion';
  multiPersonaTab.dataset.tab = 'discussion';
  
  // Add tabs to container
  tabsContainer.appendChild(researchTab);
  tabsContainer.appendChild(multiPersonaTab);
  
  // Insert tabs before chat area
  chatContainer.insertBefore(tabsContainer, chatArea);
  
  // Create header showing selected personas
  const selectedPersonasHeader = document.createElement('div');
  selectedPersonasHeader.className = 'selected-personas-header';
  selectedPersonasHeader.textContent = 'Selected Persona Types: Analytical/Diagnostic, Scientific/STEM Explorer, Spiritual/Philosophical';
  
  // Insert header after tabs
  chatContainer.insertBefore(selectedPersonasHeader, chatArea);
}

// Create right sidebar for Research Context
function createRightSidebar() {
  // Check if the main element exists
  const main = document.querySelector('.cl-main');
  if (!main) return;
  
  // Create the research context sidebar
  const researchContext = document.createElement('div');
  researchContext.className = 'research-context';
  
  // Create tabs for the research context
  const tabs = document.createElement('div');
  tabs.className = 'research-tabs';
  
  const contextTab = document.createElement('div');
  contextTab.className = 'research-tab active';
  contextTab.textContent = 'Context';
  
  const sourcesTab = document.createElement('div');
  sourcesTab.className = 'research-tab';
  sourcesTab.textContent = 'Sources';
  
  const settingsTab = document.createElement('div');
  settingsTab.className = 'research-tab';
  settingsTab.textContent = 'Settings';
  
  tabs.appendChild(contextTab);
  tabs.appendChild(sourcesTab);
  tabs.appendChild(settingsTab);
  
  // Create active personas section
  const activePersonasSection = document.createElement('div');
  activePersonasSection.className = 'context-section';
  activePersonasSection.innerHTML = `
    <h3>Active Personas</h3>
    
    <!-- Analytical persona -->
    <div class="active-persona">
      <div class="persona-header">
        <h4>Analytical/Diagnostic</h4>
        <span>▼</span>
      </div>
      <div class="persona-content">
        <p>Methodical examination of details and logical connections for problem-solving.</p>
        <div class="available-personalities">
          <p>Available Personalities:</p>
          <div class="personality-option">
            <input type="checkbox" id="sherlock-holmes" checked>
            <label for="sherlock-holmes">Sherlock Holmes</label>
          </div>
          <div class="personality-option">
            <input type="checkbox" id="gregory-house">
            <label for="gregory-house">Dr. Gregory House MD</label>
          </div>
          <div class="personality-option">
            <input type="checkbox" id="hercule-poirot">
            <label for="hercule-poirot">Hercule Poirot</label>
          </div>
          <div class="personality-option">
            <input type="checkbox" id="christopher-nolan">
            <label for="christopher-nolan">Christopher Nolan</label>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Scientific persona -->
    <div class="active-persona">
      <div class="persona-header">
        <h4>Scientific/STEM Explorer</h4>
        <span>▼</span>
      </div>
      <div class="persona-content">
        <p>Evidence-based reasoning using empirical data and research.</p>
        <div class="available-personalities">
          <p>Available Personalities:</p>
          <div class="personality-option">
            <input type="checkbox" id="richard-feynman" checked>
            <label for="richard-feynman">Richard Feynman</label>
          </div>
          <div class="personality-option">
            <input type="checkbox" id="david-deutsch">
            <label for="david-deutsch">David Deutsch</label>
          </div>
          <div class="personality-option">
            <input type="checkbox" id="hans-rosling">
            <label for="hans-rosling">Hans Rosling</label>
          </div>
          <div class="personality-option">
            <input type="checkbox" id="hannah-fry">
            <label for="hannah-fry">Hannah Fry</label>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Philosophical persona -->
    <div class="active-persona">
      <div class="persona-header">
        <h4>Spiritual/Philosophical</h4>
        <span>▼</span>
      </div>
      <div class="persona-content">
        <p>Holistic perspectives examining deeper meaning and interconnectedness.</p>
        <div class="available-personalities">
          <p>Available Personalities:</p>
          <div class="personality-option">
            <input type="checkbox" id="jiddu-krishnamurti" checked>
            <label for="jiddu-krishnamurti">Jiddu Krishnamurti</label>
          </div>
          <div class="personality-option">
            <input type="checkbox" id="swami-vivekananda">
            <label for="swami-vivekananda">Swami Vivekananda</label>
          </div>
          <div class="personality-option">
            <input type="checkbox" id="dalai-lama">
            <label for="dalai-lama">Dalai Lama</label>
          </div>
        </div>
      </div>
    </div>
  `;
  
  // Create additional context section
  const additionalContextSection = document.createElement('div');
  additionalContextSection.className = 'additional-context';
  additionalContextSection.innerHTML = `
    <h3>Additional Context</h3>
    <p>Add background information or specific instructions</p>
    <textarea placeholder="Enter additional research context or instructions here..."></textarea>
    <button class="apply-context-btn">Apply Context</button>
  `;
  
  // Assemble the research context sidebar
  researchContext.appendChild(tabs);
  researchContext.appendChild(activePersonasSection);
  researchContext.appendChild(additionalContextSection);
  
  // Add the research context to the main element
  const parent = main.parentElement;
  parent.appendChild(researchContext);
}

// Set up event listeners for interactive elements
function setupEventListeners() {
  // Toggle persona accordions
  const personaHeaders = document.querySelectorAll('.persona-header');
  personaHeaders.forEach(header => {
    header.addEventListener('click', function() {
      const content = this.nextElementSibling;
      const indicator = this.querySelector('span');
      
      if (content.style.display === 'none') {
        content.style.display = 'block';
        indicator.textContent = '▼';
      } else {
        content.style.display = 'none';
        indicator.textContent = '▶';
      }
    });
  });
  
  // Handle tab switching
  const appTabs = document.querySelectorAll('.app-tab');
  appTabs.forEach(tab => {
    tab.addEventListener('click', function() {
      // Remove active class from all tabs
      appTabs.forEach(t => t.classList.remove('active'));
      // Add active class to clicked tab
      this.classList.add('active');
      
      // Update the selected personas header based on tab
      const header = document.querySelector('.selected-personas-header');
      if (header) {
        if (this.dataset.tab === 'research') {
          header.textContent = 'Selected Persona Types: Analytical/Diagnostic, Scientific/STEM Explorer, Spiritual/Philosophical';
        } else {
          header.textContent = 'Multi-Persona Discussion Mode: All personas participate independently';
        }
      }
    });
  });
  
  // Handle research context tabs
  const researchTabs = document.querySelectorAll('.research-tab');
  researchTabs.forEach(tab => {
    tab.addEventListener('click', function() {
      // Remove active class from all tabs
      researchTabs.forEach(t => t.classList.remove('active'));
      // Add active class to clicked tab
      this.classList.add('active');
    });
  });
  
  // Handle clear chat button
  const clearChatBtn = document.getElementById('clearChat');
  if (clearChatBtn) {
    clearChatBtn.addEventListener('click', function() {
      // This would typically call a Chainlit function to clear the chat
      // For now, just clear the messages in the UI
      const messageList = document.querySelector('.cl-message-list');
      if (messageList) {
        messageList.innerHTML = '';
      }
    });
  }
}

// Periodically check if UI needs to be refreshed (in case of Chainlit UI refreshes)
setInterval(function() {
  const customUi = document.querySelector('.app-title');
  if (!customUi) {
    createCustomUI();
  }
}, 3000); 
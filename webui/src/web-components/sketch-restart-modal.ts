import { css, html, LitElement } from "lit";
import { customElement, property, state } from "lit/decorators.js";
import { AgentMessage, State } from "../types";

@customElement("sketch-restart-modal")
export class SketchRestartModal extends LitElement {
  @property({ type: Boolean })
  open = false;

  @property({ attribute: false })
  containerState: State | null = null;

  @property({ attribute: false })
  messages: AgentMessage[] = [];

  @state()
  private restartType: "initial" | "current" | "other" = "current";

  @state()
  private customRevision = "";

  @state()
  private promptOption: "suggested" | "original" | "new" = "suggested";

  @state()
  private commitDescriptions: Record<string, string> = {
    current: "",
    initial: "",
  };

  @state()
  private suggestedPrompt = "";

  @state()
  private originalPrompt = "";

  @state()
  private newPrompt = "";

  @state()
  private isLoading = false;

  @state()
  private isSuggestionLoading = false;

  @state()
  private isOriginalPromptLoading = false;

  @state()
  private errorMessage = "";

  static styles = css`
    :host {
      display: block;
      font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Roboto,
        sans-serif;
    }

    .modal-description {
      margin: 0 0 20px 0;
      color: #555;
      font-size: 14px;
      line-height: 1.5;
    }

    .container-message {
      margin: 10px 0;
      padding: 8px 12px;
      background-color: #f8f9fa;
      border-left: 4px solid #6c757d;
      color: #555;
      font-size: 14px;
      line-height: 1.5;
      border-radius: 4px;
    }

    .modal-backdrop {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.5);
      z-index: 1000;
      display: flex;
      justify-content: center;
      align-items: center;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease-in-out;
    }

    .modal-backdrop.open {
      opacity: 1;
      pointer-events: auto;
    }

    .modal-container {
      background: white;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      width: 600px;
      max-width: 90%;
      max-height: 90vh;
      overflow-y: auto;
      padding: 20px;
    }

    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      border-bottom: 1px solid #eee;
      padding-bottom: 10px;
    }

    .modal-title {
      font-size: 18px;
      font-weight: 600;
      margin: 0;
    }

    .close-button {
      background: none;
      border: none;
      font-size: 18px;
      cursor: pointer;
      color: #666;
    }

    .close-button:hover {
      color: #333;
    }

    .form-group {
      margin-bottom: 16px;
    }

    .horizontal-radio-group {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 16px;
    }

    .revision-option {
      border: 1px solid #e0e0e0;
      border-radius: 4px;
      padding: 8px 12px;
      min-width: 180px;
      cursor: pointer;
      transition: all 0.2s;
    }

    .revision-option label {
      font-size: 0.9em;
      font-weight: bold;
    }

    .revision-option:hover {
      border-color: #2196f3;
      background-color: #f5f9ff;
    }

    .revision-option.selected {
      border-color: #2196f3;
      background-color: #e3f2fd;
    }

    .revision-option input[type="radio"] {
      margin-right: 8px;
    }

    .revision-description {
      margin-top: 4px;
      color: #666;
      font-size: 0.8em;
      font-family: monospace;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 200px;
    }

    .form-group label {
      display: block;
      margin-bottom: 8px;
      font-weight: 500;
    }

    .radio-group {
      margin-bottom: 8px;
    }

    .radio-option {
      display: flex;
      align-items: center;
      margin-bottom: 8px;
    }

    .radio-option input {
      margin-right: 8px;
    }

    .custom-revision {
      margin-left: 24px;
      margin-top: 8px;
      width: calc(100% - 24px);
      padding: 6px 8px;
      border: 1px solid #ddd;
      border-radius: 4px;
      display: block;
    }

    .prompt-container {
      position: relative;
      margin-top: 16px;
    }

    .prompt-textarea {
      display: block;
      box-sizing: border-box;
      width: 100%;
      min-height: 120px;
      padding: 8px;
      border: 1px solid #ddd;
      border-radius: 4px;
      font-family: inherit;
      resize: vertical;
      background-color: white;
      color: #333;
    }

    .prompt-textarea.disabled {
      background-color: #f5f5f5;
      color: #999;
      cursor: not-allowed;
    }

    .actions {
      display: flex;
      justify-content: flex-end;
      gap: 12px;
      margin-top: 20px;
    }

    .btn {
      padding: 8px 16px;
      border-radius: 4px;
      font-weight: 500;
      cursor: pointer;
      border: none;
    }

    .btn-cancel {
      background: #f2f2f2;
      color: #333;
    }

    .btn-restart {
      background: #4caf50;
      color: white;
    }

    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .error-message {
      color: #e53935;
      margin-top: 16px;
      font-size: 14px;
    }

    .loading-indicator {
      display: inline-block;
      margin-right: 8px;
      margin-left: 8px;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: white;
      animation: spin 1s linear infinite;
    }

    .prompt-container .loading-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(255, 255, 255, 0.7);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 2;
      border-radius: 4px;
    }

    .prompt-container .loading-overlay .loading-indicator {
      width: 24px;
      height: 24px;
      border: 2px solid rgba(0, 0, 0, 0.1);
      border-top-color: #2196f3;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    .radio-option .loading-indicator {
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 1.5px solid rgba(0, 0, 0, 0.2);
      border-top-color: #2196f3;
      vertical-align: middle;
      margin-left: 8px;
    }

    .radio-option .status-ready {
      display: inline-block;
      width: 16px;
      height: 16px;
      color: #4caf50;
      margin-left: 8px;
      font-weight: bold;
      vertical-align: middle;
    }

    @keyframes spin {
      to {
        transform: rotate(360deg);
      }
    }
  `;

  constructor() {
    super();
    this.handleEscape = this.handleEscape.bind(this);
  }

  connectedCallback() {
    super.connectedCallback();
    document.addEventListener("keydown", this.handleEscape);
  }

  // Handle keyboard navigation
  firstUpdated() {
    if (this.shadowRoot) {
      // Set up proper tab navigation by ensuring all focusable elements are included
      const focusableElements =
        this.shadowRoot.querySelectorAll('[tabindex="0"]');
      if (focusableElements.length > 0) {
        // Set initial focus when modal opens
        (focusableElements[0] as HTMLElement).focus();
      }
    }
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    document.removeEventListener("keydown", this.handleEscape);
  }

  handleEscape(e: KeyboardEvent) {
    if (e.key === "Escape" && this.open) {
      this.closeModal();
    }
  }

  closeModal() {
    this.open = false;
    this.dispatchEvent(new CustomEvent("close"));
  }

  async loadCommitDescription(
    revision: string,
    target: "current" | "initial" | "other" = "other",
  ) {
    try {
      const response = await fetch(
        `./commit-description?revision=${encodeURIComponent(revision)}`,
      );
      if (!response.ok) {
        throw new Error(
          `Failed to load commit description: ${response.statusText}`,
        );
      }

      const data = await response.json();

      if (target === "other") {
        // For custom revisions, update the customRevision directly
        this.customRevision = `${revision.slice(0, 8)} - ${data.description}`;
      } else {
        // For known targets, update the commitDescriptions object
        this.commitDescriptions = {
          ...this.commitDescriptions,
          [target]: data.description,
        };
      }
    } catch (error) {
      console.error(`Error loading commit description for ${revision}:`, error);
    }
  }

  handleRevisionChange(e?: Event) {
    if (e) {
      const target = e.target as HTMLInputElement;
      this.restartType = target.value as "initial" | "current" | "other";
    }

    // Load commit description for any custom revision if needed
    if (
      this.restartType === "other" &&
      this.customRevision &&
      !this.customRevision.includes(" - ")
    ) {
      this.loadCommitDescription(this.customRevision, "other");
    }
  }

  handleCustomRevisionChange(e: Event) {
    const target = e.target as HTMLInputElement;
    this.customRevision = target.value;
  }

  handlePromptOptionChange(e: Event) {
    const target = e.target as HTMLInputElement;
    this.promptOption = target.value as "suggested" | "original" | "new";

    if (
      this.promptOption === "suggested" &&
      !this.isSuggestionLoading &&
      this.suggestedPrompt === ""
    ) {
      this.loadSuggestedPrompt();
    } else if (
      this.promptOption === "original" &&
      !this.isOriginalPromptLoading &&
      this.originalPrompt === ""
    ) {
      this.loadOriginalPrompt();
    }
  }

  handleSuggestedPromptChange(e: Event) {
    const target = e.target as HTMLTextAreaElement;
    this.suggestedPrompt = target.value;
  }

  handleOriginalPromptChange(e: Event) {
    const target = e.target as HTMLTextAreaElement;
    this.originalPrompt = target.value;
  }

  handleNewPromptChange(e: Event) {
    const target = e.target as HTMLTextAreaElement;
    this.newPrompt = target.value;
  }

  async loadSuggestedPrompt() {
    try {
      this.isSuggestionLoading = true;
      this.errorMessage = "";

      const response = await fetch("./suggest-reprompt");
      if (!response.ok) {
        throw new Error(`Failed to load suggestion: ${response.statusText}`);
      }

      const data = await response.json();
      this.suggestedPrompt = data.prompt;
    } catch (error) {
      console.error("Error loading suggested prompt:", error);
      this.errorMessage =
        error instanceof Error ? error.message : "Failed to load suggestion";
    } finally {
      this.isSuggestionLoading = false;
    }
  }

  async loadOriginalPrompt() {
    try {
      this.isOriginalPromptLoading = true;
      this.errorMessage = "";

      // Get the first message index from the container state
      const firstMessageIndex = this.containerState?.first_message_index || 0;

      // Find the first user message after the first_message_index
      let firstUserMessage = "";

      if (this.messages && this.messages.length > 0) {
        for (const msg of this.messages) {
          // Only look at messages starting from first_message_index
          if (msg.idx >= firstMessageIndex && msg.type === "user") {
            // Simply use the content field if it's a string
            if (typeof msg.content === "string") {
              firstUserMessage = msg.content;
            } else {
              // Fallback to stringifying content field for any other type
              firstUserMessage = JSON.stringify(msg.content);
            }
            break;
          }
        }
      }

      if (!firstUserMessage) {
        console.warn("Could not find original user message", this.messages);
      }

      this.originalPrompt = firstUserMessage;
    } catch (error) {
      console.error("Error loading original prompt:", error);
      this.errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to load original prompt";
    } finally {
      this.isOriginalPromptLoading = false;
    }
  }

  async handleRestart() {
    try {
      this.isLoading = true;
      this.errorMessage = "";

      let revision = "";
      switch (this.restartType) {
        case "initial":
          // We'll leave revision empty for this case, backend will handle it
          break;
        case "current":
          // We'll leave revision empty for this case too, backend will use current HEAD
          break;
        case "other":
          revision = this.customRevision.trim();
          if (!revision) {
            throw new Error("Please enter a valid revision");
          }
          break;
      }

      // Determine which prompt to use based on selected option
      let initialPrompt = "";
      switch (this.promptOption) {
        case "suggested":
          initialPrompt = this.suggestedPrompt.trim();
          if (!initialPrompt && this.isSuggestionLoading) {
            throw new Error(
              "Suggested prompt is still loading. Please wait or choose another option.",
            );
          }
          break;
        case "original":
          initialPrompt = this.originalPrompt.trim();
          if (!initialPrompt && this.isOriginalPromptLoading) {
            throw new Error(
              "Original prompt is still loading. Please wait or choose another option.",
            );
          }
          break;
        case "new":
          initialPrompt = this.newPrompt.trim();
          break;
      }

      // Validate we have a prompt when needed
      if (!initialPrompt && this.promptOption !== "new") {
        throw new Error(
          "Unable to get prompt text. Please enter a new prompt or try again.",
        );
      }

      const response = await fetch("./restart", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          revision: revision,
          initial_prompt: initialPrompt,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to restart: ${errorText}`);
      }

      // Reload the page after successful restart
      window.location.reload();
    } catch (error) {
      console.error("Error restarting conversation:", error);
      this.errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to restart conversation";
    } finally {
      this.isLoading = false;
    }
  }

  updated(changedProperties: Map<string, any>) {
    if (changedProperties.has("open") && this.open) {
      // Reset form when opening
      this.restartType = "current";
      this.customRevision = "";
      this.promptOption = "suggested";
      this.suggestedPrompt = "";
      this.originalPrompt = "";
      this.newPrompt = "";
      this.errorMessage = "";
      this.commitDescriptions = {
        current: "",
        initial: "",
      };

      // Pre-load all available prompts and commit descriptions in the background
      setTimeout(() => {
        // Load prompt data
        this.loadSuggestedPrompt();
        this.loadOriginalPrompt();

        // Load commit descriptions
        this.loadCommitDescription("HEAD", "current");
        if (this.containerState?.initial_commit) {
          this.loadCommitDescription(
            this.containerState.initial_commit,
            "initial",
          );
        }

        // Set focus to the first radio button for keyboard navigation
        if (this.shadowRoot) {
          const firstInput = this.shadowRoot.querySelector(
            'input[type="radio"]',
          ) as HTMLElement;
          if (firstInput) {
            firstInput.focus();
          }
        }
      }, 0);
    }
  }

  render() {
    const inContainer = this.containerState?.in_container || false;

    return html`
      <div class="modal-backdrop ${this.open ? "open" : ""}">
        <div class="modal-container">
          <div class="modal-header">
            <h2 class="modal-title">Restart Conversation</h2>
            <button class="close-button" @click=${this.closeModal}>×</button>
          </div>

          <p class="modal-description">
            Restarting the conversation hides the history from the agent. If you
            want the agent to take a different direction, restart with a new
            prompt.
          </p>

          <div class="form-group">
            <label>Reset to which revision?</label>
            <div class="horizontal-radio-group">
              <div
                class="revision-option ${this.restartType === "current"
                  ? "selected"
                  : ""}"
                @click=${() => {
                  this.restartType = "current";
                  this.handleRevisionChange();
                }}
              >
                <input
                  type="radio"
                  id="restart-current"
                  name="restart-type"
                  value="current"
                  ?checked=${this.restartType === "current"}
                  @change=${this.handleRevisionChange}
                  tabindex="0"
                />
                <label for="restart-current">Current HEAD</label>
                ${this.commitDescriptions.current
                  ? html`<div class="revision-description">
                      ${this.commitDescriptions.current}
                    </div>`
                  : ""}
              </div>

              ${inContainer
                ? html`
                    <div
                      class="revision-option ${this.restartType === "initial"
                        ? "selected"
                        : ""}"
                      @click=${() => {
                        this.restartType = "initial";
                        this.handleRevisionChange();
                      }}
                    >
                      <input
                        type="radio"
                        id="restart-initial"
                        name="restart-type"
                        value="initial"
                        ?checked=${this.restartType === "initial"}
                        @change=${this.handleRevisionChange}
                        tabindex="0"
                      />
                      <label for="restart-initial">Initial commit</label>
                      ${this.commitDescriptions.initial
                        ? html`<div class="revision-description">
                            ${this.commitDescriptions.initial}
                          </div>`
                        : ""}
                    </div>

                    <div
                      class="revision-option ${this.restartType === "other"
                        ? "selected"
                        : ""}"
                      @click=${() => {
                        this.restartType = "other";
                        this.handleRevisionChange();
                      }}
                    >
                      <input
                        type="radio"
                        id="restart-other"
                        name="restart-type"
                        value="other"
                        ?checked=${this.restartType === "other"}
                        @change=${this.handleRevisionChange}
                        tabindex="0"
                      />
                      <label for="restart-other">Other revision</label>
                    </div>
                  `
                : html`
                    <div class="container-message">
                      Additional revision options are not available because
                      Sketch is not running inside a container.
                    </div>
                  `}
            </div>

            ${this.restartType === "other" && inContainer
              ? html`
                  <input
                    type="text"
                    class="custom-revision"
                    placeholder="Enter commit hash"
                    .value=${this.customRevision}
                    @input=${this.handleCustomRevisionChange}
                    tabindex="0"
                  />
                `
              : ""}
          </div>

          <div class="form-group">
            <label>Prompt options:</label>
            <div class="radio-group">
              <div class="radio-option">
                <input
                  type="radio"
                  id="prompt-suggested"
                  name="prompt-type"
                  value="suggested"
                  ?checked=${this.promptOption === "suggested"}
                  @change=${this.handlePromptOptionChange}
                  tabindex="0"
                />
                <label for="prompt-suggested">
                  Suggest prompt based on history (default)
                </label>
              </div>

              <div class="radio-option">
                <input
                  type="radio"
                  id="prompt-original"
                  name="prompt-type"
                  value="original"
                  ?checked=${this.promptOption === "original"}
                  @change=${this.handlePromptOptionChange}
                  tabindex="0"
                />
                <label for="prompt-original"> Original prompt </label>
              </div>

              <div class="radio-option">
                <input
                  type="radio"
                  id="prompt-new"
                  name="prompt-type"
                  value="new"
                  ?checked=${this.promptOption === "new"}
                  @change=${this.handlePromptOptionChange}
                  tabindex="0"
                />
                <label for="prompt-new">New prompt</label>
              </div>
            </div>
          </div>

          <div class="prompt-container">
            ${this.promptOption === "suggested"
              ? html`
                  <textarea
                    class="prompt-textarea${this.isSuggestionLoading
                      ? " disabled"
                      : ""}"
                    placeholder="Loading suggested prompt..."
                    .value=${this.suggestedPrompt}
                    ?disabled=${this.isSuggestionLoading}
                    @input=${this.handleSuggestedPromptChange}
                    tabindex="0"
                  ></textarea>
                  ${this.isSuggestionLoading
                    ? html`
                        <div class="loading-overlay">
                          <div class="loading-indicator"></div>
                        </div>
                      `
                    : ""}
                `
              : this.promptOption === "original"
                ? html`
                    <textarea
                      class="prompt-textarea${this.isOriginalPromptLoading
                        ? " disabled"
                        : ""}"
                      placeholder="Loading original prompt..."
                      .value=${this.originalPrompt}
                      ?disabled=${this.isOriginalPromptLoading}
                      @input=${this.handleOriginalPromptChange}
                      tabindex="0"
                    ></textarea>
                    ${this.isOriginalPromptLoading
                      ? html`
                          <div class="loading-overlay">
                            <div class="loading-indicator"></div>
                          </div>
                        `
                      : ""}
                  `
                : html`
                    <textarea
                      class="prompt-textarea"
                      placeholder="Enter a new prompt..."
                      .value=${this.newPrompt}
                      @input=${this.handleNewPromptChange}
                      tabindex="0"
                    ></textarea>
                  `}
          </div>

          ${this.errorMessage
            ? html` <div class="error-message">${this.errorMessage}</div> `
            : ""}

          <div class="actions">
            <button
              class="btn btn-cancel"
              @click=${this.closeModal}
              ?disabled=${this.isLoading}
              tabindex="0"
            >
              Cancel
            </button>
            <button
              class="btn btn-restart"
              @click=${this.handleRestart}
              ?disabled=${this.isLoading}
              tabindex="0"
            >
              ${this.isLoading
                ? html`<span class="loading-indicator"></span>`
                : ""}
              Restart
            </button>
          </div>
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "sketch-restart-modal": SketchRestartModal;
  }
}

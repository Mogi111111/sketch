import { css, html, LitElement } from "lit";
import { customElement, property, state } from "lit/decorators.js";
import { ConnectionStatus, DataManager } from "../data";
import { AgentMessage, GitCommit, State } from "../types";
import { aggregateAgentMessages } from "./aggregateAgentMessages";
import "./sketch-charts";
import "./sketch-chat-input";
import "./sketch-container-status";
import "./sketch-diff-view";
import { SketchDiffView } from "./sketch-diff-view";
import "./sketch-network-status";
import "./sketch-call-status";
import "./sketch-terminal";
import "./sketch-timeline";
import "./sketch-view-mode-select";
import "./sketch-restart-modal";

import { createRef, ref } from "lit/directives/ref.js";

type ViewMode = "chat" | "diff" | "charts" | "terminal";

@customElement("sketch-app-shell")
export class SketchAppShell extends LitElement {
  // Current view mode (chat, diff, charts, terminal)
  @state()
  viewMode: "chat" | "diff" | "charts" | "terminal" = "chat";

  // Current commit hash for diff view
  @state()
  currentCommitHash: string = "";

  // Last commit information
  @state()
  lastCommit: { hash: string; pushedBranch?: string } | null = null;

  // See https://lit.dev/docs/components/styles/ for how lit-element handles CSS.
  // Note that these styles only apply to the scope of this web component's
  // shadow DOM node, so they won't leak out or collide with CSS declared in
  // other components or the containing web page (...unless you want it to do that).
  static styles = css`
    /* Last commit display styling */
    .last-commit {
      display: flex;
      align-items: center;
      padding: 3px 8px;
      background: #f0f7ff;
      border: 1px solid #c8e1ff;
      border-radius: 4px;
      font-family: monospace;
      font-size: 12px;
      color: #0366d6;
      cursor: pointer;
      position: relative;
      margin: 0 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 180px;
      transition: background-color 0.2s ease;
    }

    .last-commit:hover {
      background-color: #dbedff;
    }

    .last-commit::before {
      content: "Last Commit: ";
      color: #666;
      margin-right: 4px;
      font-family: system-ui, sans-serif;
      font-size: 11px;
    }

    .copied-indicator {
      position: absolute;
      top: -20px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(40, 167, 69, 0.9);
      color: white;
      padding: 2px 6px;
      border-radius: 3px;
      font-size: 10px;
      font-family: system-ui, sans-serif;
      animation: fadeInOut 2s ease;
      pointer-events: none;
    }

    @keyframes fadeInOut {
      0% {
        opacity: 0;
      }
      20% {
        opacity: 1;
      }
      80% {
        opacity: 1;
      }
      100% {
        opacity: 0;
      }
    }

    .commit-branch-indicator {
      color: #28a745;
    }

    .commit-hash-indicator {
      color: #0366d6;
    }
    :host {
      display: block;
      font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Roboto,
        sans-serif;
      color: #333;
      line-height: 1.4;
      height: 100vh;
      width: 100%;
      position: relative;
      overflow-x: hidden;
      display: flex;
      flex-direction: column;
    }

    /* Top banner with combined elements */
    #top-banner {
      display: flex;
      align-self: stretch;
      justify-content: space-between;
      align-items: center;
      padding: 0 20px;
      margin-bottom: 0;
      border-bottom: 1px solid #eee;
      gap: 20px;
      background: white;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      width: 100%;
      height: 48px;
      padding-right: 30px; /* Extra padding on the right to prevent elements from hitting the edge */
    }

    /* View mode container styles - mirroring timeline.css structure */
    #view-container {
      align-self: stretch;
      overflow-y: auto;
      flex: 1;
    }

    #view-container-inner {
      max-width: 1200px;
      margin: 0 auto;
      position: relative;
      padding-bottom: 10px;
      padding-top: 10px;
    }

    #chat-input {
      align-self: flex-end;
      width: 100%;
      box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    }

    .banner-title {
      font-size: 18px;
      font-weight: 600;
      margin: 0;
      min-width: 6em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .chat-title {
      margin: 0;
      padding: 0;
      color: rgba(82, 82, 82, 0.85);
      font-size: 14px;
      font-weight: normal;
      font-style: italic;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Allow the container to expand to full width in diff mode */
    #view-container-inner.diff-active {
      max-width: 100%;
      width: 100%;
    }

    /* Individual view styles */
    .chat-view,
    .diff-view,
    .chart-view,
    .terminal-view {
      display: none; /* Hidden by default */
      width: 100%;
    }

    /* Active view styles - these will be applied via JavaScript */
    .view-active {
      display: flex;
      flex-direction: column;
    }

    .title-container {
      display: flex;
      flex-direction: column;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 30%;
      padding: 6px 0;
    }

    .refresh-control {
      display: flex;
      align-items: center;
      margin-bottom: 0;
      flex-wrap: nowrap;
      white-space: nowrap;
      flex-shrink: 0;
      gap: 15px;
      padding-left: 15px;
      margin-right: 50px;
    }

    .restart-button,
    .stop-button {
      background: #2196f3;
      color: white;
      border: none;
      padding: 4px 10px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      margin-right: 5px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .restart-button:hover {
      background-color: #0b7dda;
    }

    .restart-button:disabled {
      background-color: #ccc;
      cursor: not-allowed;
      opacity: 0.6;
    }

    .stop-button {
      background: #dc3545;
      color: white;
    }

    .stop-button:hover:not(:disabled) {
      background-color: #c82333;
    }

    .stop-button:disabled {
      background-color: #e9a8ad;
      cursor: not-allowed;
      opacity: 0.7;
    }

    .stop-button:disabled:hover {
      background-color: #e9a8ad;
    }

    .button-icon {
      width: 16px;
      height: 16px;
    }

    @media (max-width: 1400px) {
      .button-text {
        display: none;
      }

      .restart-button,
      .stop-button {
        padding: 6px;
      }
    }

    /* Removed poll-updates class */

    .notifications-toggle {
      display: flex;
      align-items: center;
      font-size: 12px;
      margin-right: 10px;
      cursor: pointer;
    }

    .bell-icon {
      width: 20px;
      height: 20px;
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    .bell-disabled::before {
      content: "";
      position: absolute;
      width: 2px;
      height: 24px;
      background-color: #dc3545;
      transform: rotate(45deg);
      transform-origin: center center;
    }
  `;

  // Header bar: Network connection status details
  @property()
  connectionStatus: ConnectionStatus = "disconnected";

  // Track if the last commit info has been copied
  @state()
  lastCommitCopied: boolean = false;

  // Track notification preferences
  @state()
  notificationsEnabled: boolean = false;

  // Track if the window is focused to control notifications
  @state()
  private _windowFocused: boolean = document.hasFocus();

  @property()
  connectionErrorMessage: string = "";

  // Chat messages
  @property({ attribute: false })
  messages: AgentMessage[] = [];

  @property()
  set title(value: string) {
    const oldValue = this._title;
    this._title = value;
    this.requestUpdate("title", oldValue);
    // Update document title when title property changes
    this.updateDocumentTitle();
  }

  get title(): string {
    return this._title;
  }

  private _title: string = "";

  private dataManager = new DataManager();

  @property({ attribute: false })
  containerState: State = {
    title: "",
    os: "",
    message_count: 0,
    hostname: "",
    working_dir: "",
    initial_commit: "",
    outstanding_llm_calls: 0,
    outstanding_tool_calls: [],
    session_id: "",
    ssh_available: false,
    ssh_error: "",
    in_container: false,
    first_message_index: 0,
  };

  @state()
  private restartModalOpen = false;

  // Mutation observer to detect when new messages are added
  private mutationObserver: MutationObserver | null = null;

  constructor() {
    super();

    // Binding methods to this
    this._handleViewModeSelect = this._handleViewModeSelect.bind(this);
    this._handleShowCommitDiff = this._handleShowCommitDiff.bind(this);
    this._handlePopState = this._handlePopState.bind(this);
    this._handleStopClick = this._handleStopClick.bind(this);
    this._handleNotificationsToggle =
      this._handleNotificationsToggle.bind(this);
    this._handleWindowFocus = this._handleWindowFocus.bind(this);
    this._handleWindowBlur = this._handleWindowBlur.bind(this);

    // Load notification preference from localStorage
    try {
      const savedPref = localStorage.getItem("sketch-notifications-enabled");
      if (savedPref !== null) {
        this.notificationsEnabled = savedPref === "true";
      }
    } catch (error) {
      console.error("Error loading notification preference:", error);
    }
  }

  // See https://lit.dev/docs/components/lifecycle/
  connectedCallback() {
    super.connectedCallback();

    // Initialize client-side nav history.
    const url = new URL(window.location.href);
    const mode = url.searchParams.get("view") || "chat";
    window.history.replaceState({ mode }, "", url.toString());

    this.toggleViewMode(mode as ViewMode, false);
    // Add popstate event listener to handle browser back/forward navigation
    window.addEventListener("popstate", this._handlePopState);

    // Add event listeners
    window.addEventListener("view-mode-select", this._handleViewModeSelect);
    window.addEventListener("show-commit-diff", this._handleShowCommitDiff);

    // Add window focus/blur listeners for controlling notifications
    window.addEventListener("focus", this._handleWindowFocus);
    window.addEventListener("blur", this._handleWindowBlur);

    // register event listeners
    this.dataManager.addEventListener(
      "dataChanged",
      this.handleDataChanged.bind(this),
    );
    this.dataManager.addEventListener(
      "connectionStatusChanged",
      this.handleConnectionStatusChanged.bind(this),
    );

    // Set initial document title
    this.updateDocumentTitle();

    // Initialize the data manager
    this.dataManager.initialize();

    // Process existing messages for commit info
    if (this.messages && this.messages.length > 0) {
      this.updateLastCommitInfo(this.messages);
    }
  }

  // See https://lit.dev/docs/components/lifecycle/
  disconnectedCallback() {
    super.disconnectedCallback();
    window.removeEventListener("popstate", this._handlePopState);

    // Remove event listeners
    window.removeEventListener("view-mode-select", this._handleViewModeSelect);
    window.removeEventListener("show-commit-diff", this._handleShowCommitDiff);
    window.removeEventListener("focus", this._handleWindowFocus);
    window.removeEventListener("blur", this._handleWindowBlur);

    // unregister data manager event listeners
    this.dataManager.removeEventListener(
      "dataChanged",
      this.handleDataChanged.bind(this),
    );
    this.dataManager.removeEventListener(
      "connectionStatusChanged",
      this.handleConnectionStatusChanged.bind(this),
    );

    // Disconnect mutation observer if it exists
    if (this.mutationObserver) {
      this.mutationObserver.disconnect();
      this.mutationObserver = null;
    }
  }

  updateUrlForViewMode(mode: "chat" | "diff" | "charts" | "terminal"): void {
    // Get the current URL without search parameters
    const url = new URL(window.location.href);

    // Clear existing parameters
    url.search = "";

    // Only add view parameter if not in default chat view
    if (mode !== "chat") {
      url.searchParams.set("view", mode);
      const diffView = this.shadowRoot?.querySelector(
        ".diff-view",
      ) as SketchDiffView;

      // If in diff view and there's a commit hash, include that too
      if (mode === "diff" && diffView.commitHash) {
        url.searchParams.set("commit", diffView.commitHash);
      }
    }

    // Update the browser history without reloading the page
    window.history.pushState({ mode }, "", url.toString());
  }

  private _handlePopState(event: PopStateEvent) {
    if (event.state && event.state.mode) {
      this.toggleViewMode(event.state.mode, false);
    } else {
      this.toggleViewMode("chat", false);
    }
  }

  /**
   * Handle view mode selection event
   */
  private _handleViewModeSelect(event: CustomEvent) {
    const mode = event.detail.mode as "chat" | "diff" | "charts" | "terminal";
    this.toggleViewMode(mode, true);
  }

  /**
   * Handle show commit diff event
   */
  private _handleShowCommitDiff(event: CustomEvent) {
    const { commitHash } = event.detail;
    if (commitHash) {
      this.showCommitDiff(commitHash);
    }
  }

  /**
   * Listen for commit diff event
   * @param commitHash The commit hash to show diff for
   */
  private showCommitDiff(commitHash: string): void {
    // Store the commit hash
    this.currentCommitHash = commitHash;

    // Switch to diff view
    this.toggleViewMode("diff", true);

    // Wait for DOM update to complete
    this.updateComplete.then(() => {
      // Get the diff view component
      const diffView = this.shadowRoot?.querySelector("sketch-diff-view");
      if (diffView) {
        // Call the showCommitDiff method
        (diffView as any).showCommitDiff(commitHash);
      }
    });
  }

  /**
   * Toggle between different view modes: chat, diff, charts, terminal
   */
  private toggleViewMode(mode: ViewMode, updateHistory: boolean): void {
    // Don't do anything if the mode is already active
    if (this.viewMode === mode) return;

    // Update the view mode
    this.viewMode = mode;

    if (updateHistory) {
      // Update URL with the current view mode
      this.updateUrlForViewMode(mode);
    }

    // Wait for DOM update to complete
    this.updateComplete.then(() => {
      // Update active view
      const viewContainerInner = this.shadowRoot?.querySelector(
        "#view-container-inner",
      );
      const chatView = this.shadowRoot?.querySelector(".chat-view");
      const diffView = this.shadowRoot?.querySelector(".diff-view");
      const chartView = this.shadowRoot?.querySelector(".chart-view");
      const terminalView = this.shadowRoot?.querySelector(".terminal-view");

      // Remove active class from all views
      chatView?.classList.remove("view-active");
      diffView?.classList.remove("view-active");
      chartView?.classList.remove("view-active");
      terminalView?.classList.remove("view-active");

      // Add/remove diff-active class on view container
      if (mode === "diff") {
        viewContainerInner?.classList.add("diff-active");
      } else {
        viewContainerInner?.classList.remove("diff-active");
      }

      // Add active class to the selected view
      switch (mode) {
        case "chat":
          chatView?.classList.add("view-active");
          break;
        case "diff":
          diffView?.classList.add("view-active");
          // Load diff content if we have a diff view
          const diffViewComp =
            this.shadowRoot?.querySelector("sketch-diff-view");
          if (diffViewComp && this.currentCommitHash) {
            (diffViewComp as any).showCommitDiff(this.currentCommitHash);
          } else if (diffViewComp) {
            (diffViewComp as any).loadDiffContent();
          }
          break;
        case "charts":
          chartView?.classList.add("view-active");
          break;
        case "terminal":
          terminalView?.classList.add("view-active");
          break;
      }

      // Update view mode buttons
      const viewModeSelect = this.shadowRoot?.querySelector(
        "sketch-view-mode-select",
      );
      if (viewModeSelect) {
        const event = new CustomEvent("update-active-mode", {
          detail: { mode },
          bubbles: true,
          composed: true,
        });
        viewModeSelect.dispatchEvent(event);
      }

      // FIXME: This is a hack to get vega chart in sketch-charts.ts to work properly
      // When the chart is in the background, its container has a width of 0, so vega
      // renders width 0 and only changes that width on a resize event.
      // See https://github.com/vega/react-vega/issues/85#issuecomment-1826421132
      window.dispatchEvent(new Event("resize"));
    });
  }

  /**
   * Updates the document title based on current title and connection status
   */
  private updateDocumentTitle(): void {
    let docTitle = `sk: ${this.title || "untitled"}`;

    // Add red circle emoji if disconnected
    if (this.connectionStatus === "disconnected") {
      docTitle += " 🔴";
    }

    document.title = docTitle;
  }

  // Check and request notification permission if needed
  private async checkNotificationPermission(): Promise<boolean> {
    // Check if the Notification API is supported
    if (!("Notification" in window)) {
      console.log("This browser does not support notifications");
      return false;
    }

    // Check if permission is already granted
    if (Notification.permission === "granted") {
      return true;
    }

    // If permission is not denied, request it
    if (Notification.permission !== "denied") {
      const permission = await Notification.requestPermission();
      return permission === "granted";
    }

    return false;
  }

  // Handle notifications toggle click
  private _handleNotificationsToggle(): void {
    this.notificationsEnabled = !this.notificationsEnabled;

    // If enabling notifications, check permissions
    if (this.notificationsEnabled) {
      this.checkNotificationPermission();
    }

    // Save preference to localStorage
    try {
      localStorage.setItem(
        "sketch-notifications-enabled",
        String(this.notificationsEnabled),
      );
    } catch (error) {
      console.error("Error saving notification preference:", error);
    }
  }

  // Handle window focus event
  private _handleWindowFocus(): void {
    this._windowFocused = true;
  }

  // Handle window blur event
  private _handleWindowBlur(): void {
    this._windowFocused = false;
  }

  // Show notification for message with EndOfTurn=true
  private async showEndOfTurnNotification(
    message: AgentMessage,
  ): Promise<void> {
    // Don't show notifications if they're disabled
    if (!this.notificationsEnabled) return;

    // Don't show notifications if the window is focused
    if (this._windowFocused) return;

    // Check if we have permission to show notifications
    const hasPermission = await this.checkNotificationPermission();
    if (!hasPermission) return;

    // Only show notifications for agent messages with end_of_turn=true and no parent_conversation_id
    if (
      message.type !== "agent" ||
      !message.end_of_turn ||
      message.parent_conversation_id
    )
      return;

    // Create a title that includes the sketch title
    const notificationTitle = `Sketch: ${this.title || "untitled"}`;

    // Extract the beginning of the message content (first 100 chars)
    const messagePreview = message.content
      ? message.content.substring(0, 100) +
        (message.content.length > 100 ? "..." : "")
      : "Agent has completed its turn";

    // Create and show the notification
    try {
      new Notification(notificationTitle, {
        body: messagePreview,
        icon: "https://sketch.dev/favicon.ico", // Use sketch.dev favicon for notification
      });
    } catch (error) {
      console.error("Error showing notification:", error);
    }
  }

  private handleDataChanged(eventData: {
    state: State;
    newMessages: AgentMessage[];
  }): void {
    const { state, newMessages } = eventData;

    // Update state if we received it
    if (state) {
      // Ensure we're using the latest call status to prevent indicators from being stuck
      if (
        state.outstanding_llm_calls === 0 &&
        state.outstanding_tool_calls.length === 0
      ) {
        // Force reset containerState calls when nothing is reported as in progress
        state.outstanding_llm_calls = 0;
        state.outstanding_tool_calls = [];
      }

      this.containerState = state;
      this.title = state.title;

      // Update document title when sketch title changes
      this.updateDocumentTitle();
    }

    // Update messages
    this.messages = aggregateAgentMessages(this.messages, newMessages);

    // Process new messages to find commit messages
    this.updateLastCommitInfo(newMessages);

    // Check for agent messages with end_of_turn=true and show notifications
    if (newMessages && newMessages.length > 0) {
      for (const message of newMessages) {
        if (
          message.type === "agent" &&
          message.end_of_turn &&
          !message.parent_conversation_id
        ) {
          this.showEndOfTurnNotification(message);
          break; // Only show one notification per batch of messages
        }
      }
    }
  }

  private handleConnectionStatusChanged(
    status: ConnectionStatus,
    errorMessage?: string,
  ): void {
    this.connectionStatus = status;
    this.connectionErrorMessage = errorMessage || "";

    // Update document title when connection status changes
    this.updateDocumentTitle();
  }

  // Update last commit information when new messages arrive
  private updateLastCommitInfo(newMessages: AgentMessage[]): void {
    if (!newMessages || newMessages.length === 0) return;

    // Process messages in chronological order (latest last)
    for (const message of newMessages) {
      if (
        message.type === "commit" &&
        message.commits &&
        message.commits.length > 0
      ) {
        // Get the first commit from the list
        const commit = message.commits[0];
        if (commit) {
          this.lastCommit = {
            hash: commit.hash,
            pushedBranch: commit.pushed_branch,
          };
          this.lastCommitCopied = false;
        }
      }
    }
  }

  // Copy commit info to clipboard
  private copyCommitInfo(event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();

    if (!this.lastCommit) return;

    const textToCopy =
      this.lastCommit.pushedBranch || this.lastCommit.hash.substring(0, 8);

    navigator.clipboard
      .writeText(textToCopy)
      .then(() => {
        this.lastCommitCopied = true;
        // Reset the copied state after 2 seconds
        setTimeout(() => {
          this.lastCommitCopied = false;
        }, 2000);
      })
      .catch((err) => {
        console.error("Failed to copy commit info:", err);
      });
  }

  private async _handleStopClick(): Promise<void> {
    try {
      const response = await fetch("cancel", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ reason: "user requested cancellation" }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(
          `Failed to stop operation: ${response.status} - ${errorData}`,
        );
      }

      // Stop request sent
    } catch (error) {
      console.error("Error stopping operation:", error);
    }
  }

  openRestartModal() {
    this.restartModalOpen = true;
  }

  handleRestartModalClose() {
    this.restartModalOpen = false;
  }

  async _sendChat(e: CustomEvent) {
    console.log("app shell: _sendChat", e);
    const message = e.detail.message?.trim();
    if (message == "") {
      return;
    }
    try {
      // Send the message to the server
      const response = await fetch("chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Server error: ${response.status} - ${errorData}`);
      }

      // TOOD(philip): If the data manager is getting messages out of order, there's a bug?
      // Reset data manager state to force a full refresh after sending a message
      // This ensures we get all messages in the correct order
      // Use private API for now - TODO: add a resetState() method to DataManager
      (this.dataManager as any).nextFetchIndex = 0;
      (this.dataManager as any).currentFetchStartIndex = 0;

      // // If in diff view, switch to conversation view
      // if (this.viewMode === "diff") {
      //   await this.toggleViewMode("chat");
      // }

      // Refresh the timeline data to show the new message
      await this.dataManager.fetchData();
    } catch (error) {
      console.error("Error sending chat message:", error);
      const statusText = document.getElementById("statusText");
      if (statusText) {
        statusText.textContent = "Error sending message";
      }
    }
  }

  private scrollContainerRef = createRef<HTMLElement>();

  render() {
    return html`
      <div id="top-banner">
        <div class="title-container">
          <h1 class="banner-title">sketch</h1>
          <h2 id="chatTitle" class="chat-title">${this.title}</h2>
        </div>

        <!-- Container status info moved above tabs -->
        <sketch-container-status
          .state=${this.containerState}
        ></sketch-container-status>

        <!-- Views section with tabs - repositioned -->
        <sketch-view-mode-select></sketch-view-mode-select>

        ${this.lastCommit
          ? html`
              <div
                class="last-commit"
                @click=${(e: MouseEvent) => this.copyCommitInfo(e)}
                title="Click to copy"
              >
                ${this.lastCommitCopied
                  ? html`<span class="copied-indicator">Copied!</span>`
                  : ""}
                ${this.lastCommit.pushedBranch
                  ? html`<span class="commit-branch-indicator"
                      >${this.lastCommit.pushedBranch}</span
                    >`
                  : html`<span class="commit-hash-indicator"
                      >${this.lastCommit.hash.substring(0, 8)}</span
                    >`}
              </div>
            `
          : ""}

        <div class="refresh-control">
          <button
            id="restartButton"
            class="restart-button"
            ?disabled=${this.containerState.message_count === 0}
            @click=${this.openRestartModal}
          >
            <svg
              class="button-icon"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
            </svg>
            <span class="button-text">Restart</span>
          </button>
          <button
            id="stopButton"
            class="stop-button"
            ?disabled=${(this.containerState?.outstanding_llm_calls || 0) ===
              0 &&
            (this.containerState?.outstanding_tool_calls || []).length === 0}
          >
            <svg
              class="button-icon"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <rect x="6" y="6" width="12" height="12" />
            </svg>
            <span class="button-text">Stop</span>
          </button>

          <div
            class="notifications-toggle"
            @click=${this._handleNotificationsToggle}
            title="${this.notificationsEnabled
              ? "Disable"
              : "Enable"} notifications when the agent completes its turn"
          >
            <div
              class="bell-icon ${!this.notificationsEnabled
                ? "bell-disabled"
                : ""}"
            >
              <!-- Bell SVG icon -->
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                fill="currentColor"
                viewBox="0 0 16 16"
              >
                <path
                  d="M8 16a2 2 0 0 0 2-2H6a2 2 0 0 0 2 2zM8 1.918l-.797.161A4.002 4.002 0 0 0 4 6c0 .628-.134 2.197-.459 3.742-.16.767-.376 1.566-.663 2.258h10.244c-.287-.692-.502-1.49-.663-2.258C12.134 8.197 12 6.628 12 6a4.002 4.002 0 0 0-3.203-3.92L8 1.917zM14.22 12c.223.447.481.801.78 1H1c.299-.199.557-.553.78-1C2.68 10.2 3 6.88 3 6c0-2.42 1.72-4.44 4.005-4.901a1 1 0 1 1 1.99 0A5.002 5.002 0 0 1 13 6c0 .88.32 4.2 1.22 6z"
                />
              </svg>
            </div>
          </div>

          <sketch-network-status
            connection=${this.connectionStatus}
            error=${this.connectionErrorMessage}
          ></sketch-network-status>

          <sketch-call-status
            .agentState=${this.containerState?.agent_state}
            .llmCalls=${this.containerState?.outstanding_llm_calls || 0}
            .toolCalls=${this.containerState?.outstanding_tool_calls || []}
          ></sketch-call-status>
        </div>
      </div>

      <div id="view-container" ${ref(this.scrollContainerRef)}>
        <div id="view-container-inner">
          <div
            class="chat-view ${this.viewMode === "chat" ? "view-active" : ""}"
          >
            <sketch-timeline
              .messages=${this.messages}
              .scrollContainer=${this.scrollContainerRef}
            ></sketch-timeline>
          </div>
          <div
            class="diff-view ${this.viewMode === "diff" ? "view-active" : ""}"
          >
            <sketch-diff-view
              .commitHash=${this.currentCommitHash}
            ></sketch-diff-view>
          </div>
          <div
            class="chart-view ${this.viewMode === "charts"
              ? "view-active"
              : ""}"
          >
            <sketch-charts .messages=${this.messages}></sketch-charts>
          </div>
          <div
            class="terminal-view ${this.viewMode === "terminal"
              ? "view-active"
              : ""}"
          >
            <sketch-terminal></sketch-terminal>
          </div>
        </div>
      </div>

      <div id="chat-input">
        <sketch-chat-input @send-chat="${this._sendChat}"></sketch-chat-input>
      </div>

      <sketch-restart-modal
        ?open=${this.restartModalOpen}
        @close=${this.handleRestartModalClose}
        .containerState=${this.containerState}
        .messages=${this.messages}
      ></sketch-restart-modal>
    `;
  }

  /**
   * Lifecycle callback when component is first connected to DOM
   */
  firstUpdated(): void {
    if (this.viewMode !== "chat") {
      return;
    }

    // Initial scroll to bottom when component is first rendered
    setTimeout(
      () => this.scrollTo({ top: this.scrollHeight, behavior: "smooth" }),
      50,
    );

    // Setup stop button
    const stopButton = this.renderRoot?.querySelector(
      "#stopButton",
    ) as HTMLButtonElement;
    stopButton?.addEventListener("click", async () => {
      try {
        const response = await fetch("cancel", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ reason: "User clicked stop button" }),
        });
        if (!response.ok) {
          console.error("Failed to cancel:", await response.text());
        }
      } catch (error) {
        console.error("Error cancelling operation:", error);
      }
    });

    // Always enable polling by default
    this.dataManager.setPollingEnabled(true);

    // Process any existing messages to find commit information
    if (this.messages && this.messages.length > 0) {
      this.updateLastCommitInfo(this.messages);
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "sketch-app-shell": SketchAppShell;
  }
}

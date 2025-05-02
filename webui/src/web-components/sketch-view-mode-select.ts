import { css, html, LitElement } from "lit";
import { customElement, property } from "lit/decorators.js";
import "./sketch-container-status";

@customElement("sketch-view-mode-select")
export class SketchViewModeSelect extends LitElement {
  // Current active mode
  @property()
  activeMode: "chat" | "diff" | "charts" | "terminal" = "chat";
  // Header bar: view mode buttons

  static styles = css`
    /* Tab-style View Mode Styles */
    .tab-nav {
      display: flex;
      margin-right: 10px;
      background-color: #f8f8f8;
      border-radius: 4px;
      overflow: hidden;
      border: 1px solid #ddd;
    }

    .tab-btn {
      padding: 8px 12px;
      background: none;
      border: none;
      cursor: pointer;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 5px;
      color: #666;
      border-bottom: 2px solid transparent;
      transition: all 0.2s ease;
      white-space: nowrap;
    }

    @media (max-width: 1400px) {
      .tab-btn span:not(.tab-icon) {
        display: none;
      }

      .tab-btn {
        padding: 8px 10px;
      }
    }

    .tab-btn:not(:last-child) {
      border-right: 1px solid #eee;
    }

    .tab-btn:hover {
      background-color: #f0f0f0;
    }

    .tab-btn.active {
      border-bottom: 2px solid #4a90e2;
      color: #4a90e2;
      font-weight: 500;
      background-color: #e6f7ff;
    }

    .tab-icon {
      font-size: 16px;
    }
  `;

  constructor() {
    super();

    // Binding methods
    this._handleViewModeClick = this._handleViewModeClick.bind(this);
    this._handleUpdateActiveMode = this._handleUpdateActiveMode.bind(this);
  }

  // See https://lit.dev/docs/components/lifecycle/
  connectedCallback() {
    super.connectedCallback();

    // Listen for update-active-mode events
    this.addEventListener(
      "update-active-mode",
      this._handleUpdateActiveMode as EventListener,
    );
  }

  /**
   * Handle view mode button clicks
   */
  private _handleViewModeClick(mode: "chat" | "diff" | "charts" | "terminal") {
    // Dispatch a custom event to notify the app shell to change the view
    const event = new CustomEvent("view-mode-select", {
      detail: { mode },
      bubbles: true,
      composed: true,
    });
    this.dispatchEvent(event);
  }

  /**
   * Handle updates to the active mode
   */
  private _handleUpdateActiveMode(event: CustomEvent) {
    const { mode } = event.detail;
    if (mode) {
      this.activeMode = mode;
    }
  }

  // See https://lit.dev/docs/components/lifecycle/
  disconnectedCallback() {
    super.disconnectedCallback();

    // Remove event listeners
    this.removeEventListener(
      "update-active-mode",
      this._handleUpdateActiveMode as EventListener,
    );
  }

  render() {
    return html`
      <div class="tab-nav">
        <button
          id="showConversationButton"
          class="tab-btn ${this.activeMode === "chat" ? "active" : ""}"
          title="Conversation View"
          @click=${() => this._handleViewModeClick("chat")}
        >
          <span class="tab-icon">💬</span>
          <span>Chat</span>
        </button>
        <button
          id="showDiffButton"
          class="tab-btn ${this.activeMode === "diff" ? "active" : ""}"
          title="Diff View"
          @click=${() => this._handleViewModeClick("diff")}
        >
          <span class="tab-icon">±</span>
          <span>Diff</span>
        </button>
        <button
          id="showChartsButton"
          class="tab-btn ${this.activeMode === "charts" ? "active" : ""}"
          title="Charts View"
          @click=${() => this._handleViewModeClick("charts")}
        >
          <span class="tab-icon">📈</span>
          <span>Charts</span>
        </button>
        <button
          id="showTerminalButton"
          class="tab-btn ${this.activeMode === "terminal" ? "active" : ""}"
          title="Terminal View"
          @click=${() => this._handleViewModeClick("terminal")}
        >
          <span class="tab-icon">💻</span>
          <span>Terminal</span>
        </button>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "sketch-view-mode-select": SketchViewModeSelect;
  }
}

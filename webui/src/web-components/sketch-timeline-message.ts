import { css, html, LitElement } from "lit";
import { unsafeHTML } from "lit/directives/unsafe-html.js";
import { customElement, property } from "lit/decorators.js";
import { AgentMessage } from "../types";
import { marked, MarkedOptions, Renderer, Tokens } from "marked";
import mermaid from "mermaid";
import "./sketch-tool-calls";
@customElement("sketch-timeline-message")
export class SketchTimelineMessage extends LitElement {
  @property()
  message: AgentMessage;

  @property()
  previousMessage: AgentMessage;

  @property()
  open: boolean = false;

  // See https://lit.dev/docs/components/styles/ for how lit-element handles CSS.
  // Note that these styles only apply to the scope of this web component's
  // shadow DOM node, so they won't leak out or collide with CSS declared in
  // other components or the containing web page (...unless you want it to do that).
  static styles = css`
    .message {
      position: relative;
      margin-bottom: 5px;
      padding-left: 30px;
    }

    .message-icon {
      position: absolute;
      left: 10px;
      top: 0;
      transform: translateX(-50%);
      width: 16px;
      height: 16px;
      border-radius: 3px;
      text-align: center;
      line-height: 16px;
      color: #fff;
      font-size: 10px;
    }

    .message-content {
      position: relative;
      padding: 5px 10px;
      background: #fff;
      border-radius: 3px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
      border-left: 3px solid transparent;
    }

    /* Copy button styles */
    .message-text-container,
    .tool-result-container {
      position: relative;
    }

    .message-actions {
      position: absolute;
      top: 5px;
      right: 5px;
      z-index: 10;
      opacity: 0;
      transition: opacity 0.2s ease;
    }

    .message-text-container:hover .message-actions,
    .tool-result-container:hover .message-actions {
      opacity: 1;
    }

    .copy-button {
      background-color: rgba(255, 255, 255, 0.9);
      border: 1px solid #ddd;
      border-radius: 4px;
      color: #555;
      cursor: pointer;
      font-size: 12px;
      padding: 2px 8px;
      transition: all 0.2s ease;
    }

    .copy-button:hover {
      background-color: #f0f0f0;
      color: #333;
    }

    /* Removed arrow decoration for a more compact look */

    .message-header {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-bottom: 3px;
      font-size: 12px;
    }

    .message-timestamp {
      font-size: 10px;
      color: #888;
      font-style: italic;
      margin-left: 3px;
    }

    .message-usage {
      font-size: 10px;
      color: #888;
      margin-left: 3px;
    }

    .conversation-id {
      font-family: monospace;
      font-size: 12px;
      padding: 2px 4px;
      background-color: #f0f0f0;
      border-radius: 3px;
      margin-left: auto;
    }

    .parent-info {
      font-size: 11px;
      opacity: 0.8;
    }

    .subconversation {
      border-left: 2px solid transparent;
      padding-left: 5px;
      margin-left: 20px;
      transition: margin-left 0.3s ease;
    }

    .message-text {
      overflow-x: auto;
      margin-bottom: 3px;
      font-family: monospace;
      padding: 3px 5px;
      background: rgb(236, 236, 236);
      border-radius: 6px;
      user-select: text;
      cursor: text;
      -webkit-user-select: text;
      -moz-user-select: text;
      -ms-user-select: text;
      font-size: 13px;
      line-height: 1.3;
    }

    .tool-details {
      margin-top: 3px;
      padding-top: 3px;
      border-top: 1px dashed #e0e0e0;
      font-size: 12px;
    }

    .tool-name {
      font-size: 12px;
      font-weight: bold;
      margin-bottom: 2px;
      background: #f0f0f0;
      padding: 2px 4px;
      border-radius: 2px;
      display: flex;
      align-items: center;
      gap: 3px;
    }

    .tool-input,
    .tool-result {
      margin-top: 2px;
      padding: 3px 5px;
      background: #f7f7f7;
      border-radius: 2px;
      font-family: monospace;
      font-size: 12px;
      overflow-x: auto;
      white-space: pre;
      line-height: 1.3;
      user-select: text;
      cursor: text;
      -webkit-user-select: text;
      -moz-user-select: text;
      -ms-user-select: text;
    }

    .tool-result {
      max-height: 300px;
      overflow-y: auto;
    }

    .usage-info {
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px dashed #e0e0e0;
      font-size: 12px;
      color: #666;
    }

    /* Custom styles for IRC-like experience */
    .user .message-content {
      border-left-color: #2196f3;
    }

    .agent .message-content {
      border-left-color: #4caf50;
    }

    .tool .message-content {
      border-left-color: #ff9800;
    }

    .error .message-content {
      border-left-color: #f44336;
    }

    /* Make message type display bold but without the IRC-style markers */
    .message-type {
      font-weight: bold;
    }

    /* Commit message styling */
    .message.commit {
      background-color: #f0f7ff;
      border-left: 4px solid #0366d6;
    }

    .commits-container {
      margin-top: 10px;
      padding: 5px;
    }

    .commits-header {
      font-weight: bold;
      margin-bottom: 5px;
      color: #24292e;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .commit-boxes-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }

    .commit-box {
      border: 1px solid #d1d5da;
      border-radius: 4px;
      overflow: hidden;
      background-color: #ffffff;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
      max-width: 100%;
      display: flex;
      flex-direction: column;
    }

    .commit-preview {
      padding: 8px 12px;
      font-family: monospace;
      background-color: #f6f8fa;
      border-bottom: 1px dashed #d1d5da;
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 4px;
    }

    .commit-preview:hover {
      background-color: #eef2f6;
    }

    .commit-hash {
      color: #0366d6;
      font-weight: bold;
      cursor: pointer;
      margin-right: 8px;
      text-decoration: none;
      position: relative;
    }

    .commit-hash:hover {
      text-decoration: underline;
    }

    .commit-hash:hover::after {
      content: "📋";
      font-size: 10px;
      position: absolute;
      top: -8px;
      right: -12px;
      opacity: 0.7;
    }

    .branch-wrapper {
      margin-right: 8px;
      color: #555;
    }

    .commit-branch {
      color: #28a745;
      font-weight: 500;
      cursor: pointer;
      text-decoration: none;
      position: relative;
    }

    .commit-branch:hover {
      text-decoration: underline;
    }

    .commit-branch:hover::after {
      content: "📋";
      font-size: 10px;
      position: absolute;
      top: -8px;
      right: -12px;
      opacity: 0.7;
    }

    .commit-preview {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 4px;
    }

    .commit-details {
      padding: 8px 12px;
      max-height: 200px;
      overflow-y: auto;
    }

    .commit-details pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .commit-details.is-hidden {
      display: none;
    }

    .pushed-branch {
      color: #28a745;
      font-weight: 500;
      margin-left: 6px;
    }

    .commit-diff-button {
      padding: 3px 6px;
      border: 1px solid #ccc;
      border-radius: 3px;
      background-color: #f7f7f7;
      color: #24292e;
      font-size: 11px;
      cursor: pointer;
      transition: all 0.2s ease;
      margin-left: auto;
    }

    .commit-diff-button:hover {
      background-color: #e7e7e7;
      border-color: #aaa;
    }

    /* Tool call cards */
    .tool-call-cards-container {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 8px;
    }

    /* Message type styles */

    .user .message-icon {
      background-color: #2196f3;
    }

    .agent .message-icon {
      background-color: #4caf50;
    }

    .tool .message-icon {
      background-color: #ff9800;
    }

    .error .message-icon {
      background-color: #f44336;
    }

    .end-of-turn {
      margin-bottom: 15px;
    }

    .end-of-turn::after {
      content: "End of Turn";
      position: absolute;
      left: 15px;
      bottom: -10px;
      transform: translateX(-50%);
      font-size: 10px;
      color: #666;
      background: #f0f0f0;
      padding: 1px 4px;
      border-radius: 3px;
    }

    .markdown-content {
      box-sizing: border-box;
      min-width: 200px;
      margin: 0 auto;
    }

    .markdown-content p {
      margin-block-start: 0.5em;
      margin-block-end: 0.5em;
    }

    /* Mermaid diagram styling */
    .mermaid-container {
      margin: 1em 0;
      padding: 0.5em;
      background-color: #f8f8f8;
      border-radius: 4px;
      overflow-x: auto;
    }

    .mermaid {
      text-align: center;
    }
  `;

  // Track mermaid diagrams that need rendering
  private mermaidDiagrams = new Map();

  constructor() {
    super();
    // Initialize mermaid with specific config
    mermaid.initialize({
      startOnLoad: false,
      suppressErrorRendering: true,
      theme: "default",
      securityLevel: "loose", // Allows more flexibility but be careful with user-generated content
      fontFamily: "monospace",
    });
  }

  // See https://lit.dev/docs/components/lifecycle/
  connectedCallback() {
    super.connectedCallback();
  }

  // After the component is updated and rendered, render any mermaid diagrams
  updated(changedProperties: Map<string, unknown>) {
    super.updated(changedProperties);
    this.renderMermaidDiagrams();
  }

  // Render mermaid diagrams after the component is updated
  renderMermaidDiagrams() {
    // Add a small delay to ensure the DOM is fully rendered
    setTimeout(() => {
      // Find all mermaid containers in our shadow root
      const containers = this.shadowRoot?.querySelectorAll(".mermaid");
      if (!containers || containers.length === 0) return;

      // Process each mermaid diagram
      containers.forEach((container) => {
        const id = container.id;
        const code = container.textContent || "";
        if (!code || !id) return; // Use return for forEach instead of continue

        try {
          // Clear any previous content
          container.innerHTML = code;

          // Render the mermaid diagram using promise
          mermaid
            .render(`${id}-svg`, code)
            .then(({ svg }) => {
              container.innerHTML = svg;
            })
            .catch((err) => {
              console.error("Error rendering mermaid diagram:", err);
              // Show the original code as fallback
              container.innerHTML = `<pre>${code}</pre>`;
            });
        } catch (err) {
          console.error("Error processing mermaid diagram:", err);
          // Show the original code as fallback
          container.innerHTML = `<pre>${code}</pre>`;
        }
      });
    }, 100); // Small delay to ensure DOM is ready
  }

  // See https://lit.dev/docs/components/lifecycle/
  disconnectedCallback() {
    super.disconnectedCallback();
  }

  renderMarkdown(markdownContent: string): string {
    try {
      // Create a custom renderer
      const renderer = new Renderer();
      const originalCodeRenderer = renderer.code.bind(renderer);

      // Override the code renderer to handle mermaid diagrams
      renderer.code = function ({ text, lang, escaped }: Tokens.Code): string {
        if (lang === "mermaid") {
          // Generate a unique ID for this diagram
          const id = `mermaid-diagram-${Math.random().toString(36).substring(2, 10)}`;

          // Just create the container and mermaid div - we'll render it in the updated() lifecycle method
          return `<div class="mermaid-container">
                   <div class="mermaid" id="${id}">${text}</div>
                 </div>`;
        }
        // Default rendering for other code blocks
        return originalCodeRenderer({ text, lang, escaped });
      };

      // Set markdown options for proper code block highlighting and safety
      const markedOptions: MarkedOptions = {
        gfm: true, // GitHub Flavored Markdown
        breaks: true, // Convert newlines to <br>
        async: false,
        renderer: renderer,
        // DOMPurify is recommended for production, but not included in this implementation
      };
      return marked.parse(markdownContent, markedOptions) as string;
    } catch (error) {
      console.error("Error rendering markdown:", error);
      // Fallback to plain text if markdown parsing fails
      return markdownContent;
    }
  }

  /**
   * Format timestamp for display
   */
  formatTimestamp(
    timestamp: string | number | Date | null | undefined,
    defaultValue: string = "",
  ): string {
    if (!timestamp) return defaultValue;
    try {
      const date = new Date(timestamp);
      if (isNaN(date.getTime())) return defaultValue;

      // Format: Mar 13, 2025 09:53:25 AM
      return date.toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
      });
    } catch (e) {
      return defaultValue;
    }
  }

  formatNumber(
    num: number | null | undefined,
    defaultValue: string = "0",
  ): string {
    if (num === undefined || num === null) return defaultValue;
    try {
      return num.toLocaleString();
    } catch (e) {
      return String(num);
    }
  }
  formatCurrency(
    num: number | string | null | undefined,
    defaultValue: string = "$0.00",
    isMessageLevel: boolean = false,
  ): string {
    if (num === undefined || num === null) return defaultValue;
    try {
      // Use 4 decimal places for message-level costs, 2 for totals
      const decimalPlaces = isMessageLevel ? 4 : 2;
      return `$${parseFloat(String(num)).toFixed(decimalPlaces)}`;
    } catch (e) {
      return defaultValue;
    }
  }

  showCommit(commitHash: string) {
    this.dispatchEvent(
      new CustomEvent("show-commit-diff", {
        bubbles: true,
        composed: true,
        detail: { commitHash },
      }),
    );
  }

  copyToClipboard(text: string, event: Event) {
    const element = event.currentTarget as HTMLElement;
    const rect = element.getBoundingClientRect();

    navigator.clipboard
      .writeText(text)
      .then(() => {
        this.showFloatingMessage("Copied!", rect, "success");
      })
      .catch((err) => {
        console.error("Failed to copy text: ", err);
        this.showFloatingMessage("Failed to copy!", rect, "error");
      });
  }

  showFloatingMessage(
    message: string,
    targetRect: DOMRect,
    type: "success" | "error",
  ) {
    // Create floating message element
    const floatingMsg = document.createElement("div");
    floatingMsg.textContent = message;
    floatingMsg.className = `floating-message ${type}`;

    // Position it near the clicked element
    // Position just above the element
    const top = targetRect.top - 30;
    const left = targetRect.left + targetRect.width / 2 - 40;

    floatingMsg.style.position = "fixed";
    floatingMsg.style.top = `${top}px`;
    floatingMsg.style.left = `${left}px`;
    floatingMsg.style.zIndex = "9999";

    // Add to document body
    document.body.appendChild(floatingMsg);

    // Animate in
    floatingMsg.style.opacity = "0";
    floatingMsg.style.transform = "translateY(10px)";

    setTimeout(() => {
      floatingMsg.style.opacity = "1";
      floatingMsg.style.transform = "translateY(0)";
    }, 10);

    // Remove after animation
    setTimeout(() => {
      floatingMsg.style.opacity = "0";
      floatingMsg.style.transform = "translateY(-10px)";

      setTimeout(() => {
        document.body.removeChild(floatingMsg);
      }, 300);
    }, 1500);
  }

  render() {
    return html`
      <div
        class="message ${this.message?.type} ${this.message?.end_of_turn
          ? "end-of-turn"
          : ""}"
      >
        ${this.previousMessage?.type != this.message?.type
          ? html`<div class="message-icon">
              ${this.message?.type.toUpperCase()[0]}
            </div>`
          : ""}
        <div class="message-content">
          <div class="message-header">
            <span class="message-type">${this.message?.type}</span>
            <span class="message-timestamp"
              >${this.formatTimestamp(this.message?.timestamp)}
              ${this.message?.elapsed
                ? html`(${(this.message?.elapsed / 1e9).toFixed(2)}s)`
                : ""}</span
            >
            ${this.message?.usage
              ? html` <span class="message-usage">
                  <span title="Input tokens"
                    >In:
                    ${this.formatNumber(
                      (this.message?.usage?.input_tokens || 0) +
                        (this.message?.usage?.cache_read_input_tokens || 0) +
                        (this.message?.usage?.cache_creation_input_tokens || 0),
                    )}</span
                  >
                  <span title="Output tokens"
                    >Out:
                    ${this.formatNumber(
                      this.message?.usage?.output_tokens,
                    )}</span
                  >
                  <span title="Message cost"
                    >(${this.formatCurrency(
                      this.message?.usage?.cost_usd,
                    )})</span
                  >
                </span>`
              : ""}
          </div>
          <div class="message-text-container">
            <div class="message-actions">
              ${copyButton(this.message?.content)}
            </div>
            ${this.message?.content
              ? html`
                  <div class="message-text markdown-content">
                    ${unsafeHTML(this.renderMarkdown(this.message?.content))}
                  </div>
                `
              : ""}
          </div>
          <sketch-tool-calls
            .toolCalls=${this.message?.tool_calls}
            .open=${this.open}
          ></sketch-tool-calls>
          ${this.message?.commits
            ? html`
                <div class="commits-container">
                  <div class="commits-header">
                    ${this.message.commits.length} new
                    commit${this.message.commits.length > 1 ? "s" : ""} detected
                  </div>
                  ${this.message.commits.map((commit) => {
                    return html`
                      <div class="commit-boxes-row">
                        <div class="commit-box">
                          <div class="commit-preview">
                            <span
                              class="commit-hash"
                              title="Click to copy: ${commit.hash}"
                              @click=${(e) =>
                                this.copyToClipboard(
                                  commit.hash.substring(0, 8),
                                  e,
                                )}
                            >
                              ${commit.hash.substring(0, 8)}
                            </span>
                            ${commit.pushed_branch
                              ? html`
                                  <span class="branch-wrapper">
                                    (<span
                                      class="commit-branch pushed-branch"
                                      title="Click to copy: ${commit.pushed_branch}"
                                      @click=${(e) =>
                                        this.copyToClipboard(
                                          commit.pushed_branch,
                                          e,
                                        )}
                                      >${commit.pushed_branch}</span
                                    >)
                                  </span>
                                `
                              : ``}
                            <span class="commit-subject"
                              >${commit.subject}</span
                            >
                            <button
                              class="commit-diff-button"
                              @click=${() => this.showCommit(commit.hash)}
                            >
                              View Diff
                            </button>
                          </div>
                          <div class="commit-details is-hidden">
                            <pre>${commit.body}</pre>
                          </div>
                        </div>
                      </div>
                    `;
                  })}
                </div>
              `
            : ""}
        </div>
      </div>
    `;
  }
}

function copyButton(textToCopy: string) {
  // Add click event listener to handle copying
  const buttonClass = "copy-button";
  const buttonContent = "Copy";
  const successContent = "Copied!";
  const failureContent = "Failed";

  const ret = html`<button
    class="${buttonClass}"
    title="Copy to clipboard"
    @click=${(e: Event) => {
      e.stopPropagation();
      const copyButton = e.currentTarget as HTMLButtonElement;
      navigator.clipboard
        .writeText(textToCopy)
        .then(() => {
          copyButton.textContent = successContent;
          setTimeout(() => {
            copyButton.textContent = buttonContent;
          }, 2000);
        })
        .catch((err) => {
          console.error("Failed to copy text: ", err);
          copyButton.textContent = failureContent;
          setTimeout(() => {
            copyButton.textContent = buttonContent;
          }, 2000);
        });
    }}
  >
    ${buttonContent}
  </button>`;

  return ret;
}

// Create global styles for floating messages
const floatingMessageStyles = document.createElement("style");
floatingMessageStyles.textContent = `
  .floating-message {
    background-color: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-family: system-ui, sans-serif;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    pointer-events: none;
    transition: opacity 0.3s ease, transform 0.3s ease;
  }
  
  .floating-message.success {
    background-color: rgba(40, 167, 69, 0.9);
  }
  
  .floating-message.error {
    background-color: rgba(220, 53, 69, 0.9);
  }
`;
document.head.appendChild(floatingMessageStyles);

declare global {
  interface HTMLElementTagNameMap {
    "sketch-timeline-message": SketchTimelineMessage;
  }
}

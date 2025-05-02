import { css, html, LitElement } from "lit";
import { customElement, property } from "lit/decorators.js";
import { unsafeHTML } from "lit/directives/unsafe-html.js";

@customElement("sketch-call-status")
export class SketchCallStatus extends LitElement {
  @property()
  llmCalls: number = 0;

  @property()
  toolCalls: string[] = [];

  @property()
  agentState: string | null = null;

  static styles = css`
    @keyframes gentle-pulse {
      0% {
        transform: scale(1);
        opacity: 1;
      }
      50% {
        transform: scale(1.15);
        opacity: 0.8;
      }
      100% {
        transform: scale(1);
        opacity: 1;
      }
    }

    .call-status-container {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 10px;
    }

    .indicator {
      display: flex;
      justify-content: center;
      align-items: center;
      width: 32px;
      height: 32px;
      border-radius: 4px;
      transition: all 0.2s ease;
      position: relative;
    }

    /* LLM indicator (lightbulb) */
    .llm-indicator {
      background-color: transparent;
      color: #9ca3af; /* Gray when inactive */
    }

    .llm-indicator.active {
      background-color: #fef3c7; /* Light yellow */
      color: #f59e0b; /* Yellow/amber when active */
      animation: gentle-pulse 1.5s infinite ease-in-out;
    }

    /* Tool indicator (wrench) */
    .tool-indicator {
      background-color: transparent;
      color: #9ca3af; /* Gray when inactive */
    }

    .tool-indicator.active {
      background-color: #dbeafe; /* Light blue */
      color: #3b82f6; /* Blue when active */
      animation: gentle-pulse 1.5s infinite ease-in-out;
    }

    svg {
      width: 20px;
      height: 20px;
    }
  `;

  render() {
    const lightbulbSVG = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"></path>
      <path d="M9 18h6"></path>
      <path d="M10 22h4"></path>
    </svg>`;

    const wrenchSVG = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
    </svg>`;

    const agentState = `${this.agentState ? " (" + this.agentState + ")" : ""}`;

    return html`
      <div class="call-status-container">
        <div
          class="indicator llm-indicator ${this.llmCalls > 0 ? "active" : ""}"
          title="${this.llmCalls > 0
            ? `${this.llmCalls} LLM ${this.llmCalls === 1 ? "call" : "calls"} in progress`
            : "No LLM calls in progress"}${agentState}"
        >
          ${unsafeHTML(lightbulbSVG)}
        </div>
        <div
          class="indicator tool-indicator ${this.toolCalls.length > 0
            ? "active"
            : ""}"
          title="${this.toolCalls.length > 0
            ? `${this.toolCalls.length} tool ${this.toolCalls.length === 1 ? "call" : "calls"} in progress: ${this.toolCalls.join(", ")}`
            : "No tool calls in progress"}${agentState}"
        >
          ${unsafeHTML(wrenchSVG)}
        </div>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "sketch-call-status": SketchCallStatus;
  }
}

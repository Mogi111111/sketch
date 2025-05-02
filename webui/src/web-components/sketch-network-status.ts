import { css, html, LitElement } from "lit";
import { customElement, property } from "lit/decorators.js";

@customElement("sketch-network-status")
export class SketchNetworkStatus extends LitElement {
  @property()
  connection: string;

  @property()
  error: string;

  // See https://lit.dev/docs/components/styles/ for how lit-element handles CSS.
  // Note that these styles only apply to the scope of this web component's
  // shadow DOM node, so they won't leak out or collide with CSS declared in
  // other components or the containing web page (...unless you want it to do that).

  static styles = css`
    .status-container {
      display: flex;
      align-items: center;
    }

    .status-text {
      font-size: 11px;
      color: #666;
    }
  `;

  constructor() {
    super();
  }

  // See https://lit.dev/docs/components/lifecycle/
  connectedCallback() {
    super.connectedCallback();
  }

  // See https://lit.dev/docs/components/lifecycle/
  disconnectedCallback() {
    super.disconnectedCallback();
  }

  render() {
    // Only render if there's an error to display
    if (!this.error) {
      return html``;
    }

    return html`
      <div class="status-container">
        <span id="statusText" class="status-text">${this.error}</span>
      </div>
    `;
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "sketch-network-status": SketchNetworkStatus;
  }
}

import { html, css, LitElement } from "lit";
import { customElement } from "lit/decorators.js";
import "../../sketch-timeline-message.ts";
// Using simple objects matching the AgentMessage interface

@customElement("mermaid-test-component")
export class MermaidTestComponent extends LitElement {
  static styles = css`
    :host {
      display: block;
    }
    .test-section {
      margin-bottom: 30px;
    }
    h2 {
      margin-top: 0;
      color: #444;
    }
  `;

  render() {
    // Create a sample message with Mermaid diagrams
    const flowchartMessage = {
      id: "test-1",
      type: "agent",
      content: `# Mermaid Flowchart Example

This is a test of a flowchart diagram in Mermaid syntax:

\`\`\`mermaid
graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great!]
    B -->|No| D[Try again]
    C --> E[Continue]
    D --> B
\`\`\`

The above should render as a Mermaid diagram.`,
      timestamp: new Date().toISOString(),
    };

    const sequenceDiagramMessage = {
      id: "test-2",
      type: "agent",
      content: `# Mermaid Sequence Diagram Example

Here's a sequence diagram showing a typical HTTP request:

\`\`\`mermaid
sequenceDiagram
    participant Browser
    participant Server
    Browser->>Server: GET /index.html
    Server-->>Browser: HTML Response
    Browser->>Server: GET /style.css
    Server-->>Browser: CSS Response
    Browser->>Server: GET /script.js
    Server-->>Browser: JS Response
\`\`\`

Complex diagrams should render properly.`,
      timestamp: new Date().toISOString(),
    };

    const classDiagramMessage = {
      id: "test-3",
      type: "agent",
      content: `# Mermaid Class Diagram Example

A simple class diagram in Mermaid:

\`\`\`mermaid
classDiagram
    class Animal {
        +string name
        +makeSound() void
    }
    class Dog {
        +bark() void
    }
    class Cat {
        +meow() void
    }
    Animal <|-- Dog
    Animal <|-- Cat
\`\`\`

This represents a basic inheritance diagram.`,
      timestamp: new Date().toISOString(),
    };

    const normalMarkdownMessage = {
      id: "test-4",
      type: "agent",
      content: `# Regular Markdown

This is regular markdown with:

- A bullet list
- **Bold text**
- *Italic text*

\`\`\`javascript
// Regular code block
const x = 10;
console.log('This is not Mermaid');
\`\`\`

Regular markdown should continue to work properly.`,
      timestamp: new Date().toISOString(),
    };

    return html`
      <div class="test-section">
        <h2>Flowchart Diagram Test</h2>
        <sketch-timeline-message
          .message=${flowchartMessage}
        ></sketch-timeline-message>
      </div>

      <div class="test-section">
        <h2>Sequence Diagram Test</h2>
        <sketch-timeline-message
          .message=${sequenceDiagramMessage}
        ></sketch-timeline-message>
      </div>

      <div class="test-section">
        <h2>Class Diagram Test</h2>
        <sketch-timeline-message
          .message=${classDiagramMessage}
        ></sketch-timeline-message>
      </div>

      <div class="test-section">
        <h2>Normal Markdown Test</h2>
        <sketch-timeline-message
          .message=${normalMarkdownMessage}
        ></sketch-timeline-message>
      </div>
    `;
  }
}

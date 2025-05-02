package loop

import (
	"cmp"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"runtime/debug"
	"slices"
	"strings"
	"sync"
	"time"

	"sketch.dev/ant"
	"sketch.dev/browser"
	"sketch.dev/claudetool"
	"sketch.dev/claudetool/bashkit"
)

const (
	userCancelMessage = "user requested agent to stop handling responses"
)

type MessageIterator interface {
	// Next blocks until the next message is available. It may
	// return nil if the underlying iterator context is done.
	Next() *AgentMessage
	Close()
}

type CodingAgent interface {
	// Init initializes an agent inside a docker container.
	Init(AgentInit) error

	// Ready returns a channel closed after Init successfully called.
	Ready() <-chan struct{}

	// URL reports the HTTP URL of this agent.
	URL() string

	// UserMessage enqueues a message to the agent and returns immediately.
	UserMessage(ctx context.Context, msg string)

	// Returns an iterator that finishes when the context is done and
	// starts with the given message index.
	NewIterator(ctx context.Context, nextMessageIdx int) MessageIterator

	// Loop begins the agent loop returns only when ctx is cancelled.
	Loop(ctx context.Context)

	CancelTurn(cause error)

	CancelToolUse(toolUseID string, cause error) error

	// Returns a subset of the agent's message history.
	Messages(start int, end int) []AgentMessage

	// Returns the current number of messages in the history
	MessageCount() int

	TotalUsage() ant.CumulativeUsage
	OriginalBudget() ant.Budget

	WorkingDir() string

	// Diff returns a unified diff of changes made since the agent was instantiated.
	// If commit is non-nil, it shows the diff for just that specific commit.
	Diff(commit *string) (string, error)

	// InitialCommit returns the Git commit hash that was saved when the agent was instantiated.
	InitialCommit() string

	// Title returns the current title of the conversation.
	Title() string

	// BranchName returns the git branch name for the conversation.
	BranchName() string

	// OS returns the operating system of the client.
	OS() string

	// SessionID returns the unique session identifier.
	SessionID() string

	// OutstandingLLMCallCount returns the number of outstanding LLM calls.
	OutstandingLLMCallCount() int

	// OutstandingToolCalls returns the names of outstanding tool calls.
	OutstandingToolCalls() []string
	OutsideOS() string
	OutsideHostname() string
	OutsideWorkingDir() string
	GitOrigin() string
	// OpenBrowser is a best-effort attempt to open a browser at url in outside sketch.
	OpenBrowser(url string)

	// RestartConversation resets the conversation history
	RestartConversation(ctx context.Context, rev string, initialPrompt string) error
	// SuggestReprompt suggests a re-prompt based on the current conversation.
	SuggestReprompt(ctx context.Context) (string, error)
	// IsInContainer returns true if the agent is running in a container
	IsInContainer() bool
	// FirstMessageIndex returns the index of the first message in the current conversation
	FirstMessageIndex() int

	CurrentStateName() string
}

type CodingAgentMessageType string

const (
	UserMessageType    CodingAgentMessageType = "user"
	AgentMessageType   CodingAgentMessageType = "agent"
	ErrorMessageType   CodingAgentMessageType = "error"
	BudgetMessageType  CodingAgentMessageType = "budget" // dedicated for "out of budget" errors
	ToolUseMessageType CodingAgentMessageType = "tool"
	CommitMessageType  CodingAgentMessageType = "commit" // for displaying git commits
	AutoMessageType    CodingAgentMessageType = "auto"   // for automated notifications like autoformatting

	cancelToolUseMessage = "Stop responding to my previous message. Wait for me to ask you something else before attempting to use any more tools."
)

type AgentMessage struct {
	Type CodingAgentMessageType `json:"type"`
	// EndOfTurn indicates that the AI is done working and is ready for the next user input.
	EndOfTurn bool `json:"end_of_turn"`

	Content    string `json:"content"`
	ToolName   string `json:"tool_name,omitempty"`
	ToolInput  string `json:"input,omitempty"`
	ToolResult string `json:"tool_result,omitempty"`
	ToolError  bool   `json:"tool_error,omitempty"`
	ToolCallId string `json:"tool_call_id,omitempty"`

	// ToolCalls is a list of all tool calls requested in this message (name and input pairs)
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// ToolResponses is a list of all responses to tool calls requested in this message (name and input pairs)
	ToolResponses []AgentMessage `json:"toolResponses,omitempty"`

	// Commits is a list of git commits for a commit message
	Commits []*GitCommit `json:"commits,omitempty"`

	Timestamp            time.Time  `json:"timestamp"`
	ConversationID       string     `json:"conversation_id"`
	ParentConversationID *string    `json:"parent_conversation_id,omitempty"`
	Usage                *ant.Usage `json:"usage,omitempty"`

	// Message timing information
	StartTime *time.Time     `json:"start_time,omitempty"`
	EndTime   *time.Time     `json:"end_time,omitempty"`
	Elapsed   *time.Duration `json:"elapsed,omitempty"`

	// Turn duration - the time taken for a complete agent turn
	TurnDuration *time.Duration `json:"turnDuration,omitempty"`

	Idx int `json:"idx"`
}

// SetConvo sets m.ConversationID and m.ParentConversationID based on convo.
func (m *AgentMessage) SetConvo(convo *ant.Convo) {
	if convo == nil {
		m.ConversationID = ""
		m.ParentConversationID = nil
		return
	}
	m.ConversationID = convo.ID
	if convo.Parent != nil {
		m.ParentConversationID = &convo.Parent.ID
	}
}

// GitCommit represents a single git commit for a commit message
type GitCommit struct {
	Hash         string `json:"hash"`                    // Full commit hash
	Subject      string `json:"subject"`                 // Commit subject line
	Body         string `json:"body"`                    // Full commit message body
	PushedBranch string `json:"pushed_branch,omitempty"` // If set, this commit was pushed to this branch
}

// ToolCall represents a single tool call within an agent message
type ToolCall struct {
	Name          string        `json:"name"`
	Input         string        `json:"input"`
	ToolCallId    string        `json:"tool_call_id"`
	ResultMessage *AgentMessage `json:"result_message,omitempty"`
	Args          string        `json:"args,omitempty"`
	Result        string        `json:"result,omitempty"`
}

func (a *AgentMessage) Attr() slog.Attr {
	var attrs []any = []any{
		slog.String("type", string(a.Type)),
	}
	attrs = append(attrs, slog.Int("idx", a.Idx))
	if a.EndOfTurn {
		attrs = append(attrs, slog.Bool("end_of_turn", a.EndOfTurn))
	}
	if a.Content != "" {
		attrs = append(attrs, slog.String("content", a.Content))
	}
	if a.ToolName != "" {
		attrs = append(attrs, slog.String("tool_name", a.ToolName))
	}
	if a.ToolInput != "" {
		attrs = append(attrs, slog.String("tool_input", a.ToolInput))
	}
	if a.Elapsed != nil {
		attrs = append(attrs, slog.Int64("elapsed", a.Elapsed.Nanoseconds()))
	}
	if a.TurnDuration != nil {
		attrs = append(attrs, slog.Int64("turnDuration", a.TurnDuration.Nanoseconds()))
	}
	if a.ToolResult != "" {
		attrs = append(attrs, slog.String("tool_result", a.ToolResult))
	}
	if a.ToolError {
		attrs = append(attrs, slog.Bool("tool_error", a.ToolError))
	}
	if len(a.ToolCalls) > 0 {
		toolCallAttrs := make([]any, 0, len(a.ToolCalls))
		for i, tc := range a.ToolCalls {
			toolCallAttrs = append(toolCallAttrs, slog.Group(
				fmt.Sprintf("tool_call_%d", i),
				slog.String("name", tc.Name),
				slog.String("input", tc.Input),
			))
		}
		attrs = append(attrs, slog.Group("tool_calls", toolCallAttrs...))
	}
	if a.ConversationID != "" {
		attrs = append(attrs, slog.String("convo_id", a.ConversationID))
	}
	if a.ParentConversationID != nil {
		attrs = append(attrs, slog.String("parent_convo_id", *a.ParentConversationID))
	}
	if a.Usage != nil && !a.Usage.IsZero() {
		attrs = append(attrs, a.Usage.Attr())
	}
	// TODO: timestamp, convo ids, idx?
	return slog.Group("agent_message", attrs...)
}

func errorMessage(err error) AgentMessage {
	// It's somewhat unknowable whether error messages are "end of turn" or not, but it seems like the best approach.
	if os.Getenv(("DEBUG")) == "1" {
		return AgentMessage{Type: ErrorMessageType, Content: err.Error() + " Stacktrace: " + string(debug.Stack()), EndOfTurn: true}
	}

	return AgentMessage{Type: ErrorMessageType, Content: err.Error(), EndOfTurn: true}
}

func budgetMessage(err error) AgentMessage {
	return AgentMessage{Type: BudgetMessageType, Content: err.Error(), EndOfTurn: true}
}

// ConvoInterface defines the interface for conversation interactions
type ConvoInterface interface {
	CumulativeUsage() ant.CumulativeUsage
	ResetBudget(ant.Budget)
	OverBudget() error
	SendMessage(message ant.Message) (*ant.MessageResponse, error)
	SendUserTextMessage(s string, otherContents ...ant.Content) (*ant.MessageResponse, error)
	GetID() string
	ToolResultContents(ctx context.Context, resp *ant.MessageResponse) ([]ant.Content, error)
	ToolResultCancelContents(resp *ant.MessageResponse) ([]ant.Content, error)
	CancelToolUse(toolUseID string, cause error) error
	SubConvoWithHistory() *ant.Convo
}

type Agent struct {
	convo             ConvoInterface
	config            AgentConfig // config for this agent
	workingDir        string
	repoRoot          string // workingDir may be a subdir of repoRoot
	url               string
	firstMessageIndex int           // index of the first message in the current conversation
	lastHEAD          string        // hash of the last HEAD that was pushed to the host (only when under docker)
	initialCommit     string        // hash of the Git HEAD when the agent was instantiated or Init()
	gitRemoteAddr     string        // HTTP URL of the host git repo (only when under docker)
	outsideHTTP       string        // base address of the outside webserver (only when under docker)
	ready             chan struct{} // closed when the agent is initialized (only when under docker)
	startedAt         time.Time
	originalBudget    ant.Budget
	title             string
	branchName        string
	codereview        *claudetool.CodeReviewer
	// State machine to track agent state
	stateMachine *StateMachine
	// Outside information
	outsideHostname   string
	outsideOS         string
	outsideWorkingDir string
	// URL of the git remote 'origin' if it exists
	gitOrigin string

	// Time when the current turn started (reset at the beginning of InnerLoop)
	startOfTurn time.Time

	// Inbox - for messages from the user to the agent.
	// sent on by UserMessage
	// . e.g. when user types into the chat textarea
	// read from by GatherMessages
	inbox chan string

	// protects cancelTurn
	cancelTurnMu sync.Mutex
	// cancels potentially long-running tool_use calls or chains of them
	cancelTurn context.CancelCauseFunc

	// protects following
	mu sync.Mutex

	// Stores all messages for this agent
	history []AgentMessage

	// Iterators add themselves here when they're ready to be notified of new messages.
	subscribers []chan *AgentMessage

	// Track git commits we've already seen (by hash)
	seenCommits map[string]bool

	// Track outstanding LLM call IDs
	outstandingLLMCalls map[string]struct{}

	// Track outstanding tool calls by ID with their names
	outstandingToolCalls map[string]string
}

// NewIterator implements CodingAgent.
func (a *Agent) NewIterator(ctx context.Context, nextMessageIdx int) MessageIterator {
	a.mu.Lock()
	defer a.mu.Unlock()

	return &MessageIteratorImpl{
		agent:          a,
		ctx:            ctx,
		nextMessageIdx: nextMessageIdx,
		ch:             make(chan *AgentMessage, 100),
	}
}

type MessageIteratorImpl struct {
	agent          *Agent
	ctx            context.Context
	nextMessageIdx int
	ch             chan *AgentMessage
	subscribed     bool
}

func (m *MessageIteratorImpl) Close() {
	m.agent.mu.Lock()
	defer m.agent.mu.Unlock()
	// Delete ourselves from the subscribers list
	m.agent.subscribers = slices.DeleteFunc(m.agent.subscribers, func(x chan *AgentMessage) bool {
		return x == m.ch
	})
	close(m.ch)
}

func (m *MessageIteratorImpl) Next() *AgentMessage {
	// We avoid subscription at creation to let ourselves catch up to "current state"
	// before subscribing.
	if !m.subscribed {
		m.agent.mu.Lock()
		if m.nextMessageIdx < len(m.agent.history) {
			msg := &m.agent.history[m.nextMessageIdx]
			m.nextMessageIdx++
			m.agent.mu.Unlock()
			return msg
		}
		// The next message doesn't exist yet, so let's subscribe
		m.agent.subscribers = append(m.agent.subscribers, m.ch)
		m.subscribed = true
		m.agent.mu.Unlock()
	}

	for {
		select {
		case <-m.ctx.Done():
			m.agent.mu.Lock()
			// Delete ourselves from the subscribers list
			m.agent.subscribers = slices.DeleteFunc(m.agent.subscribers, func(x chan *AgentMessage) bool {
				return x == m.ch
			})
			m.subscribed = false
			m.agent.mu.Unlock()
			return nil
		case msg, ok := <-m.ch:
			if !ok {
				// Close may have been called
				return nil
			}
			if msg.Idx == m.nextMessageIdx {
				m.nextMessageIdx++
				return msg
			}
			slog.Debug("Out of order messages", "expected", m.nextMessageIdx, "got", msg.Idx, "m", msg.Content)
			panic("out of order message")
		}
	}
}

// Assert that Agent satisfies the CodingAgent interface.
var _ CodingAgent = &Agent{}

// StateName implements CodingAgent.
func (a *Agent) CurrentStateName() string {
	if a.stateMachine == nil {
		return ""
	}
	return a.stateMachine.currentState.String()
}

func (a *Agent) URL() string { return a.url }

// Title returns the current title of the conversation.
// If no title has been set, returns an empty string.
func (a *Agent) Title() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.title
}

// BranchName returns the git branch name for the conversation.
func (a *Agent) BranchName() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.branchName
}

// OutstandingLLMCallCount returns the number of outstanding LLM calls.
func (a *Agent) OutstandingLLMCallCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.outstandingLLMCalls)
}

// OutstandingToolCalls returns the names of outstanding tool calls.
func (a *Agent) OutstandingToolCalls() []string {
	a.mu.Lock()
	defer a.mu.Unlock()

	tools := make([]string, 0, len(a.outstandingToolCalls))
	for _, toolName := range a.outstandingToolCalls {
		tools = append(tools, toolName)
	}
	return tools
}

// OS returns the operating system of the client.
func (a *Agent) OS() string {
	return a.config.ClientGOOS
}

func (a *Agent) SessionID() string {
	return a.config.SessionID
}

// OutsideOS returns the operating system of the outside system.
func (a *Agent) OutsideOS() string {
	return a.outsideOS
}

// OutsideHostname returns the hostname of the outside system.
func (a *Agent) OutsideHostname() string {
	return a.outsideHostname
}

// OutsideWorkingDir returns the working directory on the outside system.
func (a *Agent) OutsideWorkingDir() string {
	return a.outsideWorkingDir
}

// GitOrigin returns the URL of the git remote 'origin' if it exists.
func (a *Agent) GitOrigin() string {
	return a.gitOrigin
}

func (a *Agent) OpenBrowser(url string) {
	if !a.IsInContainer() {
		browser.Open(url)
		return
	}
	// We're in Docker, need to send a request to the Git server
	// to signal that the outer process should open the browser.
	httpc := &http.Client{Timeout: 5 * time.Second}
	resp, err := httpc.Post(a.outsideHTTP+"/browser", "text/plain", strings.NewReader(url))
	if err != nil {
		slog.Debug("browser launch request connection failed", "err", err, "url", url)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		return
	}
	body, _ := io.ReadAll(resp.Body)
	slog.Debug("browser launch request execution failed", "status", resp.Status, "body", string(body))
}

// CurrentState returns the current state of the agent's state machine.
func (a *Agent) CurrentState() State {
	return a.stateMachine.CurrentState()
}

func (a *Agent) IsInContainer() bool {
	return a.config.InDocker
}

func (a *Agent) FirstMessageIndex() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.firstMessageIndex
}

// SetTitleBranch sets the title and branch name of the conversation.
func (a *Agent) SetTitleBranch(title, branchName string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.title = title
	a.branchName = branchName

	// TODO: We could potentially notify listeners of a state change, but,
	// realistically, a new message will be sent for the tool result as well.
}

// OnToolCall implements ant.Listener and tracks the start of a tool call.
func (a *Agent) OnToolCall(ctx context.Context, convo *ant.Convo, id string, toolName string, toolInput json.RawMessage, content ant.Content) {
	// Track the tool call
	a.mu.Lock()
	a.outstandingToolCalls[id] = toolName
	a.mu.Unlock()
}

// OnToolResult implements ant.Listener.
func (a *Agent) OnToolResult(ctx context.Context, convo *ant.Convo, toolID string, toolName string, toolInput json.RawMessage, content ant.Content, result *string, err error) {
	// Remove the tool call from outstanding calls
	a.mu.Lock()
	delete(a.outstandingToolCalls, toolID)
	a.mu.Unlock()

	m := AgentMessage{
		Type:       ToolUseMessageType,
		Content:    content.Text,
		ToolResult: content.ToolResult,
		ToolError:  content.ToolError,
		ToolName:   toolName,
		ToolInput:  string(toolInput),
		ToolCallId: content.ToolUseID,
		StartTime:  content.StartTime,
		EndTime:    content.EndTime,
	}

	// Calculate the elapsed time if both start and end times are set
	if content.StartTime != nil && content.EndTime != nil {
		elapsed := content.EndTime.Sub(*content.StartTime)
		m.Elapsed = &elapsed
	}

	m.SetConvo(convo)
	a.pushToOutbox(ctx, m)
}

// OnRequest implements ant.Listener.
func (a *Agent) OnRequest(ctx context.Context, convo *ant.Convo, id string, msg *ant.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.outstandingLLMCalls[id] = struct{}{}
	// We already get tool results from the above. We send user messages to the outbox in the agent loop.
}

// OnResponse implements ant.Listener. Responses contain messages from the LLM
// that need to be displayed (as well as tool calls that we send along when
// they're done). (It would be reasonable to also mention tool calls when they're
// started, but we don't do that yet.)
func (a *Agent) OnResponse(ctx context.Context, convo *ant.Convo, id string, resp *ant.MessageResponse) {
	// Remove the LLM call from outstanding calls
	a.mu.Lock()
	delete(a.outstandingLLMCalls, id)
	a.mu.Unlock()

	if resp == nil {
		// LLM API call failed
		m := AgentMessage{
			Type:    ErrorMessageType,
			Content: "API call failed, type 'continue' to try again",
		}
		m.SetConvo(convo)
		a.pushToOutbox(ctx, m)
		return
	}

	endOfTurn := false
	if resp.StopReason != ant.StopReasonToolUse {
		endOfTurn = true
	}
	m := AgentMessage{
		Type:      AgentMessageType,
		Content:   collectTextContent(resp),
		EndOfTurn: endOfTurn,
		Usage:     &resp.Usage,
		StartTime: resp.StartTime,
		EndTime:   resp.EndTime,
	}

	// Extract any tool calls from the response
	if resp.StopReason == ant.StopReasonToolUse {
		var toolCalls []ToolCall
		for _, part := range resp.Content {
			if part.Type == ant.ContentTypeToolUse {
				toolCalls = append(toolCalls, ToolCall{
					Name:       part.ToolName,
					Input:      string(part.ToolInput),
					ToolCallId: part.ID,
				})
			}
		}
		m.ToolCalls = toolCalls
	}

	// Calculate the elapsed time if both start and end times are set
	if resp.StartTime != nil && resp.EndTime != nil {
		elapsed := resp.EndTime.Sub(*resp.StartTime)
		m.Elapsed = &elapsed
	}

	m.SetConvo(convo)
	a.pushToOutbox(ctx, m)
}

// WorkingDir implements CodingAgent.
func (a *Agent) WorkingDir() string {
	return a.workingDir
}

// MessageCount implements CodingAgent.
func (a *Agent) MessageCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.history)
}

// Messages implements CodingAgent.
func (a *Agent) Messages(start int, end int) []AgentMessage {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.history[start:end])
}

func (a *Agent) OriginalBudget() ant.Budget {
	return a.originalBudget
}

// AgentConfig contains configuration for creating a new Agent.
type AgentConfig struct {
	Context          context.Context
	AntURL           string
	APIKey           string
	HTTPC            *http.Client
	Budget           ant.Budget
	GitUsername      string
	GitEmail         string
	SessionID        string
	ClientGOOS       string
	ClientGOARCH     string
	InDocker         bool
	UseAnthropicEdit bool
	// Outside information
	OutsideHostname   string
	OutsideOS         string
	OutsideWorkingDir string
}

// NewAgent creates a new Agent.
// It is not usable until Init() is called.
func NewAgent(config AgentConfig) *Agent {
	agent := &Agent{
		config:               config,
		ready:                make(chan struct{}),
		inbox:                make(chan string, 100),
		subscribers:          make([]chan *AgentMessage, 0),
		startedAt:            time.Now(),
		originalBudget:       config.Budget,
		seenCommits:          make(map[string]bool),
		outsideHostname:      config.OutsideHostname,
		outsideOS:            config.OutsideOS,
		outsideWorkingDir:    config.OutsideWorkingDir,
		outstandingLLMCalls:  make(map[string]struct{}),
		outstandingToolCalls: make(map[string]string),
		stateMachine:         NewStateMachine(),
	}
	return agent
}

type AgentInit struct {
	WorkingDir string
	NoGit      bool // only for testing

	InDocker      bool
	Commit        string
	OutsideHTTP   string
	GitRemoteAddr string
	HostAddr      string
}

func (a *Agent) Init(ini AgentInit) error {
	if a.convo != nil {
		return fmt.Errorf("Agent.Init: already initialized")
	}
	ctx := a.config.Context
	if ini.InDocker {
		cmd := exec.CommandContext(ctx, "git", "stash")
		cmd.Dir = ini.WorkingDir
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("git stash: %s: %v", out, err)
		}
		cmd = exec.CommandContext(ctx, "git", "remote", "add", "sketch-host", ini.GitRemoteAddr)
		cmd.Dir = ini.WorkingDir
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("git remote add: %s: %v", out, err)
		}
		cmd = exec.CommandContext(ctx, "git", "fetch", "--prune", "sketch-host")
		cmd.Dir = ini.WorkingDir
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("git fetch: %s: %w", out, err)
		}
		cmd = exec.CommandContext(ctx, "git", "checkout", "-f", ini.Commit)
		cmd.Dir = ini.WorkingDir
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("git checkout %s: %s: %w", ini.Commit, out, err)
		}
		a.lastHEAD = ini.Commit
		a.gitRemoteAddr = ini.GitRemoteAddr
		a.outsideHTTP = ini.OutsideHTTP
		a.initialCommit = ini.Commit
		if ini.HostAddr != "" {
			a.url = "http://" + ini.HostAddr
		}
	}
	a.workingDir = ini.WorkingDir

	if !ini.NoGit {
		repoRoot, err := repoRoot(ctx, a.workingDir)
		if err != nil {
			return fmt.Errorf("repoRoot: %w", err)
		}
		a.repoRoot = repoRoot

		commitHash, err := resolveRef(ctx, a.repoRoot, "HEAD")
		if err != nil {
			return fmt.Errorf("resolveRef: %w", err)
		}
		a.initialCommit = commitHash

		codereview, err := claudetool.NewCodeReviewer(ctx, a.repoRoot, a.initialCommit)
		if err != nil {
			return fmt.Errorf("Agent.Init: claudetool.NewCodeReviewer: %w", err)
		}
		a.codereview = codereview

		a.gitOrigin = getGitOrigin(ctx, ini.WorkingDir)
	}
	a.lastHEAD = a.initialCommit
	a.convo = a.initConvo()
	close(a.ready)
	return nil
}

//go:embed agent_system_prompt.txt
var agentSystemPrompt string

// initConvo initializes the conversation.
// It must not be called until all agent fields are initialized,
// particularly workingDir and git.
func (a *Agent) initConvo() *ant.Convo {
	ctx := a.config.Context
	convo := ant.NewConvo(ctx, a.config.APIKey)
	if a.config.HTTPC != nil {
		convo.HTTPC = a.config.HTTPC
	}
	if a.config.AntURL != "" {
		convo.URL = a.config.AntURL
	}
	convo.PromptCaching = true
	convo.Budget = a.config.Budget

	var editPrompt string
	if a.config.UseAnthropicEdit {
		editPrompt = "Then use the str_replace_editor tool to make those edits. For short complete file replacements, you may use the bash tool with cat and heredoc stdin."
	} else {
		editPrompt = "Then use the patch tool to make those edits. Combine all edits to any given file into a single patch tool call."
	}

	convo.SystemPrompt = fmt.Sprintf(agentSystemPrompt, editPrompt, a.config.ClientGOOS, a.config.ClientGOARCH, a.workingDir, a.repoRoot, a.initialCommit)

	// Define a permission callback for the bash tool to check if the branch name is set before allowing git commits
	bashPermissionCheck := func(command string) error {
		// Check if branch name is set
		a.mu.Lock()
		branchSet := a.branchName != ""
		a.mu.Unlock()

		// If branch is set, all commands are allowed
		if branchSet {
			return nil
		}

		// If branch is not set, check if this is a git commit command
		willCommit, err := bashkit.WillRunGitCommit(command)
		if err != nil {
			// If there's an error checking, we should allow the command to proceed
			return nil
		}

		// If it's a git commit and branch is not set, return an error
		if willCommit {
			return fmt.Errorf("you must use the title tool before making git commits")
		}

		return nil
	}

	// Create a custom bash tool with the permission check
	bashTool := claudetool.NewBashTool(bashPermissionCheck)

	// Register all tools with the conversation
	// When adding, removing, or modifying tools here, double-check that the termui tool display
	// template in termui/termui.go has pretty-printing support for all tools.
	convo.Tools = []*ant.Tool{
		bashTool, claudetool.Keyword,
		claudetool.Think, a.titleTool(), makeDoneTool(a.codereview, a.config.GitUsername, a.config.GitEmail),
		a.codereview.Tool(),
	}
	if a.config.UseAnthropicEdit {
		convo.Tools = append(convo.Tools, claudetool.AnthropicEditTool)
	} else {
		convo.Tools = append(convo.Tools, claudetool.Patch)
	}
	convo.Listener = a
	return convo
}

// branchExists reports whether branchName exists, either locally or in well-known remotes.
func branchExists(dir, branchName string) bool {
	refs := []string{
		"refs/heads/",
		"refs/remotes/origin/",
		"refs/remotes/sketch-host/",
	}
	for _, ref := range refs {
		cmd := exec.Command("git", "show-ref", "--verify", "--quiet", ref+branchName)
		cmd.Dir = dir
		if cmd.Run() == nil { // exit code 0 means branch exists
			return true
		}
	}
	return false
}

func (a *Agent) titleTool() *ant.Tool {
	title := &ant.Tool{
		Name:        "title",
		Description: `Sets the conversation title and creates a git branch for tracking work. MANDATORY: You must use this tool before making any git commits.`,
		InputSchema: json.RawMessage(`{
	"type": "object",
	"properties": {
		"title": {
			"type": "string",
			"description": "A concise title summarizing what this conversation is about, imperative tense preferred"
		},
		"branch_name": {
			"type": "string",
			"description": "A 2-3 word alphanumeric hyphenated slug for the git branch name"
		}
	},
	"required": ["title", "branch_name"]
}`),
		Run: func(ctx context.Context, input json.RawMessage) (string, error) {
			var params struct {
				Title      string `json:"title"`
				BranchName string `json:"branch_name"`
			}
			if err := json.Unmarshal(input, &params); err != nil {
				return "", err
			}
			// It's unfortunate to not allow title changes,
			// but it avoids having multiple branches.
			t := a.Title()
			if t != "" {
				return "", fmt.Errorf("title already set to: %s", t)
			}

			if params.BranchName == "" {
				return "", fmt.Errorf("branch_name parameter cannot be empty")
			}
			if params.Title == "" {
				return "", fmt.Errorf("title parameter cannot be empty")
			}
			if params.BranchName != cleanBranchName(params.BranchName) {
				return "", fmt.Errorf("branch_name parameter must be alphanumeric hyphenated slug")
			}
			branchName := "sketch/" + params.BranchName
			if branchExists(a.workingDir, branchName) {
				return "", fmt.Errorf("branch %q already exists; please choose a different branch name", branchName)
			}

			a.SetTitleBranch(params.Title, branchName)

			response := fmt.Sprintf("Title set to %q, branch name set to %q", params.Title, branchName)
			return response, nil
		},
	}
	return title
}

func (a *Agent) Ready() <-chan struct{} {
	return a.ready
}

func (a *Agent) UserMessage(ctx context.Context, msg string) {
	a.pushToOutbox(ctx, AgentMessage{Type: UserMessageType, Content: msg})
	a.inbox <- msg
}

func (a *Agent) CancelToolUse(toolUseID string, cause error) error {
	return a.convo.CancelToolUse(toolUseID, cause)
}

func (a *Agent) CancelTurn(cause error) {
	a.cancelTurnMu.Lock()
	defer a.cancelTurnMu.Unlock()
	if a.cancelTurn != nil {
		// Force state transition to cancelled state
		ctx := a.config.Context
		a.stateMachine.ForceTransition(ctx, StateCancelled, "User cancelled turn: "+cause.Error())
		a.cancelTurn(cause)
	}
}

func (a *Agent) Loop(ctxOuter context.Context) {
	for {
		select {
		case <-ctxOuter.Done():
			return
		default:
			ctxInner, cancel := context.WithCancelCause(ctxOuter)
			a.cancelTurnMu.Lock()
			// Set .cancelTurn so the user can cancel whatever is happening
			// inside the conversation loop without canceling this outer Loop execution.
			// This cancelTurn func is intended be called from other goroutines,
			// hence the mutex.
			a.cancelTurn = cancel
			a.cancelTurnMu.Unlock()
			err := a.processTurn(ctxInner) // Renamed from InnerLoop to better reflect its purpose
			if err != nil {
				slog.ErrorContext(ctxOuter, "Error in processing turn", "error", err)
			}
			cancel(nil)
		}
	}
}

func (a *Agent) pushToOutbox(ctx context.Context, m AgentMessage) {
	if m.Timestamp.IsZero() {
		m.Timestamp = time.Now()
	}

	// If this is an end-of-turn message, calculate the turn duration and add it to the message
	if m.EndOfTurn && m.Type == AgentMessageType {
		turnDuration := time.Since(a.startOfTurn)
		m.TurnDuration = &turnDuration
		slog.InfoContext(ctx, "Turn completed", "turnDuration", turnDuration)
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	m.Idx = len(a.history)
	slog.InfoContext(ctx, "agent message", m.Attr())
	a.history = append(a.history, m)

	// Notify all subscribers
	for _, ch := range a.subscribers {
		ch <- &m
	}
}

func (a *Agent) GatherMessages(ctx context.Context, block bool) ([]ant.Content, error) {
	var m []ant.Content
	if block {
		select {
		case <-ctx.Done():
			return m, ctx.Err()
		case msg := <-a.inbox:
			m = append(m, ant.StringContent(msg))
		}
	}
	for {
		select {
		case msg := <-a.inbox:
			m = append(m, ant.StringContent(msg))
		default:
			return m, nil
		}
	}
}

// processTurn handles a single conversation turn with the user
func (a *Agent) processTurn(ctx context.Context) error {
	// Reset the start of turn time
	a.startOfTurn = time.Now()

	// Transition to waiting for user input state
	a.stateMachine.Transition(ctx, StateWaitingForUserInput, "Starting turn")

	// Process initial user message
	initialResp, err := a.processUserMessage(ctx)
	if err != nil {
		a.stateMachine.Transition(ctx, StateError, "Error processing user message: "+err.Error())
		return err
	}

	// Handle edge case where both initialResp and err are nil
	if initialResp == nil {
		err := fmt.Errorf("unexpected nil response from processUserMessage with no error")
		a.stateMachine.Transition(ctx, StateError, "Error processing user message: "+err.Error())

		a.pushToOutbox(ctx, errorMessage(err))
		return err
	}

	// We do this as we go, but let's also do it at the end of the turn
	defer func() {
		if _, err := a.handleGitCommits(ctx); err != nil {
			// Just log the error, don't stop execution
			slog.WarnContext(ctx, "Failed to check for new git commits", "error", err)
		}
	}()

	// Main response loop - continue as long as the model is using tools or a tool use fails.
	resp := initialResp
	for {
		// Check if we are over budget
		if err := a.overBudget(ctx); err != nil {
			a.stateMachine.Transition(ctx, StateBudgetExceeded, "Budget exceeded: "+err.Error())
			return err
		}

		// If the model is not requesting to use a tool, we're done
		if resp.StopReason != ant.StopReasonToolUse {
			a.stateMachine.Transition(ctx, StateEndOfTurn, "LLM completed response, ending turn")
			break
		}

		// Transition to tool use requested state
		a.stateMachine.Transition(ctx, StateToolUseRequested, "LLM requested tool use")

		// Handle tool execution
		continueConversation, toolResp := a.handleToolExecution(ctx, resp)
		if !continueConversation {
			return nil
		}

		if toolResp == nil {
			return fmt.Errorf("cannot continue conversation with a nil tool response")
		}

		// Set the response for the next iteration
		resp = toolResp
	}

	return nil
}

// processUserMessage waits for user messages and sends them to the model
func (a *Agent) processUserMessage(ctx context.Context) (*ant.MessageResponse, error) {
	// Wait for at least one message from the user
	msgs, err := a.GatherMessages(ctx, true)
	if err != nil { // e.g. the context was canceled while blocking in GatherMessages
		a.stateMachine.Transition(ctx, StateError, "Error gathering messages: "+err.Error())
		return nil, err
	}

	userMessage := ant.Message{
		Role:    ant.MessageRoleUser,
		Content: msgs,
	}

	// Transition to sending to LLM state
	a.stateMachine.Transition(ctx, StateSendingToLLM, "Sending user message to LLM")

	// Send message to the model
	resp, err := a.convo.SendMessage(userMessage)
	if err != nil {
		a.stateMachine.Transition(ctx, StateError, "Error sending to LLM: "+err.Error())
		a.pushToOutbox(ctx, errorMessage(err))
		return nil, err
	}

	// Transition to processing LLM response state
	a.stateMachine.Transition(ctx, StateProcessingLLMResponse, "Processing LLM response")

	return resp, nil
}

// handleToolExecution processes a tool use request from the model
func (a *Agent) handleToolExecution(ctx context.Context, resp *ant.MessageResponse) (bool, *ant.MessageResponse) {
	var results []ant.Content
	cancelled := false

	// Transition to checking for cancellation state
	a.stateMachine.Transition(ctx, StateCheckingForCancellation, "Checking if user requested cancellation")

	// Check if the operation was cancelled by the user
	select {
	case <-ctx.Done():
		// Don't actually run any of the tools, but rather build a response
		// for each tool_use message letting the LLM know that user canceled it.
		var err error
		results, err = a.convo.ToolResultCancelContents(resp)
		if err != nil {
			a.stateMachine.Transition(ctx, StateError, "Error creating cancellation response: "+err.Error())
			a.pushToOutbox(ctx, errorMessage(err))
		}
		cancelled = true
		a.stateMachine.Transition(ctx, StateCancelled, "Operation cancelled by user")
	default:
		// Transition to running tool state
		a.stateMachine.Transition(ctx, StateRunningTool, "Executing requested tool")

		// Add working directory to context for tool execution
		ctx = claudetool.WithWorkingDir(ctx, a.workingDir)

		// Execute the tools
		var err error
		results, err = a.convo.ToolResultContents(ctx, resp)
		if ctx.Err() != nil { // e.g. the user canceled the operation
			cancelled = true
			a.stateMachine.Transition(ctx, StateCancelled, "Operation cancelled during tool execution")
		} else if err != nil {
			a.stateMachine.Transition(ctx, StateError, "Error executing tool: "+err.Error())
			a.pushToOutbox(ctx, errorMessage(err))
		}
	}

	// Process git commits that may have occurred during tool execution
	a.stateMachine.Transition(ctx, StateCheckingGitCommits, "Checking for git commits")
	autoqualityMessages := a.processGitChanges(ctx)

	// Check budget again after tool execution
	a.stateMachine.Transition(ctx, StateCheckingBudget, "Checking budget after tool execution")
	if err := a.overBudget(ctx); err != nil {
		a.stateMachine.Transition(ctx, StateBudgetExceeded, "Budget exceeded after tool execution: "+err.Error())
		return false, nil
	}

	// Continue the conversation with tool results and any user messages
	return a.continueTurnWithToolResults(ctx, results, autoqualityMessages, cancelled)
}

// processGitChanges checks for new git commits and runs autoformatters if needed
func (a *Agent) processGitChanges(ctx context.Context) []string {
	// Check for git commits after tool execution
	newCommits, err := a.handleGitCommits(ctx)
	if err != nil {
		// Just log the error, don't stop execution
		slog.WarnContext(ctx, "Failed to check for new git commits", "error", err)
		return nil
	}

	// Run autoformatters if there was exactly one new commit
	var autoqualityMessages []string
	if len(newCommits) == 1 {
		a.stateMachine.Transition(ctx, StateRunningAutoformatters, "Running autoformatters on new commit")
		formatted := a.codereview.Autoformat(ctx)
		if len(formatted) > 0 {
			msg := fmt.Sprintf(`
I ran autoformatters and they updated these files:

%s

Please amend your latest git commit with these changes and then continue with what you were doing.`,
				strings.Join(formatted, "\n"),
			)[1:]
			a.pushToOutbox(ctx, AgentMessage{
				Type:      AutoMessageType,
				Content:   msg,
				Timestamp: time.Now(),
			})
			autoqualityMessages = append(autoqualityMessages, msg)
		}
	}

	return autoqualityMessages
}

// continueTurnWithToolResults continues the conversation with tool results
func (a *Agent) continueTurnWithToolResults(ctx context.Context, results []ant.Content, autoqualityMessages []string, cancelled bool) (bool, *ant.MessageResponse) {
	// Get any messages the user sent while tools were executing
	a.stateMachine.Transition(ctx, StateGatheringAdditionalMessages, "Gathering additional user messages")
	msgs, err := a.GatherMessages(ctx, false)
	if err != nil {
		a.stateMachine.Transition(ctx, StateError, "Error gathering additional messages: "+err.Error())
		return false, nil
	}

	// Inject any auto-generated messages from quality checks
	for _, msg := range autoqualityMessages {
		msgs = append(msgs, ant.StringContent(msg))
	}

	// Handle cancellation by appending a message about it
	if cancelled {
		msgs = append(msgs, ant.StringContent(cancelToolUseMessage))
		// EndOfTurn is false here so that the client of this agent keeps processing
		// further messages; the conversation is not over.
		a.pushToOutbox(ctx, AgentMessage{Type: ErrorMessageType, Content: userCancelMessage, EndOfTurn: false})
	} else if err := a.convo.OverBudget(); err != nil {
		// Handle budget issues by appending a message about it
		budgetMsg := "We've exceeded our budget. Please ask the user to confirm before continuing by ending the turn."
		msgs = append(msgs, ant.StringContent(budgetMsg))
		a.pushToOutbox(ctx, budgetMessage(fmt.Errorf("warning: %w (ask to keep trying, if you'd like)", err)))
	}

	// Combine tool results with user messages
	results = append(results, msgs...)

	// Send the combined message to continue the conversation
	a.stateMachine.Transition(ctx, StateSendingToolResults, "Sending tool results back to LLM")
	resp, err := a.convo.SendMessage(ant.Message{
		Role:    ant.MessageRoleUser,
		Content: results,
	})
	if err != nil {
		a.stateMachine.Transition(ctx, StateError, "Error sending tool results: "+err.Error())
		a.pushToOutbox(ctx, errorMessage(fmt.Errorf("error: failed to continue conversation: %s", err.Error())))
		return true, nil // Return true to continue the conversation, but with no response
	}

	// Transition back to processing LLM response
	a.stateMachine.Transition(ctx, StateProcessingLLMResponse, "Processing LLM response to tool results")

	if cancelled {
		return false, nil
	}

	return true, resp
}

func (a *Agent) overBudget(ctx context.Context) error {
	if err := a.convo.OverBudget(); err != nil {
		a.stateMachine.Transition(ctx, StateBudgetExceeded, "Budget exceeded: "+err.Error())
		m := budgetMessage(err)
		m.Content = m.Content + "\n\nBudget reset."
		a.pushToOutbox(ctx, budgetMessage(err))
		a.convo.ResetBudget(a.originalBudget)
		return err
	}
	return nil
}

func collectTextContent(msg *ant.MessageResponse) string {
	// Collect all text content
	var allText strings.Builder
	for _, content := range msg.Content {
		if content.Type == ant.ContentTypeText && content.Text != "" {
			if allText.Len() > 0 {
				allText.WriteString("\n\n")
			}
			allText.WriteString(content.Text)
		}
	}
	return allText.String()
}

func (a *Agent) TotalUsage() ant.CumulativeUsage {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.convo.CumulativeUsage()
}

// Diff returns a unified diff of changes made since the agent was instantiated.
func (a *Agent) Diff(commit *string) (string, error) {
	if a.initialCommit == "" {
		return "", fmt.Errorf("no initial commit reference available")
	}

	// Find the repository root
	ctx := context.Background()

	// If a specific commit hash is provided, show just that commit's changes
	if commit != nil && *commit != "" {
		// Validate that the commit looks like a valid git SHA
		if !isValidGitSHA(*commit) {
			return "", fmt.Errorf("invalid git commit SHA format: %s", *commit)
		}

		// Get the diff for just this commit
		cmd := exec.CommandContext(ctx, "git", "show", "--unified=10", *commit)
		cmd.Dir = a.repoRoot
		output, err := cmd.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("failed to get diff for commit %s: %w - %s", *commit, err, string(output))
		}
		return string(output), nil
	}

	// Otherwise, get the diff between the initial commit and the current state using exec.Command
	cmd := exec.CommandContext(ctx, "git", "diff", "--unified=10", a.initialCommit)
	cmd.Dir = a.repoRoot
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to get diff: %w - %s", err, string(output))
	}

	return string(output), nil
}

// InitialCommit returns the Git commit hash that was saved when the agent was instantiated.
func (a *Agent) InitialCommit() string {
	return a.initialCommit
}

// handleGitCommits() highlights new commits to the user. When running
// under docker, new HEADs are pushed to a branch according to the title.
func (a *Agent) handleGitCommits(ctx context.Context) ([]*GitCommit, error) {
	if a.repoRoot == "" {
		return nil, nil
	}

	head, err := resolveRef(ctx, a.repoRoot, "HEAD")
	if err != nil {
		return nil, err
	}
	if head == a.lastHEAD {
		return nil, nil // nothing to do
	}
	defer func() {
		a.lastHEAD = head
	}()

	// Get new commits. Because it's possible that the agent does rebases, fixups, and
	// so forth, we use, as our fixed point, the "initialCommit", and we limit ourselves
	// to the last 100 commits.
	var commits []*GitCommit

	// Get commits since the initial commit
	// Format: <hash>\0<subject>\0<body>\0
	// This uses NULL bytes as separators to avoid issues with newlines in commit messages
	// Limit to 100 commits to avoid overwhelming the user
	cmd := exec.CommandContext(ctx, "git", "log", "-n", "100", "--pretty=format:%H%x00%s%x00%b%x00", "^"+a.initialCommit, head)
	cmd.Dir = a.repoRoot
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get git log: %w", err)
	}

	// Parse git log output and filter out already seen commits
	parsedCommits := parseGitLog(string(output))

	var headCommit *GitCommit

	// Filter out commits we've already seen
	for _, commit := range parsedCommits {
		if commit.Hash == head {
			headCommit = &commit
		}

		// Skip if we've seen this commit before. If our head has changed, always include that.
		if a.seenCommits[commit.Hash] && commit.Hash != head {
			continue
		}

		// Mark this commit as seen
		a.seenCommits[commit.Hash] = true

		// Add to our list of new commits
		commits = append(commits, &commit)
	}

	if a.gitRemoteAddr != "" {
		if headCommit == nil {
			// I think this can only happen if we have a bug or if there's a race.
			headCommit = &GitCommit{}
			headCommit.Hash = head
			headCommit.Subject = "unknown"
			commits = append(commits, headCommit)
		}

		branch := cmp.Or(a.branchName, "sketch/"+a.config.SessionID)

		// TODO: I don't love the force push here. We could see if the push is a fast-forward, and,
		// if it's not, we could make a backup with a unique name (perhaps append a timestamp) and
		// then use push with lease to replace.
		cmd = exec.Command("git", "push", "--force", a.gitRemoteAddr, "HEAD:refs/heads/"+branch)
		cmd.Dir = a.workingDir
		if out, err := cmd.CombinedOutput(); err != nil {
			a.pushToOutbox(ctx, errorMessage(fmt.Errorf("git push to host: %s: %v", out, err)))
		} else {
			headCommit.PushedBranch = branch
		}
	}

	// If we found new commits, create a message
	if len(commits) > 0 {
		msg := AgentMessage{
			Type:      CommitMessageType,
			Timestamp: time.Now(),
			Commits:   commits,
		}
		a.pushToOutbox(ctx, msg)
	}
	return commits, nil
}

func cleanBranchName(s string) string {
	return strings.Map(func(r rune) rune {
		// lowercase
		if r >= 'A' && r <= 'Z' {
			return r + 'a' - 'A'
		}
		// replace spaces with dashes
		if r == ' ' {
			return '-'
		}
		// allow alphanumerics and dashes
		if (r >= 'a' && r <= 'z') || r == '-' || (r >= '0' && r <= '9') {
			return r
		}
		return -1
	}, s)
}

// parseGitLog parses the output of git log with format '%H%x00%s%x00%b%x00'
// and returns an array of GitCommit structs.
func parseGitLog(output string) []GitCommit {
	var commits []GitCommit

	// No output means no commits
	if len(output) == 0 {
		return commits
	}

	// Split by NULL byte
	parts := strings.Split(output, "\x00")

	// Process in triplets (hash, subject, body)
	for i := 0; i < len(parts); i++ {
		// Skip empty parts
		if parts[i] == "" {
			continue
		}

		// This should be a hash
		hash := strings.TrimSpace(parts[i])

		// Make sure we have at least a subject part available
		if i+1 >= len(parts) {
			break // No more parts available
		}

		// Get the subject
		subject := strings.TrimSpace(parts[i+1])

		// Get the body if available
		body := ""
		if i+2 < len(parts) {
			body = strings.TrimSpace(parts[i+2])
		}

		// Skip to the next triplet
		i += 2

		commits = append(commits, GitCommit{
			Hash:    hash,
			Subject: subject,
			Body:    body,
		})
	}

	return commits
}

func repoRoot(ctx context.Context, dir string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--show-toplevel")
	stderr := new(strings.Builder)
	cmd.Stderr = stderr
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git rev-parse failed: %w\n%s", err, stderr)
	}
	return strings.TrimSpace(string(out)), nil
}

func resolveRef(ctx context.Context, dir, refName string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", "rev-parse", refName)
	stderr := new(strings.Builder)
	cmd.Stderr = stderr
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git rev-parse failed: %w\n%s", err, stderr)
	}
	// TODO: validate that out is valid hex
	return strings.TrimSpace(string(out)), nil
}

// isValidGitSHA validates if a string looks like a valid git SHA hash.
// Git SHAs are hexadecimal strings of at least 4 characters but typically 7, 8, or 40 characters.
func isValidGitSHA(sha string) bool {
	// Git SHA must be a hexadecimal string with at least 4 characters
	if len(sha) < 4 || len(sha) > 40 {
		return false
	}

	// Check if the string only contains hexadecimal characters
	for _, char := range sha {
		if !(char >= '0' && char <= '9') && !(char >= 'a' && char <= 'f') && !(char >= 'A' && char <= 'F') {
			return false
		}
	}

	return true
}

// getGitOrigin returns the URL of the git remote 'origin' if it exists
func getGitOrigin(ctx context.Context, dir string) string {
	cmd := exec.CommandContext(ctx, "git", "config", "--get", "remote.origin.url")
	cmd.Dir = dir
	stderr := new(strings.Builder)
	cmd.Stderr = stderr
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func (a *Agent) initGitRevision(ctx context.Context, workingDir, revision string) error {
	cmd := exec.CommandContext(ctx, "git", "stash")
	cmd.Dir = workingDir
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git stash: %s: %v", out, err)
	}
	cmd = exec.CommandContext(ctx, "git", "fetch", "--prune", "sketch-host")
	cmd.Dir = workingDir
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git fetch: %s: %w", out, err)
	}
	cmd = exec.CommandContext(ctx, "git", "checkout", "-f", revision)
	cmd.Dir = workingDir
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git checkout %s: %s: %w", revision, out, err)
	}
	a.lastHEAD = revision
	a.initialCommit = revision
	return nil
}

func (a *Agent) RestartConversation(ctx context.Context, rev string, initialPrompt string) error {
	a.mu.Lock()
	a.title = ""
	a.firstMessageIndex = len(a.history)
	a.convo = a.initConvo()
	gitReset := func() error {
		if a.config.InDocker && rev != "" {
			err := a.initGitRevision(ctx, a.workingDir, rev)
			if err != nil {
				return err
			}
		} else if !a.config.InDocker && rev != "" {
			return fmt.Errorf("Not resetting git repo when working outside of a container.")
		}
		return nil
	}
	err := gitReset()
	a.mu.Unlock()
	if err != nil {
		a.pushToOutbox(a.config.Context, errorMessage(err))
	}

	a.pushToOutbox(a.config.Context, AgentMessage{
		Type: AgentMessageType, Content: "Conversation restarted.",
	})
	if initialPrompt != "" {
		a.UserMessage(ctx, initialPrompt)
	}
	return nil
}

func (a *Agent) SuggestReprompt(ctx context.Context) (string, error) {
	msg := `The user has requested a suggestion for a re-prompt.

	Given the current conversation thus far, suggest a re-prompt that would
	capture the instructions and feedback so far, as well as any
	research or other information that would be helpful in implementing
	the task.

	Reply with ONLY the reprompt text.
	`
	userMessage := ant.UserStringMessage(msg)
	// By doing this in a subconversation, the agent doesn't call tools (because
	// there aren't any), and there's not a concurrency risk with on-going other
	// outstanding conversations.
	convo := a.convo.SubConvoWithHistory()
	resp, err := convo.SendMessage(userMessage)
	if err != nil {
		a.pushToOutbox(ctx, errorMessage(err))
		return "", err
	}
	textContent := collectTextContent(resp)
	return textContent, nil
}

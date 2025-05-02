package dockerimg

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/kevinburke/ssh_config"
	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

const keyBitSize = 2048

// SSHTheater does the necessary key pair generation, known_hosts updates, ssh_config file updates etc steps
// so that ssh can connect to a locally running sketch container to other local processes like vscode without
// the user having to run the usual ssh obstacle course.
//
// SSHTheater does not modify your default .ssh/config, or known_hosts files.  However, in order for you
// to be able to use it properly you will have to make a one-time edit to your ~/.ssh/config file.
//
// In your ~/.ssh/config file, add the following line:
//
// Include $HOME/.config/sketch/ssh_config
//
// where $HOME is your home directory.
type SSHTheater struct {
	cntrName string
	sshHost  string
	sshPort  string

	knownHostsPath     string
	userIdentityPath   string
	sshConfigPath      string
	serverIdentityPath string

	serverPublicKey ssh.PublicKey
	serverIdentity  []byte
	userIdentity    []byte

	fs FileSystem
	kg KeyGenerator
}

// NewSSHTheather will set up everything so that you can use ssh on localhost to connect to
// the sketch container.  Call #Clean when you are done with the container to remove the
// various entries it created in its known_hosts and ssh_config files. Also note that
// this will generate key pairs for both the ssh server identity and the user identity, if
// these files do not already exist.  These key pair files are not deleted by #Cleanup,
// so they can be re-used across invocations of sketch. This means every sketch container
// that runs on this host will use the same ssh server identity.
//
// If this doesn't return an error, you should be able to run "ssh <cntrName>"
// in a terminal on your host machine to open a shell into the container without having
// to manually accept changes to your known_hosts file etc.
func NewSSHTheather(cntrName, sshHost, sshPort string) (*SSHTheater, error) {
	return newSSHTheatherWithDeps(cntrName, sshHost, sshPort, &RealFileSystem{}, &RealKeyGenerator{})
}

// newSSHTheatherWithDeps creates a new SSHTheater with the specified dependencies
func newSSHTheatherWithDeps(cntrName, sshHost, sshPort string, fs FileSystem, kg KeyGenerator) (*SSHTheater, error) {
	base := filepath.Join(os.Getenv("HOME"), ".config", "sketch")
	if _, err := fs.Stat(base); err != nil {
		if err := fs.MkdirAll(base, 0o777); err != nil {
			return nil, fmt.Errorf("couldn't create %s: %w", base, err)
		}
	}

	cst := &SSHTheater{
		cntrName:           cntrName,
		sshHost:            sshHost,
		sshPort:            sshPort,
		knownHostsPath:     filepath.Join(base, "known_hosts"),
		userIdentityPath:   filepath.Join(base, "container_user_identity"),
		serverIdentityPath: filepath.Join(base, "container_server_identity"),
		sshConfigPath:      filepath.Join(base, "ssh_config"),
		fs:                 fs,
		kg:                 kg,
	}
	if _, err := cst.createKeyPairIfMissing(cst.serverIdentityPath); err != nil {
		return nil, fmt.Errorf("couldn't create server identity: %w", err)
	}
	if _, err := cst.createKeyPairIfMissing(cst.userIdentityPath); err != nil {
		return nil, fmt.Errorf("couldn't create user identity: %w", err)
	}

	serverIdentity, err := fs.ReadFile(cst.serverIdentityPath)
	if err != nil {
		return nil, fmt.Errorf("couldn't read container's ssh server identity: %w", err)
	}
	cst.serverIdentity = serverIdentity

	serverPubKeyBytes, err := fs.ReadFile(cst.serverIdentityPath + ".pub")
	if err != nil {
		return nil, fmt.Errorf("couldn't read ssh server public key file: %w", err)
	}
	serverPubKey, _, _, _, err := ssh.ParseAuthorizedKey(serverPubKeyBytes)
	if err != nil {
		return nil, fmt.Errorf("couldn't parse ssh server public key: %w", err)
	}
	cst.serverPublicKey = serverPubKey

	userIdentity, err := fs.ReadFile(cst.userIdentityPath + ".pub")
	if err != nil {
		return nil, fmt.Errorf("couldn't read ssh user identity: %w", err)
	}
	cst.userIdentity = userIdentity

	if err := cst.addContainerToSSHConfig(); err != nil {
		return nil, fmt.Errorf("couldn't add container to ssh_config: %w", err)
	}

	if err := cst.addContainerToKnownHosts(); err != nil {
		return nil, fmt.Errorf("couldn't update known hosts: %w", err)
	}

	return cst, nil
}

func CheckForIncludeWithFS(fs FileSystem) error {
	sketchSSHPathInclude := "Include " + filepath.Join(os.Getenv("HOME"), ".config", "sketch", "ssh_config")
	defaultSSHPath := filepath.Join(os.Getenv("HOME"), ".ssh", "config")

	// Read the existing SSH config file
	existingContent, err := fs.ReadFile(defaultSSHPath)
	if err != nil {
		// If the file doesn't exist, create a new one with just the include line
		if os.IsNotExist(err) {
			return fs.SafeWriteFile(defaultSSHPath, []byte(sketchSSHPathInclude+"\n"), 0o644)
		}
		return fmt.Errorf("⚠️  SSH connections are disabled. cannot open SSH config file: %s: %w", defaultSSHPath, err)
	}

	// Parse the config file
	cfg, err := ssh_config.Decode(bytes.NewReader(existingContent))
	if err != nil {
		return fmt.Errorf("couldn't decode ssh_config: %w", err)
	}

	var sketchInludePos *ssh_config.Position
	var firstNonIncludePos *ssh_config.Position
	for _, host := range cfg.Hosts {
		for _, node := range host.Nodes {
			inc, ok := node.(*ssh_config.Include)
			if ok {
				if strings.TrimSpace(inc.String()) == sketchSSHPathInclude {
					pos := inc.Pos()
					sketchInludePos = &pos
				}
			} else if firstNonIncludePos == nil && !strings.HasPrefix(strings.TrimSpace(node.String()), "#") {
				pos := node.Pos()
				firstNonIncludePos = &pos
			}
		}
	}

	if sketchInludePos == nil {
		// Include line not found, add it to the top of the file
		cfgBytes, err := cfg.MarshalText()
		if err != nil {
			return fmt.Errorf("couldn't marshal ssh_config: %w", err)
		}

		// Add the include line to the beginning
		cfgBytes = append([]byte(sketchSSHPathInclude+"\n"), cfgBytes...)

		// Safely write the updated config back to the file
		if err := fs.SafeWriteFile(defaultSSHPath, cfgBytes, 0o644); err != nil {
			return fmt.Errorf("couldn't safely write ssh_config: %w", err)
		}
		return nil
	}

	if firstNonIncludePos != nil && firstNonIncludePos.Line < sketchInludePos.Line {
		fmt.Printf("⚠️  SSH confg warning: the location of the Include statement for sketch's ssh config on line %d of %s may prevent ssh from working with sketch containers. try moving it to the top of the file (before any 'Host' lines) if ssh isn't working for you.\n", sketchInludePos.Line, defaultSSHPath)
	}
	return nil
}

func removeFromHosts(cntrName string, cfgHosts []*ssh_config.Host) []*ssh_config.Host {
	hosts := []*ssh_config.Host{}
	for _, host := range cfgHosts {
		if host.Matches(cntrName) || strings.Contains(host.String(), cntrName) {
			continue
		}
		patMatch := false
		for _, pat := range host.Patterns {
			if strings.Contains(pat.String(), cntrName) {
				patMatch = true
			}
		}
		if patMatch {
			continue
		}

		hosts = append(hosts, host)
	}
	return hosts
}

func encodePrivateKeyToPEM(privateKey *rsa.PrivateKey) []byte {
	pemBlock := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(privateKey),
	}
	pemBytes := pem.EncodeToMemory(pemBlock)
	return pemBytes
}

func (c *SSHTheater) writeKeyToFile(keyBytes []byte, filename string) error {
	err := c.fs.WriteFile(filename, keyBytes, 0o600)
	return err
}

func (c *SSHTheater) createKeyPairIfMissing(idPath string) (ssh.PublicKey, error) {
	if _, err := c.fs.Stat(idPath); err == nil {
		return nil, nil
	}

	privateKey, err := c.kg.GeneratePrivateKey(keyBitSize)
	if err != nil {
		return nil, fmt.Errorf("error generating private key: %w", err)
	}

	publicRsaKey, err := c.kg.GeneratePublicKey(&privateKey.PublicKey)
	if err != nil {
		return nil, fmt.Errorf("error generating public key: %w", err)
	}

	privateKeyPEM := encodePrivateKeyToPEM(privateKey)

	err = c.writeKeyToFile(privateKeyPEM, idPath)
	if err != nil {
		return nil, fmt.Errorf("error writing private key to file %w", err)
	}
	pubKeyBytes := ssh.MarshalAuthorizedKey(publicRsaKey)

	err = c.writeKeyToFile([]byte(pubKeyBytes), idPath+".pub")
	if err != nil {
		return nil, fmt.Errorf("error writing public key to file %w", err)
	}
	return publicRsaKey, nil
}

func (c *SSHTheater) addSketchHostMatchIfMissing(cfg *ssh_config.Config) error {
	found := false
	for _, host := range cfg.Hosts {
		if strings.Contains(host.String(), "host=\"sketch-*\"") {
			found = true
			break
		}
	}
	if !found {
		hostPattern, err := ssh_config.NewPattern("host=\"sketch-*\"")
		if err != nil {
			return fmt.Errorf("couldn't add pattern to ssh_config: %w", err)
		}

		hostCfg := &ssh_config.Host{Patterns: []*ssh_config.Pattern{hostPattern}}
		hostCfg.Nodes = append(hostCfg.Nodes, &ssh_config.KV{Key: "UserKnownHostsFile", Value: c.knownHostsPath})

		hostCfg.Nodes = append(hostCfg.Nodes, &ssh_config.KV{Key: "IdentityFile", Value: c.userIdentityPath})
		hostCfg.Nodes = append(hostCfg.Nodes, &ssh_config.Empty{})

		cfg.Hosts = append([]*ssh_config.Host{hostCfg}, cfg.Hosts...)
	}
	return nil
}

func (c *SSHTheater) addContainerToSSHConfig() error {
	// Read the existing file contents or start with an empty config if file doesn't exist
	var configData []byte
	var cfg *ssh_config.Config
	var err error

	configData, err = c.fs.ReadFile(c.sshConfigPath)
	if err != nil {
		// If the file doesn't exist, create an empty config
		if os.IsNotExist(err) {
			cfg = &ssh_config.Config{}
		} else {
			return fmt.Errorf("couldn't read ssh_config: %w", err)
		}
	} else {
		// Parse the existing config
		cfg, err = ssh_config.Decode(bytes.NewReader(configData))
		if err != nil {
			return fmt.Errorf("couldn't decode ssh_config: %w", err)
		}
	}

	cntrPattern, err := ssh_config.NewPattern(c.cntrName)
	if err != nil {
		return fmt.Errorf("couldn't add pattern to ssh_config: %w", err)
	}

	// Remove any matches for this container if they already exist.
	cfg.Hosts = removeFromHosts(c.cntrName, cfg.Hosts)

	hostCfg := &ssh_config.Host{Patterns: []*ssh_config.Pattern{cntrPattern}}
	hostCfg.Nodes = append(hostCfg.Nodes, &ssh_config.KV{Key: "HostName", Value: c.sshHost})
	hostCfg.Nodes = append(hostCfg.Nodes, &ssh_config.KV{Key: "User", Value: "root"})
	hostCfg.Nodes = append(hostCfg.Nodes, &ssh_config.KV{Key: "Port", Value: c.sshPort})
	hostCfg.Nodes = append(hostCfg.Nodes, &ssh_config.KV{Key: "IdentityFile", Value: c.userIdentityPath})
	hostCfg.Nodes = append(hostCfg.Nodes, &ssh_config.KV{Key: "UserKnownHostsFile", Value: c.knownHostsPath})

	hostCfg.Nodes = append(hostCfg.Nodes, &ssh_config.Empty{})
	cfg.Hosts = append(cfg.Hosts, hostCfg)

	if err := c.addSketchHostMatchIfMissing(cfg); err != nil {
		return fmt.Errorf("couldn't add missing host match: %w", err)
	}

	cfgBytes, err := cfg.MarshalText()
	if err != nil {
		return fmt.Errorf("couldn't marshal ssh_config: %w", err)
	}

	// Safely write the updated configuration to file
	if err := c.fs.SafeWriteFile(c.sshConfigPath, cfgBytes, 0o644); err != nil {
		return fmt.Errorf("couldn't safely write ssh_config: %w", err)
	}

	return nil
}

func (c *SSHTheater) addContainerToKnownHosts() error {
	pkBytes := c.serverPublicKey.Marshal()
	if len(pkBytes) == 0 {
		return fmt.Errorf("empty serverPublicKey, this is a bug")
	}
	newHostLine := knownhosts.Line([]string{c.sshHost + ":" + c.sshPort}, c.serverPublicKey)

	// Read existing known_hosts content or start with empty if the file doesn't exist
	outputLines := []string{}
	existingContent, err := c.fs.ReadFile(c.knownHostsPath)
	if err == nil {
		scanner := bufio.NewScanner(bytes.NewReader(existingContent))
		for scanner.Scan() {
			outputLines = append(outputLines, scanner.Text())
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("couldn't read known_hosts file: %w", err)
	}

	// Add the new host line
	outputLines = append(outputLines, newHostLine)

	// Safely write the updated content to the file
	if err := c.fs.SafeWriteFile(c.knownHostsPath, []byte(strings.Join(outputLines, "\n")), 0o644); err != nil {
		return fmt.Errorf("couldn't safely write updated known_hosts to %s: %w", c.knownHostsPath, err)
	}

	return nil
}

func (c *SSHTheater) removeContainerFromKnownHosts() error {
	// Read the existing known_hosts file
	existingContent, err := c.fs.ReadFile(c.knownHostsPath)
	if err != nil {
		// If the file doesn't exist, there's nothing to do
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("couldn't read known_hosts file: %w", err)
	}

	// Line we want to remove
	lineToRemove := knownhosts.Line([]string{c.sshHost + ":" + c.sshPort}, c.serverPublicKey)

	// Filter out the line we want to remove
	outputLines := []string{}
	scanner := bufio.NewScanner(bytes.NewReader(existingContent))
	for scanner.Scan() {
		if scanner.Text() == lineToRemove {
			continue
		}
		outputLines = append(outputLines, scanner.Text())
	}

	// Safely write the updated content back to the file
	if err := c.fs.SafeWriteFile(c.knownHostsPath, []byte(strings.Join(outputLines, "\n")), 0o644); err != nil {
		return fmt.Errorf("couldn't safely write updated known_hosts to %s: %w", c.knownHostsPath, err)
	}

	return nil
}

func (c *SSHTheater) Cleanup() error {
	if err := c.removeContainerFromSSHConfig(); err != nil {
		return fmt.Errorf("couldn't remove container from ssh_config: %v\n", err)
	}
	if err := c.removeContainerFromKnownHosts(); err != nil {
		return fmt.Errorf("couldn't remove container from ssh_config: %v\n", err)
	}

	return nil
}

func (c *SSHTheater) removeContainerFromSSHConfig() error {
	// Read the existing file contents
	configData, err := c.fs.ReadFile(c.sshConfigPath)
	if err != nil {
		return fmt.Errorf("couldn't read ssh_config: %w", err)
	}

	cfg, err := ssh_config.Decode(bytes.NewReader(configData))
	if err != nil {
		return fmt.Errorf("couldn't decode ssh_config: %w", err)
	}
	cfg.Hosts = removeFromHosts(c.cntrName, cfg.Hosts)

	if err := c.addSketchHostMatchIfMissing(cfg); err != nil {
		return fmt.Errorf("couldn't add missing host match: %w", err)
	}

	cfgBytes, err := cfg.MarshalText()
	if err != nil {
		return fmt.Errorf("couldn't marshal ssh_config: %w", err)
	}

	// Safely write the updated configuration to file
	if err := c.fs.SafeWriteFile(c.sshConfigPath, cfgBytes, 0o644); err != nil {
		return fmt.Errorf("couldn't safely write ssh_config: %w", err)
	}
	return nil
}

// FileSystem represents a filesystem interface for testability
type FileSystem interface {
	Stat(name string) (fs.FileInfo, error)
	Mkdir(name string, perm fs.FileMode) error
	MkdirAll(name string, perm fs.FileMode) error
	ReadFile(name string) ([]byte, error)
	WriteFile(name string, data []byte, perm fs.FileMode) error
	OpenFile(name string, flag int, perm fs.FileMode) (*os.File, error)
	TempFile(dir, pattern string) (*os.File, error)
	Rename(oldpath, newpath string) error
	SafeWriteFile(name string, data []byte, perm fs.FileMode) error
}

func (fs *RealFileSystem) MkdirAll(name string, perm fs.FileMode) error {
	return os.MkdirAll(name, perm)
}

// RealFileSystem is the default implementation of FileSystem that uses the OS
type RealFileSystem struct{}

func (fs *RealFileSystem) Stat(name string) (fs.FileInfo, error) {
	return os.Stat(name)
}

func (fs *RealFileSystem) Mkdir(name string, perm fs.FileMode) error {
	return os.Mkdir(name, perm)
}

func (fs *RealFileSystem) ReadFile(name string) ([]byte, error) {
	return os.ReadFile(name)
}

func (fs *RealFileSystem) WriteFile(name string, data []byte, perm fs.FileMode) error {
	return os.WriteFile(name, data, perm)
}

func (fs *RealFileSystem) OpenFile(name string, flag int, perm fs.FileMode) (*os.File, error) {
	return os.OpenFile(name, flag, perm)
}

func (fs *RealFileSystem) TempFile(dir, pattern string) (*os.File, error) {
	return os.CreateTemp(dir, pattern)
}

func (fs *RealFileSystem) Rename(oldpath, newpath string) error {
	return os.Rename(oldpath, newpath)
}

// SafeWriteFile writes data to a temporary file, syncs to disk, creates a backup of the existing file if it exists,
// and then renames the temporary file to the target file name.
func (fs *RealFileSystem) SafeWriteFile(name string, data []byte, perm fs.FileMode) error {
	// Get the directory from the target filename
	dir := filepath.Dir(name)

	// Create a temporary file in the same directory
	tmpFile, err := fs.TempFile(dir, filepath.Base(name)+".*.tmp")
	if err != nil {
		return fmt.Errorf("couldn't create temporary file: %w", err)
	}
	tmpFilename := tmpFile.Name()
	defer os.Remove(tmpFilename) // Clean up if we fail

	// Write data to the temporary file
	if _, err := tmpFile.Write(data); err != nil {
		tmpFile.Close()
		return fmt.Errorf("couldn't write to temporary file: %w", err)
	}

	// Sync to disk to ensure data is written
	if err := tmpFile.Sync(); err != nil {
		tmpFile.Close()
		return fmt.Errorf("couldn't sync temporary file: %w", err)
	}

	// Close the temporary file
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("couldn't close temporary file: %w", err)
	}

	// If the original file exists, create a backup
	if _, err := fs.Stat(name); err == nil {
		backupName := name + ".bak"
		// Remove any existing backup
		_ = os.Remove(backupName) // Ignore errors if the backup doesn't exist

		// Create the backup
		if err := fs.Rename(name, backupName); err != nil {
			return fmt.Errorf("couldn't create backup file: %w", err)
		}
	}

	// Rename the temporary file to the target file
	if err := fs.Rename(tmpFilename, name); err != nil {
		return fmt.Errorf("couldn't rename temporary file to target: %w", err)
	}

	// Set permissions on the new file
	if err := os.Chmod(name, perm); err != nil {
		return fmt.Errorf("couldn't set permissions on file: %w", err)
	}

	return nil
}

// KeyGenerator represents an interface for generating SSH keys for testability
type KeyGenerator interface {
	GeneratePrivateKey(bitSize int) (*rsa.PrivateKey, error)
	GeneratePublicKey(privateKey *rsa.PublicKey) (ssh.PublicKey, error)
}

// RealKeyGenerator is the default implementation of KeyGenerator
type RealKeyGenerator struct{}

func (kg *RealKeyGenerator) GeneratePrivateKey(bitSize int) (*rsa.PrivateKey, error) {
	return rsa.GenerateKey(rand.Reader, bitSize)
}

func (kg *RealKeyGenerator) GeneratePublicKey(privateKey *rsa.PublicKey) (ssh.PublicKey, error) {
	return ssh.NewPublicKey(privateKey)
}

// CheckForInclude checks if the user's SSH config includes the Sketch SSH config file
func CheckForInclude() error {
	return CheckForIncludeWithFS(&RealFileSystem{})
}

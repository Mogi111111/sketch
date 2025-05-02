package skabandclient

import (
	"bufio"
	"context"
	"crypto/ed25519"
	crand "crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"golang.org/x/net/http2"
)

// DialAndServeLoop is a redial loop around DialAndServe.
func DialAndServeLoop(ctx context.Context, skabandAddr, sessionID, clientPubKey string, srv http.Handler, connectFn func(connected bool)) {
	if _, err := os.Stat("/.dockerenv"); err == nil { // inDocker
		if addr, err := LocalhostToDockerInternal(skabandAddr); err == nil {
			skabandAddr = addr
		}
	}

	var skabandConnected atomic.Bool
	skabandHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/skabandinit" {
			b, err := io.ReadAll(r.Body)
			if err != nil {
				fmt.Printf("skabandinit failed: %v\n", err)
				return
			}
			m := map[string]string{}
			if err := json.Unmarshal(b, &m); err != nil {
				fmt.Printf("skabandinit failed: %v\n", err)
				return
			}
			skabandConnected.Store(true)
			if connectFn != nil {
				connectFn(true)
			}
			return
		}
		srv.ServeHTTP(w, r)
	})

	var lastErrLog time.Time
	for {
		if err := DialAndServe(ctx, skabandAddr, sessionID, clientPubKey, skabandHandler); err != nil {
			// NOTE: *just* backoff the logging. Backing off dialing
			// is bad UX. Doing so saves negligble CPU and doing so
			// without huring UX requires interrupting the backoff with
			// wake-from-sleep and network-up events from the OS,
			// which are a pain to plumb.
			if time.Since(lastErrLog) > 1*time.Minute {
				slog.DebugContext(ctx, "skaband connection failed", "err", err)
				lastErrLog = time.Now()
			}
		}
		if skabandConnected.CompareAndSwap(true, false) {
			if connectFn != nil {
				connectFn(false)
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func DialAndServe(ctx context.Context, hostURL, sessionID, clientPubKey string, h http.Handler) (err error) {
	// Connect to the server.
	var conn net.Conn
	if strings.HasPrefix(hostURL, "https://") {
		u, err := url.Parse(hostURL)
		if err != nil {
			return err
		}
		port := u.Port()
		if port == "" {
			port = "443"
		}
		dialer := tls.Dialer{}
		conn, err = dialer.DialContext(ctx, "tcp4", u.Host+":"+port)
	} else if strings.HasPrefix(hostURL, "http://") {
		dialer := net.Dialer{}
		conn, err = dialer.DialContext(ctx, "tcp4", strings.TrimPrefix(hostURL, "http://"))
	} else {
		return fmt.Errorf("skabandclient.Dial: bad url, needs to be http or https: %s", hostURL)
	}
	if err != nil {
		return fmt.Errorf("skabandclient: %w", err)
	}
	if conn == nil {
		return fmt.Errorf("skabandclient: nil connection")
	}
	defer conn.Close()

	// "Upgrade" our connection, like a WebSocket does.
	req, err := http.NewRequest("POST", hostURL+"/attach", nil)
	if err != nil {
		return fmt.Errorf("skabandclient.Dial: /attach: %w", err)
	}
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "ska")
	req.Header.Set("Session-ID", sessionID)
	req.Header.Set("Public-Key", clientPubKey)

	if err := req.Write(conn); err != nil {
		return fmt.Errorf("skabandclient.Dial: write upgrade request: %w", err)
	}
	reader := bufio.NewReader(conn)
	resp, err := http.ReadResponse(reader, req)
	if err != nil {
		if resp != nil {
			b, _ := io.ReadAll(resp.Body)
			return fmt.Errorf("skabandclient.Dial: read upgrade response: %w: %s", err, b)
		} else {
			return fmt.Errorf("skabandclient.Dial: read upgrade response: %w", err)
		}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusSwitchingProtocols {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("skabandclient.Dial: unexpected status code: %d: %s", resp.StatusCode, b)
	}
	if !strings.Contains(resp.Header.Get("Upgrade"), "ska") {
		return errors.New("skabandclient.Dial: server did not upgrade to ska protocol")
	}
	if buf := reader.Buffered(); buf > 0 {
		peek, _ := reader.Peek(buf)
		return fmt.Errorf("skabandclient.Dial: buffered read after upgrade response: %d: %q", buf, string(peek))
	}

	// Send Magic.
	const magic = "skaband\n"
	if _, err := conn.Write([]byte(magic)); err != nil {
		return fmt.Errorf("skabandclient.Dial: failed to send upgrade init message: %w", err)
	}

	// We have a TCP connection to the server and have been through the upgrade dance.
	// Now we can run an HTTP server over that connection ("inverting" the HTTP flow).
	server := &http2.Server{}
	server.ServeConn(conn, &http2.ServeConnOpts{
		Handler: h,
	})

	return nil
}

func decodePrivKey(privData []byte) (ed25519.PrivateKey, error) {
	privBlock, _ := pem.Decode(privData)
	if privBlock == nil || privBlock.Type != "PRIVATE KEY" {
		return nil, fmt.Errorf("no valid private key block found")
	}
	parsedPriv, err := x509.ParsePKCS8PrivateKey(privBlock.Bytes)
	if err != nil {
		return nil, err
	}
	return parsedPriv.(ed25519.PrivateKey), nil
}

func encodePrivateKey(privKey ed25519.PrivateKey) ([]byte, error) {
	privBytes, err := x509.MarshalPKCS8PrivateKey(privKey)
	if err != nil {
		return nil, err
	}
	return pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: privBytes}), nil
}

func LoadOrCreatePrivateKey(path string) (ed25519.PrivateKey, error) {
	privData, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		_, privKey, err := ed25519.GenerateKey(crand.Reader)
		if err != nil {
			return nil, err
		}
		b, err := encodePrivateKey(privKey)
		if err := os.WriteFile(path, b, 0o600); err != nil {
			return nil, err
		}
		return privKey, nil
	} else if err != nil {
		return nil, fmt.Errorf("read key failed: %w", err)
	}
	key, err := decodePrivKey(privData)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return key, nil
}

func Login(stdout io.Writer, privKey ed25519.PrivateKey, skabandAddr, sessionID string) (pubKey, apiURL, apiKey string, err error) {
	sig := ed25519.Sign(privKey, []byte(sessionID))

	req, err := http.NewRequest("POST", skabandAddr+"/authclient", nil)
	if err != nil {
		return "", "", "", err
	}
	pubKey = hex.EncodeToString(privKey.Public().(ed25519.PublicKey))
	req.Header.Set("Public-Key", pubKey)
	req.Header.Set("Session-ID", sessionID)
	req.Header.Set("Session-ID-Sig", hex.EncodeToString(sig))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", "", "", fmt.Errorf("skaband login: %w", err)
	}
	apiURL = resp.Header.Get("X-API-URL")
	apiKey = resp.Header.Get("X-API-Key")
	defer resp.Body.Close()
	_, err = io.Copy(stdout, resp.Body)
	if err != nil {
		return "", "", "", fmt.Errorf("skaband login: %w", err)
	}
	if resp.StatusCode != 200 {
		return "", "", "", fmt.Errorf("skaband login failed: %d", resp.StatusCode)
	}
	if apiURL == "" {
		return "", "", "", fmt.Errorf("skaband returned no api url")
	}
	if apiKey == "" {
		return "", "", "", fmt.Errorf("skaband returned no api key")
	}
	return pubKey, apiURL, apiKey, nil
}

func DefaultKeyPath() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	cacheDir := filepath.Join(homeDir, ".cache", "sketch")
	os.MkdirAll(cacheDir, 0o777)
	return filepath.Join(cacheDir, "sketch.ed25519")
}

func LocalhostToDockerInternal(skabandURL string) (string, error) {
	u, err := url.Parse(skabandURL)
	if err != nil {
		return "", fmt.Errorf("localhostToDockerInternal: %w", err)
	}
	switch u.Hostname() {
	case "localhost", "127.0.0.1":
		host := "host.docker.internal"
		if port := u.Port(); port != "" {
			host += ":" + port
		}
		u.Host = host
		return u.String(), nil
	}
	return skabandURL, nil
}

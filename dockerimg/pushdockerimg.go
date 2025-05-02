//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"sketch.dev/dockerimg"
)

func main() {
	dir, err := os.MkdirTemp("", "sketch-pushdockerimg-*")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)

	name, dockerfile, tag := dockerimg.DefaultImage()
	if err := os.WriteFile(filepath.Join(dir, "Dockerfile"), []byte(dockerfile), 0o666); err != nil {
		panic(err)
	}

	fmt.Print(`NOTE: this requires:
	brew install colima docker docker-buildx qemu
	gh auth token | docker login ghcr.io -u $(gh api user --jq .login) --password-stdin
`)

	run := func(args ...string) {
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Dir = dir
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		fmt.Printf("running %v\n", cmd.Args)
		if err := cmd.Run(); err != nil {
			panic(err)
		}
	}

	path := name + ":" + tag

	run("colima", "start")
	run("docker", "buildx", "create", "--name", "arm", "--use", "--driver", "docker-container", "--bootstrap")
	run("docker", "buildx", "use", "arm")
	run("docker", "buildx", "build", "--platform", "linux/arm64", "-t", path+"arm64", "--push", ".")
	run("docker", "buildx", "rm", "arm")
	run("colima", "start", "--profile=intel", "--arch=x86_64", "--vm-type=vz", "--vz-rosetta", "--memory=4", "--disk=15")
	run("docker", "context", "use", "colima-intel")
	run("docker", "buildx", "create", "--name", "intel", "--use", "--driver", "docker-container", "--bootstrap")
	run("docker", "buildx", "use", "intel")
	run("docker", "buildx", "build", "--platform", "linux/amd64", "-t", path+"amd64", "--push", ".")
	run("docker", "buildx", "rm", "intel")
	run("docker", "context", "use", "colima")
	run("colima", "stop", "--profile=intel")
	run(
		"docker", "buildx", "imagetools", "create",
		"-t", path, path+"arm64", path+"amd64",
	)
	run("docker", "buildx", "imagetools", "inspect", path)
}

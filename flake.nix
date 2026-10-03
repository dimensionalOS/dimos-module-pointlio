{
  description = "pointlio-core (Rust Point-LIO) dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cargo
            rustc
            rustfmt
            clippy
            rust-analyzer
          ];

          shellHook = ''
            echo "pointlio-core dev shell ready"
            echo "Build: cd rust && cargo build"
            echo "Test:  cd rust && cargo test"
          '';
        };
      });
}

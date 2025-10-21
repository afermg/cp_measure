{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config.allowUnfree = true;
        };

        mpkgs = import inputs.nixpkgs_master {
          system = system;
          config.allowUnfree = true;
        };

        libList = [
          # Add needed packages here
          pkgs.libz # Numpy
          pkgs.stdenv.cc.cc
          pkgs.libGL
          pkgs.glib
        ];
      in
      with pkgs;
      {
        devShells = {
          default =
            let
              # These packages get built by Nix, and will be ahead on the PATH
              pwp = (
                python313.withPackages (
                  p: with p; [
                    python-lsp-server
                    python-lsp-ruff
                    venvShellHook
                  ]
                )
              );
            in
            mkShell {
              NIX_LD = runCommand "ld.so" { } ''
                ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
              '';
              NIX_LD_LIBRARY_PATH = lib.makeLibraryPath libList;
              packages = [
                pwp
                uv
                pkgs.gcc
              ]
              ++ libList;
              venvDir = "./.venv";
              postVenvCreation = ''
                unset SOURCE_DATE_EPOCH
              '';
              postShellHook = ''
                unset SOURCE_DATE_EPOCH
              '';
              shellHook = ''
                export UV_PYTHON=${pkgs.python313}
                export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring

                uv sync --all-groups
                export PYTHONPATH=${pwp}/${pwp.sitePackages}:$PYTHONPATH
                runHook venvShellHook
                source .venv/bin/activate
              '';
            };
        };
      }
    );
}

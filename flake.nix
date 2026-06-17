{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      git-hooks,
      treefmt-nix,
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

        treefmtEval = treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs.nixfmt.enable = true;
          programs.ruff-format.enable = true;
          programs.ruff-check.enable = true;
          programs.dprint.enable = true;
          programs.dprint.includes = [
            "*.json"
            "*.md"
            "*.yaml"
            "*.yml"
          ];
          programs.dprint.settings = {
            plugins = pkgs.dprint-plugins.getPluginList (
              plugins: with plugins; [
                dprint-plugin-json
                dprint-plugin-markdown
                g-plane-pretty_yaml
              ]
            );
          };
        };

        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          package = pkgs.prek;
          hooks = {
            treefmt = {
              enable = true;
              package = treefmtEval.config.build.wrapper;
            };
          };
        };
      in
      with pkgs;
      {
        checks = {
          inherit pre-commit-check;
          formatting = treefmtEval.config.build.check self;
        };
        formatter = treefmtEval.config.build.wrapper;
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
                gcc
                pwp
                uv
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
                ${pre-commit-check.shellHook}
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

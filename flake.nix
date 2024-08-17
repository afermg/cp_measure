{
  inputs = {
    dream2nix.url = "github:nix-community/dream2nix";
    nixpkgs.follows = "dream2nix/nixpkgs";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  outputs = { self, nixpkgs, devenv, systems, dream2nix, ... } @ inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let

        pkgs = import nixpkgs {
          system = system;
          config.allowUnfree = true;
        };

        mpkgs = import inputs.nixpkgs_master {
          system = system;
          config.allowUnfree = true;
        };

      in  {
        devShells =
          let
            python_with_pkgs = (pkgs.python39.withPackages(pp: []));
          in
            with pkgs;
            {
              default = pkgs.mkShell {
                NIX_LD = runCommand "ld.so" {} ''
                  ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
                '';
                NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
                  pkgs.zlib
                ];
                packages = [
                  mpkgs.rye
                  uv
                ];
                venvDir = "./.venv";
                postVenvCreation = ''
                  unset SOURCE_DATE_EPOCH
                '';
                postShellHook = ''
                  unset SOURCE_DATE_EPOCH
                '';
                shellHook = ''
                  export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:"${pkgs.stdenv.cc.cc.lib}/lib"
                  export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
                  runHook venvShellHook
                  export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
                  rye sync
                  source .venv/bin/activate
              '';
              };
            };
      }
    );
}
